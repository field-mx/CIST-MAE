# -*- coding: utf-8 -*-
"""
步骤 C：基于先验概率的加权掩码微调 (Weighted Mask Fine-tuning)

加载 Apriori 挖掘的传感器保留概率，替换均匀随机掩码为加权随机掩码，
使用更小学习率微调预训练模型。

Usage:
    python finetune.py
"""

import os
import time
import torch
import numpy as np
import pandas as pd

from models.dataset import DataSeperate
from models.cist_mae import CIST_MAE
from models.weighted_channel_masking import WeightedChannelMasking
from DataSave import DataSaver


def finetune():
    # ============ 1. 超参数配置 ============
    excel_path = os.path.join("data", "SensorData.xlsx")
    checkpoint_path = os.path.join("checkpoints", "best.pth")
    importance_path = os.path.join("results", "sensor_importance.csv")
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # 模型参数（与预训练一致）
    num_sensors = 61
    d_model = 128
    L = 500
    Batchsize = 64
    mask_ratio = 0.4

    # 微调参数
    epochs = 200
    learning_rate = 1e-4  # 比预训练小 10 倍
    weight_decay = 5e-2
    grad_clip = 1.0
    lambda_signal = 1
    patience = 80

    # ============ 2. 设备选择 ============
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    # ============ 3. 加载数据 ============
    print("[INFO] 加载数据...")
    ds = DataSeperate(file_path=excel_path)
    ds.process()
    train_data = ds.train_tensor.to(device)
    val_data = ds.val_tensor.to(device)
    print(f"[INFO] 训练集: {train_data.shape}")
    print(f"[INFO] 验证集: {val_data.shape}")

    # ============ 4. 加载传感器保留概率 ============
    print(f"[INFO] 加载传感器重要性: {importance_path}")
    sensor_df = pd.read_csv(importance_path)
    # 按 sensor_id 排序确保顺序正确 (S0, S1, ..., S60)
    sensor_df['sensor_idx'] = sensor_df['sensor_id'].str.replace('S', '').astype(int)
    sensor_df = sensor_df.sort_values('sensor_idx')
    retain_probs = torch.tensor(sensor_df['retain_prob'].values, dtype=torch.float32)
    
    n_golden = (sensor_df['category'] == 'golden').sum()
    n_redundant = (sensor_df['category'] == 'redundant').sum()
    print(f"[INFO] 黄金传感器: {n_golden} 个, 冗余传感器: {n_redundant} 个")
    print(f"[INFO] 保留概率范围: [{retain_probs.min():.3f}, {retain_probs.max():.3f}]")

    # ============ 5. 构建模型并加载预训练权重 ============
    model = CIST_MAE(
        num_sensors=num_sensors,
        d_model=d_model,
        L=L,
        Batchsize=Batchsize,
        mask_ratio=mask_ratio,
        lambda_signal=lambda_signal,
    ).to(device)

    print(f"[INFO] 加载预训练权重: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    # ============ 6. 替换掩码模块为加权掩码 ============
    print("[INFO] 替换 ChannelMasking -> WeightedChannelMasking")
    weighted_masking = WeightedChannelMasking(
        retain_probs=retain_probs,
        mask_ratio=mask_ratio,
    ).to(device)
    # 替换 TemporalEncoder 内部的掩码生成逻辑
    # 通过猴子补丁(monkey-patch)方式覆盖 forward 中的掩码步骤
    original_forward = model.temporal_encoder.forward

    def weighted_forward(x):
        """用加权掩码替换均匀掩码的 TemporalEncoder forward"""
        D, C = x.shape
        B = model.temporal_encoder.Batchsize
        L_val = model.temporal_encoder.L

        # 1. 批次切分
        if D <= L_val:
            starts = torch.zeros(B, dtype=torch.long)
        else:
            starts = torch.randint(0, D - L_val + 1, (B,))
        batch_X = torch.stack([x[s : s + L_val, :] for s in starts], dim=0)

        # 2. 加权掩码（替换原来的均匀随机掩码）
        visible_indices, mask_indices = weighted_masking(batch_X)
        B_dim, L_dim, C_dim = batch_X.shape
        c_dim = visible_indices.shape[1]
        batch_idx = torch.arange(B_dim, device=x.device).unsqueeze(1)
        x_visible = batch_X[batch_idx, :, visible_indices]

        # 3. 因果卷积
        x_conv_in = x_visible.reshape(B_dim * c_dim, 1, L_dim)
        out = model.temporal_encoder.conv1(x_conv_in)
        out = model.temporal_encoder.relu(out)
        out = model.temporal_encoder.conv2(out)
        out = model.temporal_encoder.relu(out)
        out = model.temporal_encoder.conv3(out)
        out = model.temporal_encoder.relu(out)
        d_conv = out.shape[1]
        L_final = out.shape[2]
        out = out.view(B_dim, c_dim, d_conv, L_final)

        # 4. 加权池化 + 线性映射
        out = model.temporal_encoder.weighted_pool(out)
        out = out.squeeze(-1)
        out = model.temporal_encoder.linear(out)

        return out, visible_indices, mask_indices, batch_X

    model.temporal_encoder.forward = weighted_forward

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] 模型可训练参数量: {num_params:,}")

    # ============ 7. 优化器与调度器 ============
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ============ 8. 微调训练循环 ============
    best_val_loss = float("inf")
    patience_counter = 0
    print(f"\n[INFO] 开始微调, 共 {epochs} 个 epoch (Early Stopping patience={patience})\n")

    saver = DataSaver(base_dir="results")
    history = {"epoch": [], "train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # --- 训练阶段 ---
        model.train()
        epoch_loss = 0.0
        steps_per_epoch = 50
        for step in range(steps_per_epoch):
            optimizer.zero_grad()
            noisy_train_data = train_data + torch.rand_like(train_data) * 0.01
            signal_out, sequence_out, train_loss = model(noisy_train_data)
            train_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += train_loss.item()

        scheduler.step()
        avg_train_loss = epoch_loss / steps_per_epoch

        # --- 验证阶段 ---
        model.eval()
        with torch.no_grad():
            val_losses = []
            for _ in range(10):
                _, _, vl = model(val_data)
                val_losses.append(vl.item())
            avg_val_loss = sum(val_losses) / len(val_losses)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:03d}/{epochs}]  "
            f"Train Loss: {avg_train_loss:.6f}  "
            f"Val Loss: {avg_val_loss:.6f}  "
            f"LR: {lr_now:.2e}  "
            f"Time: {elapsed:.1f}s"
        )

        # --- 保存最优模型 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "retain_probs": retain_probs.cpu().numpy().tolist(),
                },
                os.path.join(save_dir, "finetuned_best.pth"),
            )
            print(f"  -> 保存最优微调模型 (val_loss={best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[INFO] Early Stopping! 验证损失连续 {patience} 个 epoch 未下降。")
                break

        if epoch % 20 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"ft_epoch_{epoch:03d}.pth"))

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["lr"].append(lr_now)

    # ============ 保存损失曲线 ============
    exp_name = saver.get_experiment_name(
        mask_ratio=mask_ratio, d_model=d_model,
        Batchsize=Batchsize, learning_rate=learning_rate,
    )
    saver.save_loss_to_csv(history, filename=f"finetune_loss_{exp_name}")

    print(f"\n{'='*50}")
    print(f"[INFO] 微调完成!")
    print(f"  - 最优验证损失: {best_val_loss:.6f}")
    print(f"  - 模型保存至: checkpoints/finetuned_best.pth")
    print(f"{'='*50}")


if __name__ == "__main__":
    finetune()
