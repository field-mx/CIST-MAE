# -*- coding: utf-8 -*-
"""
步骤 C：基于传感器重要性的加权 Loss 微调 (Weighted Loss Fine-tuning)

掩码策略与预训练一致（均匀随机），但 Loss 中对重要传感器的重构误差赋予更高权重，
使模型更加关注关键通道的重构精度。

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
    epochs = 500
    learning_rate = 5e-4  # 比预训练小 10 倍
    weight_decay = 5e-2
    grad_clip = 1.0
    lambda_signal = 1
    patience = 100

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

    # ============ 6. 使用原始均匀随机掩码（不替换） ============
    print("[INFO] 掩码策略: 均匀随机掩码 (与预训练一致)")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] 模型可训练参数量: {num_params:,}")

    # ============ 7. 优化器与调度器 ============
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ============ 8. 构建通道级 Loss 权重 ============
    # 基于 importance 数组构建权重：重要度越高,说明越重构,loss 权重越大，让模型更加关注
    importance_values = torch.tensor(sensor_df.sort_values('sensor_idx')['importance'].values,
                                     dtype=torch.float32).to(device)
    # 归一化为均值=1的权重（确保总 loss 量级不变）
    # channel_weights = importance_values / importance_values.mean()
    channel_weights = importance_values
    print(f"[INFO] 通道 Loss 权重范围: [{channel_weights.min():.3f}, {channel_weights.max():.3f}]")

    def compute_weighted_loss(signal_out, sequence_out, batch_X, mask_indices):
        """
        计算通道加权 MSE Loss。
        重要传感器的重构误差被放大，冗余传感器的误差被缩小。
        """
        B = batch_X.shape[0]
        batch_idx = torch.arange(B, device=batch_X.device).unsqueeze(1)

        # --- 分支一：信号预测损失 (加权) ---
        y_signal_true = batch_X[:, -1, :].unsqueeze(-1)  # [B, C, 1]
        pred_signal = signal_out[batch_idx, mask_indices]  # [B, num_masked, 1]
        true_signal = y_signal_true[batch_idx, mask_indices]  # [B, num_masked, 1]
        # 获取被掩码传感器对应的权重 [B, num_masked]
        mask_weights = channel_weights[mask_indices]  # [B, num_masked]
        # 逐通道加权 MSE
        signal_error = ((pred_signal.squeeze(-1) - true_signal.squeeze(-1)) ** 2) * mask_weights
        loss_signal = signal_error.mean()

        # --- 分支二：时序重构损失 (加权) ---
        y_seq_true = batch_X.permute(0, 2, 1)  # [B, C, L]
        pred_seq = sequence_out[batch_idx, mask_indices]  # [B, num_masked, L]
        true_seq = y_seq_true[batch_idx, mask_indices]  # [B, num_masked, L]
        # 逐通道加权（权重广播到 L 维度）
        seq_error = ((pred_seq - true_seq) ** 2).mean(dim=-1) * mask_weights
        loss_sequence = seq_error.mean()

        # --- 总损失 ---
        loss = lambda_signal * loss_signal + (1 - lambda_signal) * loss_sequence
        return loss

    # 缓存钩子：包装原始 TemporalEncoder.forward，
    # 在每次前向传播后缓存 mask_indices 和 batch_X 供加权 Loss 使用
    _original_te_forward = model.temporal_encoder.forward

    def _caching_forward(x):
        out, visible_indices, mask_indices, batch_X = _original_te_forward(x)
        # 缓存到模型对象上，供 compute_weighted_loss 读取
        model._cached_mask_indices = mask_indices
        model._cached_batch_X = batch_X
        return out, visible_indices, mask_indices, batch_X

    model.temporal_encoder.forward = _caching_forward

    # ============ 9. 微调训练循环 ============
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
            signal_out, sequence_out, _ = model(noisy_train_data)
            # 使用缓存的中间变量计算加权 Loss
            train_loss = compute_weighted_loss(
                signal_out, sequence_out,
                model._cached_batch_X, model._cached_mask_indices
            )
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
                signal_out, sequence_out, _ = model(val_data)
                vl = compute_weighted_loss(
                    signal_out, sequence_out,
                    model._cached_batch_X, model._cached_mask_indices
                )
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
