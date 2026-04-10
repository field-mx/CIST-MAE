# -*- coding: utf-8 -*-
"""
CIST-MAE 训练脚本

Usage:
    python train.py
"""

import os
import time
import torch
from models.dataset import DataSeperate
from models.cist_mae import CIST_MAE


def train():
    # ============ 1. 超参数配置 ============
    # 数据参数
    excel_path = os.path.join("data", "SensorData.xlsx")
    num_sensors = 61

    # 模型参数
    d_model = 128
    L = 500
    Batchsize = 32
    mask_ratio = 0.5

    # 训练参数
    epochs = 100
    learning_rate = 1e-3
    weight_decay = 1e-4
    grad_clip = 1.0

    # 损失权重
    lambda_signal = 1.0
    lambda_sequence = 0.5

    # 保存路径
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # ============ 2. 设备选择 ============
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    # ============ 3. 加载数据 ============
    print("[INFO] 加载数据...")
    ds = DataSeperate(file_path=excel_path)
    ds.process()

    # train_tensor 形状: [D_train, C]，如 [16963, 61]
    train_data = ds.train_tensor.to(device)  # 训练集
    val_data = ds.val_tensor.to(device)      # 验证集

    print(f"[INFO] 训练集: {train_data.shape}")
    print(f"[INFO] 验证集: {val_data.shape}")

    # ============ 4. 构建模型 ============
    model = CIST_MAE(
        num_sensors=num_sensors,
        d_model=d_model,
        L=L,
        Batchsize=Batchsize,
        mask_ratio=mask_ratio,
        # 轻量空间编码器
        enc_heads=4,
        enc_layers=1,
        enc_ffn_dim=64,
        enc_dropout=0.4,
        # 空间解码器
        dec_heads=4,
        dec_layers=2,
        dec_ffn_dim=128,
        dec_dropout=0.4,
        # 损失权重
        lambda_signal=lambda_signal,
        lambda_sequence=lambda_sequence,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] 模型可训练参数量: {num_params:,}")

    # ============ 5. 优化器与学习率调度器 ============
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ============ 6. 训练循环 ============
    best_val_loss = float("inf")
    print(f"[INFO] 开始训练, 共 {epochs} 个 epoch\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # --- 训练阶段 ---
        model.train()
        epoch_loss = 0.0
        
        # 每个 epoch 做多次前向传播，每次随机切不同的窗口
        steps_per_epoch = 50
        for step in range(steps_per_epoch):
            optimizer.zero_grad()
            signal_out, sequence_out, train_loss = model(train_data)
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
            for _ in range(10):  # 验证也做多次取平均，减少随机性
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
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                },
                os.path.join(save_dir, "best.pth"),
            )
            print(f"  -> 保存最优模型 (val_loss={best_val_loss:.6f})")

        # --- 定期保存 ---
        if epoch % 20 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f"epoch_{epoch:03d}.pth"),
            )

    print(f"\n[INFO] 训练完成! 最优验证损失: {best_val_loss:.6f}")


if __name__ == "__main__":
    train()
