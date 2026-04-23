# -*- coding: utf-8 -*-
# 2026.04.23 version 1.0 mask clustering
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
from DataSave import DataSaver


def train():
    # ============ 1. 超参数配置 ============
    # 数据参数
    excel_path = os.path.join("data", "SensorData.xlsx")
    num_sensors = 61
    patience = 100

    # 模型参数
    d_model = 128
    L = 500
    Batchsize = 64 # 最优固定
    mask_ratio = 0.4

    # 训练参数
    # 新参数 = 当前参数 - 学习率 × ( 梯度方向 + weight_decay × 当前参数 )
    epochs = 500
    learning_rate = 1e-3# 学习率
    weight_decay = 5e-2# 正则化 每次更新的时候缩小一点点
    grad_clip = 1.0# 梯度裁剪 防止梯度爆炸

    # 损失权重
    lambda_signal = 1


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
        # 损失权重
        lambda_signal=lambda_signal,
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
    patience = patience          # 连续多少个 epoch 不降就提前停止
    patience_counter = 0
    print(f"[INFO] 开始训练, 共 {epochs} 个 epoch (Early Stopping patience={patience})\n")

    # 创建保存类实例并初始化历史字典
    saver = DataSaver(base_dir="results")
    history = {"epoch": [], "train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ============== 动态难度退火法 ==============
        # 前期用 0.2 让它轻松学会大波形，后期逐渐拉到 1.0 死磕单点精度
        '''
        progress = (epoch - 1) / epochs
        current_lambda = 0.1 + progress * (1.0 - 0.1)
        model.lambda_signal = current_lambda
        model.lambda_sequence = 1.0 - current_lambda
        '''
        # ============================================

        # --- 训练阶段 ---
        model.train()
        epoch_loss = 0.0
        
        # 每个 epoch 做多次前向传播，每次随机切不同的窗口
        steps_per_epoch = 50
        for step in range(steps_per_epoch):
            optimizer.zero_grad()
            # 增加数据扰动
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
            patience_counter = 0  # 重置计数器
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
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[INFO] Early Stopping! 验证损失连续 {patience} 个 epoch 未下降。")
                break

        # --- 定期保存 ---
        if epoch % 20 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f"epoch_{epoch:03d}.pth"),
            )
            
        # 记录每轮数据
        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["lr"].append(lr_now)

    # ============ 保存损失曲线到 CSV ============
    exp_name = saver.get_experiment_name(mask_ratio=mask_ratio, 
                                            d_model=d_model,
                                            Batchsize=Batchsize,
                                            learning_rate=learning_rate,
                                            weight_decay=weight_decay,
                                            grad_clip=grad_clip,
                                            lambda_signal=lambda_signal,
                                            )
    saver.save_loss_to_csv(history, filename=f"loss_{exp_name}")

    print(f"\n=============================================")
    print(f"[INFO] 训练完成! 统计参数如下：")
    print(f"- 训练集大小: {train_data.shape}")
    print(f"- 验证集大小: {val_data.shape}")
    print(f"- 总参数量: {num_params:,}")
    print(f"- D_model: {d_model}")
    print(f"- 编码器配置: Layers={model.spatial_encoder.layers}, Heads={model.spatial_encoder.heads}, FFN={model.spatial_encoder.ffn_dim}, Dropout={model.spatial_encoder.dropout}")
    print(f"- 解码器配置: Layers={model.spatial_decoder.layers}, Heads={model.spatial_decoder.heads}, FFN={model.spatial_decoder.ffn_dim}, Dropout={model.spatial_decoder.dropout}")
    print(f"- 损失权重: Signal={lambda_signal}, Sequence={1-lambda_signal}")
    print(f"- 掩码比例 (Mask Ratio): {mask_ratio}")
    print(f">> 最优验证损失 (Best Val Loss): {best_val_loss:.6f}")
    print(f"=============================================\n")


if __name__ == "__main__":
    train()
