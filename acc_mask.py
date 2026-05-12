# -*- coding: utf-8 -*-
"""
掩码率 vs 预测准确率 实验脚本

实验参数：
  - 横坐标：掩码率 mask_ratio ∈ [0.05, 0.95], step=0.01
  - 纵坐标：预测准确率（1 - 归一化MSE），50次掩码取平均，时序上数据打乱
  - Figure 1：随机掩码
  - Figure 2：有序掩码（从传感器重要度由轻到重依次掩码）
  - 模型：best.pth

保存文件：
  - results/acc_random_mask.csv
  - results/acc_order_mask.csv

Usage:
    python acc_mask.py
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from models.dataset import DataSeperate
from models.cist_mae import CIST_MAE


# ======================== 配置 ========================
EXCEL_PATH = os.path.join("data", "SensorData.xlsx")
CHECKPOINT_PATH = os.path.join("checkpoints", "best.pth")
IMPORTANCE_PATH = os.path.join("results", "sensor_importance.csv")
SAVE_DIR = "results"

NUM_SENSORS = 61
D_MODEL = 128
L = 500
BATCH_SIZE = 64
MASK_RATIO_TRAIN = 0.4          # 模型训练时的 mask_ratio（构建模型用）
LAMBDA_SIGNAL = 1

MASK_RATIO_MIN = 0.05
MASK_RATIO_MAX = 0.95
MASK_RATIO_STEP = 0.01
NUM_REPEATS = 50                # 每个掩码率重复 50 次取平均


def load_model(device):
    """加载 best.pth 模型"""
    model = CIST_MAE(
        num_sensors=NUM_SENSORS,
        d_model=D_MODEL,
        L=L,
        Batchsize=BATCH_SIZE,
        mask_ratio=MASK_RATIO_TRAIN,
        lambda_signal=LAMBDA_SIGNAL,
    ).to(device)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    print(f"[INFO] 模型加载成功: {CHECKPOINT_PATH}")
    return model


def compute_accuracy(model, data, visible_indices, mask_indices, device, train_mean, train_std, save_csv_path=None):
    """
    给定掩码方案，计算预测准确率。
    
    准确率定义：1-（预测值残差/真值）
    
    Args:
        model: CIST_MAE 模型
        data: 验证集数据 [D, C]
        visible_indices: np.array, 可见传感器索引
        mask_indices: np.array, 被掩码传感器索引
        device: torch device
        train_mean: 训练集均值
        train_std: 训练集标准差
    Returns:
        accuracy: 预测准确率 (float)
        mse: 均方误差 (float)
    """
    B = BATCH_SIZE
    vis_idx = torch.from_numpy(visible_indices).unsqueeze(0).expand(B, -1).to(device)
    msk_idx = torch.from_numpy(mask_indices).unsqueeze(0).expand(B, -1).to(device)

    # 手动走一遍模型获取 batch_X 和预测值，保证时间窗是对齐的
    tokens, vis_idx_out, msk_idx_out, batch_X = model.temporal_encoder.forward_with_mask(data, vis_idx, msk_idx)
    encoded = model.spatial_encoder(tokens, vis_idx_out)
    decoded = model.spatial_decoder(encoded, vis_idx_out, msk_idx_out)
    signal_out, sequence_out = model.projection_head(decoded)

    # 计算 MSE loss
    y_signal_true = batch_X[:, -1, :].unsqueeze(-1)
    batch_idx = torch.arange(B, device=device).unsqueeze(1)
    pred_signal = signal_out[batch_idx, msk_idx_out]
    true_signal = y_signal_true[batch_idx, msk_idx_out]
    mse = torch.nn.functional.mse_loss(pred_signal, true_signal).item()

    # 1. 提取预测值和真实值
    pred_std = pred_signal.squeeze(-1) # 得到标准化预测值 [B, num_masked]
    true_std = true_signal.squeeze(-1) # 得到标准化真实值 [B, num_masked]

    # 提取被掩码传感器的均值和方差
    mu = train_mean[mask_indices].to(device)
    sigma = train_std[mask_indices].to(device)
    
    # 反归一化还原物理值
    pred_phys = pred_std * sigma + mu
    true_phys = true_std * sigma + mu
    
    # 计算误差
    # 使用传感器的均值(mu)作为基准计算相对误差，避免个别时刻真实值接近0导致百分比误差爆炸
    error = torch.abs(pred_phys - true_phys) / (torch.abs(mu) )
    accuracy = 1 - error.mean().item()
    
    if save_csv_path:
        # 保存第一个样本 (Batch 0) 的真实值和预测值
        b_idx = 0
        df_save = pd.DataFrame({
            'Sensor_ID': [f"S{idx}" for idx in mask_indices],
            'True_Value': true_phys[b_idx].cpu().numpy(),
            'Pred_Value': pred_phys[b_idx].cpu().numpy(),
            'Abs_Error': torch.abs(pred_phys[b_idx] - true_phys[b_idx]).cpu().numpy()
        })
        df_save.to_csv(save_csv_path, index=False)
        print(f"[INFO] 已保存样本数据至 {save_csv_path}")

    return accuracy, mse


def get_sensor_importance_order():
    """
    从 sensor_importance.csv 读取传感器重要度排序。
    
    返回按重要度从低到高排列的传感器索引列表。
    （最不重要的排在前面，最先被掩码）
    """
    df = pd.read_csv(IMPORTANCE_PATH)
    # sensor_importance.csv 中 sensor_id 格式为 "S0", "S1", ...
    # 文件已按 retain_prob 降序排列（最重要的在前）
    # 我们需要反转：从最不重要到最重要
    sensor_ids = df['sensor_id'].tolist()
    # 提取传感器编号
    sensor_indices = [int(s[1:]) for s in sensor_ids]
    # 反转：从不重要到重要（先掩码不重要的）
    sensor_indices_low_to_high = list(reversed(sensor_indices))
    return sensor_indices_low_to_high


def run_random_mask_experiment(model, data, device, train_mean, train_std):
    """
    实验 1：随机掩码
    
    对每个 mask_ratio，随机生成掩码 50 次，取平均准确率。
    每次掩码时，temporal_encoder 内部会随机切分窗口（时序打乱）。
    """
    print("\n" + "=" * 60)
    print("  实验 1：随机掩码 (Random Mask)")
    print("=" * 60)

    mask_ratios = np.arange(MASK_RATIO_MIN, MASK_RATIO_MAX + 1e-9, MASK_RATIO_STEP)
    mask_ratios = np.round(mask_ratios, 2)

    results = []
    rng = np.random.RandomState(42)

    for mr in tqdm(mask_ratios, desc="随机掩码实验"):
        num_masked = max(1, min(int(NUM_SENSORS * mr), NUM_SENSORS - 1))
        num_visible = NUM_SENSORS - num_masked

        accs = []
        mses = []
        for rep in range(NUM_REPEATS):
            # 每次随机生成掩码
            perm = rng.permutation(NUM_SENSORS)
            mask_indices = perm[:num_masked].copy()
            visible_indices = perm[num_masked:].copy()
            # 排序使索引有序
            mask_indices.sort()
            visible_indices.sort()

            with torch.no_grad():
                # 仅在掩码率为 0.40 的第一次循环时保存一组预测数据样本
                save_path = os.path.join(SAVE_DIR, "sample_predictions_0.40.csv") if (abs(mr - 0.40) < 1e-5 and rep == 0) else None
                acc, mse = compute_accuracy(
                    model, data, visible_indices, mask_indices, device, train_mean, train_std, save_csv_path=save_path
                )
            accs.append(1-mse)
            mses.append(mse)

        avg_acc = np.mean(accs)
        std_acc = np.std(accs)
        avg_mse = np.mean(mses)

        results.append({
            'mask_ratio': mr,
            'num_masked': num_masked,
            'num_visible': num_visible,
            'avg_accuracy': avg_acc,
            'std_accuracy': std_acc,
            'avg_mse': avg_mse,
        })

        if int(mr * 100) % 10 == 0:
            tqdm.write(
                f"  mask_ratio={mr:.2f}  "
                f"num_masked={num_masked:2d}  "
                f"avg_acc={avg_acc:.4f} ± {std_acc:.4f}  "
                f"avg_mse={avg_mse:.6f}"
            )

    df = pd.DataFrame(results)
    save_path = os.path.join(SAVE_DIR, "acc_random_mask.csv")
    df.to_csv(save_path, index=False)
    print(f"\n[INFO] 随机掩码结果已保存至: {save_path}")
    return df


def run_ordered_mask_experiment(model, data, device, train_mean, train_std):
    """
    实验 2：有序掩码（传感器重要度由轻到重依次掩码）
    
    按传感器重要度排序（最不重要 → 最重要），逐步增加掩码传感器数量。
    每个 mask_ratio 对应一个确定性的掩码方案（无随机性），
    但重复 50 次（时序窗口随机切分）取平均以消除时序切分的随机性。
    """
    print("\n" + "=" * 60)
    print("  实验 2：有序掩码 (Ordered Mask, 按重要度由轻到重)")
    print("=" * 60)

    sensor_order = get_sensor_importance_order()
    print(f"[INFO] 传感器掩码顺序 (重要度↑): {sensor_order[:10]}...{sensor_order[-5:]}")

    mask_ratios = np.arange(MASK_RATIO_MIN, MASK_RATIO_MAX + 1e-9, MASK_RATIO_STEP)
    mask_ratios = np.round(mask_ratios, 2)

    results = []

    for mr in tqdm(mask_ratios, desc="有序掩码实验"):
        num_masked = max(1, min(int(NUM_SENSORS * mr), NUM_SENSORS - 1))
        num_visible = NUM_SENSORS - num_masked

        # 有序掩码：从最不重要的传感器开始依次掩码
        mask_indices = np.array(sorted(sensor_order[:num_masked]))
        visible_indices = np.array(sorted(sensor_order[num_masked:]))

        accs = []
        mses = []
        for rep in range(NUM_REPEATS):
            # 重复 50 次：虽然掩码固定，但 temporal_encoder 内部窗口切分是随机的
            with torch.no_grad():
                acc, mse = compute_accuracy(
                    model, data, visible_indices, mask_indices, device, train_mean, train_std
                )
            accs.append(1-mse)
            mses.append(mse)

        avg_acc = np.mean(accs)
        std_acc = np.std(accs)
        avg_mse = np.mean(mses)

        results.append({
            'mask_ratio': mr,
            'num_masked': num_masked,
            'num_visible': num_visible,
            'avg_accuracy': avg_acc,
            'std_accuracy': std_acc,
            'avg_mse': avg_mse,
            'masked_sensors': str(mask_indices.tolist()),
        })

        if int(mr * 100) % 10 == 0:
            tqdm.write(
                f"  mask_ratio={mr:.2f}  "
                f"num_masked={num_masked:2d}  "
                f"avg_acc={avg_acc:.4f} ± {std_acc:.4f}  "
                f"avg_mse={avg_mse:.6f}"
            )

    df = pd.DataFrame(results)
    save_path = os.path.join(SAVE_DIR, "acc_order_mask.csv")
    df.to_csv(save_path, index=False)
    print(f"\n[INFO] 有序掩码结果已保存至: {save_path}")
    return df


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ========= 设备 =========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    # ========= 加载数据 =========
    print("[INFO] 加载数据...")
    ds = DataSeperate(file_path=EXCEL_PATH)
    ds.process()
    # 它们分别保存在：
    # ds.train_mean  <- 形状为 [61] 的张量，保存每个传感器的均值
    # ds.train_std   <- 形状为 [61] 的张量，保存每个传感器的标准差
    train_mean = ds.train_mean
    train_std = ds.train_std
    val_data = ds.val_tensor.to(device)
    print(f"[INFO] 验证集: {val_data.shape}")

    # ========= 加载模型 =========
    model = load_model(device)

    # ========= 实验 1：随机掩码 =========
    df_random = run_random_mask_experiment(model, val_data, device, train_mean, train_std)

    # ========= 实验 2：有序掩码 =========
    df_ordered = run_ordered_mask_experiment(model, val_data, device, train_mean, train_std)

    # ========= 汇总 =========
    print("\n" + "=" * 60)
    print("  实验结果汇总")
    print("=" * 60)
    print(f"\n  随机掩码:")
    print(f"    准确率范围: [{df_random['avg_accuracy'].min():.4f}, {df_random['avg_accuracy'].max():.4f}]")
    print(f"    MSE 范围:   [{df_random['avg_mse'].min():.6f}, {df_random['avg_mse'].max():.6f}]")
    print(f"\n  有序掩码:")
    print(f"    准确率范围: [{df_ordered['avg_accuracy'].min():.4f}, {df_ordered['avg_accuracy'].max():.4f}]")
    print(f"    MSE 范围:   [{df_ordered['avg_mse'].min():.6f}, {df_ordered['avg_mse'].max():.6f}]")
    print(f"\n  结果文件:")
    print(f"    - results/acc_random_mask.csv")
    print(f"    - results/acc_order_mask.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()