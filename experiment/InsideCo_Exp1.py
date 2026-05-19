# -*- coding: utf-8 -*-
"""
内部聚类验证实验：三种掩码策略对比
变化掩码率，验证聚类内推断比

实验设计：
    横坐标：mask_ratio ∈ [0.05, 0.95], step=0.01
    纵坐标：全局预测准确率（物理相对误差）
    
    三组实验：
    - 实验组 (Inter-cluster Mask)：优先在不同簇之间分散掩码，
      尽可能保留每个簇内的完整信息（保护内部冗余）。
    - 对照组 1 (Intra-cluster Mask)：优先团灭整个簇，
      尽可能多地摧毁内部冗余关系。
    - 对照组 2 (Random Mask)：纯随机掩码，作为基线。

输出：
    - results/Exp1_InsideCo_sweep.csv
    - results/Exp1_InsideCo_sweep.png

Usage:
    python experiment/InsideCo_Exp1.py
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============ 路径设置 ============
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.cist_mae import CIST_MAE
from models.dataset import DataSeperate

# ============ 配置参数（与 acc_mask.py 保持一致） ============
EXCEL_PATH = os.path.join(parent_dir, "data", "SensorData.xlsx")
CHECKPOINT_PATH = os.path.join(parent_dir, "checkpoints", "best.pth")
CLUSTERS_PATH = os.path.join(parent_dir, "results", "inside_clusters.json")
SAVE_DIR = os.path.join(parent_dir, "results")

NUM_SENSORS = 61
D_MODEL = 128
L = 500
BATCH_SIZE = 64
MASK_RATIO_TRAIN = 0.4
LAMBDA_SIGNAL = 1

MASK_RATIO_MIN = 0.05
MASK_RATIO_MAX = 0.90
MASK_RATIO_STEP = 0.01
NUM_REPEATS = 50    # 每个掩码率重复次数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
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
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"[INFO] 模型已加载: {CHECKPOINT_PATH}")
    return model


def compute_accuracy(model, data, visible_indices, mask_indices, train_mean, train_std):
    """
    计算全局预测准确率（与 acc_mask.py 完全一致的前向传播逻辑）。
    """
    B = BATCH_SIZE
    vis_idx = torch.from_numpy(visible_indices).unsqueeze(0).expand(B, -1).to(device)
    msk_idx = torch.from_numpy(mask_indices).unsqueeze(0).expand(B, -1).to(device)

    tokens, vis_idx_out, msk_idx_out, batch_X = model.temporal_encoder.forward_with_mask(data, vis_idx, msk_idx)
    encoded = model.spatial_encoder(tokens, vis_idx_out)
    decoded = model.spatial_decoder(encoded, vis_idx_out, msk_idx_out)
    signal_out, sequence_out = model.projection_head(decoded)

    y_signal_true = batch_X[:, -1, :].unsqueeze(-1)
    batch_idx = torch.arange(B, device=device).unsqueeze(1)
    pred_signal = signal_out[batch_idx, msk_idx_out]
    true_signal = y_signal_true[batch_idx, msk_idx_out]
    mse = torch.nn.functional.mse_loss(pred_signal, true_signal).item()

    pred_std = pred_signal.squeeze(-1)
    true_std = true_signal.squeeze(-1)

    mu = train_mean[mask_indices].to(device)
    sigma = train_std[mask_indices].to(device)

    pred_phys = pred_std * sigma + mu
    true_phys = true_std * sigma + mu

    error = torch.abs(pred_phys - true_phys) / (torch.abs(mu) + 1e-3)
    accuracy = 1 - error.mean().item()
    return accuracy, mse


# ============ 三种掩码策略 ============

def generate_inter_cluster_mask(num_masked, clusters_list, rng):
    """
    实验组：跨簇分散掩码。
    从每个簇轮流抽 1 个传感器掩盖，尽可能保留每个簇的内部完整性。
    """
    # clusters_list: list of list, 每个元素是一个簇的传感器 ID 列表
    # 打乱每个簇内部的顺序
    shuffled = [rng.permutation(c).tolist() for c in clusters_list]
    # 记录每个簇当前已被取走多少个
    pointers = [0] * len(shuffled)
    
    masked = []
    while len(masked) < num_masked:
        added_this_round = False
        for i in range(len(shuffled)):
            if len(masked) >= num_masked:
                break
            if pointers[i] < len(shuffled[i]):
                masked.append(shuffled[i][pointers[i]])
                pointers[i] += 1
                added_this_round = True
        if not added_this_round:
            break  # 所有簇都用完了
    
    return np.array(sorted(masked[:num_masked]))


def generate_intra_cluster_mask(num_masked, clusters_list, rng):
    """
    对照组 1：簇内集中掩码（团灭）。
    优先把小簇整个团灭掉，再团灭下一个簇，直到凑够名额。
    """
    # 按簇大小从小到大排序（先团灭小簇）
    sorted_clusters = sorted(clusters_list, key=len)
    # 打乱同大小簇的顺序
    rng.shuffle(sorted_clusters)
    
    masked = []
    for cluster in sorted_clusters:
        if len(masked) >= num_masked:
            break
        # 把这个簇的所有成员加进去
        shuffled_c = rng.permutation(cluster).tolist()
        for s in shuffled_c:
            if len(masked) >= num_masked:
                break
            masked.append(s)
    
    return np.array(sorted(masked[:num_masked]))


def generate_random_mask(num_masked, real_sensors, rng):
    """
    对照组 2：纯随机掩码（仅在有簇的传感器中随机）。
    """
    indices = rng.choice(real_sensors, size=num_masked, replace=False)
    return np.sort(indices)


def main():
    # ========= 读取聚类结果 =========
    if not os.path.exists(CLUSTERS_PATH):
        print(f"[ERROR] 找不到聚类文件 {CLUSTERS_PATH}，请先运行 InsideCo.py")
        return

    with open(CLUSTERS_PATH, 'r') as f:
        clusters_raw = json.load(f)

    # 提取所有真实簇（大小 > 1 的簇）
    clusters_list = []
    real_sensors = []
    for cname, cinfo in clusters_raw.items():
        sensors = cinfo["sensors"]
        if cname != "Orphans" and len(sensors) > 1:
            clusters_list.append(sensors)
            real_sensors.extend(sensors)
    
    num_real_sensors = len(real_sensors)
    print(f"[INFO] 真实聚类数量 (size>1): {len(clusters_list)}，共包含传感器: {num_real_sensors} 个")
    for i, c in enumerate(clusters_list):
        print(f"  簇 {i}: {[f'S{s}' for s in c]}")

    # ========= 加载数据与模型 =========
    print("\n[INFO] 正在加载数据...")
    ds = DataSeperate(file_path=EXCEL_PATH)
    ds.process()
    train_mean = ds.train_mean
    train_std = ds.train_std
    val_data = ds.val_tensor.to(device)
    print(f"[INFO] 验证集: {val_data.shape}")

    model = load_model()

    # ========= 掩码率扫描 =========
    mask_ratios = np.arange(MASK_RATIO_MIN, MASK_RATIO_MAX + 1e-9, MASK_RATIO_STEP)
    
    results = []
    
    print(f"\n[INFO] 开始扫描掩码率 [{MASK_RATIO_MIN}, {MASK_RATIO_MAX}], step={MASK_RATIO_STEP}")
    print(f"       每个掩码率重复 {NUM_REPEATS} 次\n")
    
    for mr in tqdm(mask_ratios, desc="Mask Ratio Sweep"):
        num_masked = max(1, int(round(mr * num_real_sensors)))
        num_masked = min(num_masked, num_real_sensors - 1)  # 至少保留 1 个可见
        
        num_visible = NUM_SENSORS - num_masked
        
        accs_inter, mses_inter = [], []
        accs_intra, mses_intra = [], []
        accs_random, mses_random = [], []
        
        rng = np.random.RandomState(42)
        
        for _ in range(NUM_REPEATS):
            # ---- 实验组：跨簇分散掩码 ----
            mask_inter = generate_inter_cluster_mask(num_masked, clusters_list, rng)
            vis_inter = np.array([i for i in range(NUM_SENSORS) if i not in mask_inter])
            with torch.no_grad():
                acc, mse = compute_accuracy(model, val_data, vis_inter, mask_inter, train_mean, train_std)
            accs_inter.append(1-mse)  # 使用物理相对准确率
            mses_inter.append(mse)
            
            # ---- 对照组 1：簇内团灭掩码 ----
            mask_intra = generate_intra_cluster_mask(num_masked, clusters_list, rng)
            vis_intra = np.array([i for i in range(NUM_SENSORS) if i not in mask_intra])
            with torch.no_grad():
                acc, mse = compute_accuracy(model, val_data, vis_intra, mask_intra, train_mean, train_std)
            accs_intra.append(1-mse)  # 使用物理相对准确率
            mses_intra.append(mse)
            
            # ---- 对照组 2：纯随机掩码 ----
            mask_rand = generate_random_mask(num_masked, real_sensors, rng)
            vis_rand = np.array([i for i in range(NUM_SENSORS) if i not in mask_rand])
            with torch.no_grad():
                acc, mse = compute_accuracy(model, val_data, vis_rand, mask_rand, train_mean, train_std)
            accs_random.append(1-mse)  # 使用物理相对准确率
            mses_random.append(mse)
        
        results.append({
            "mask_ratio": round(mr, 2),
            "num_masked": num_masked,
            "num_visible": num_visible,
            "avg_accuracy_inter": np.mean(accs_inter),
            "std_accuracy_inter": np.std(accs_inter),
            "avg_mse_inter": np.mean(mses_inter),
            "avg_accuracy_intra": np.mean(accs_intra),
            "std_accuracy_intra": np.std(accs_intra),
            "avg_mse_intra": np.mean(mses_intra),
            "avg_accuracy_random": np.mean(accs_random),
            "std_accuracy_random": np.std(accs_random),
            "avg_mse_random": np.mean(mses_random),
        })
    
    # ========= 保存 CSV =========
    df = pd.DataFrame(results)
    os.makedirs(SAVE_DIR, exist_ok=True)
    csv_path = os.path.join(SAVE_DIR, "Exp1_InsideCo_sweep.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[INFO] 数据已保存至: {csv_path}")
    
    # ========= 可视化 =========
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df['mask_ratio'], df['avg_accuracy_inter'] * 100, 
            color='#2ca02c', linewidth=2, label='Inter-cluster Mask (preserve intra-cluster)')
    ax.fill_between(df['mask_ratio'], 
                    (df['avg_accuracy_inter'] - df['std_accuracy_inter']) * 100,
                    (df['avg_accuracy_inter'] + df['std_accuracy_inter']) * 100,
                    color='#2ca02c', alpha=0.15)
    
    ax.plot(df['mask_ratio'], df['avg_accuracy_intra'] * 100,
            color='#d62728', linewidth=2, label='Intra-cluster Mask (wipe out clusters)')
    ax.fill_between(df['mask_ratio'],
                    (df['avg_accuracy_intra'] - df['std_accuracy_intra']) * 100,
                    (df['avg_accuracy_intra'] + df['std_accuracy_intra']) * 100,
                    color='#d62728', alpha=0.15)
    
    ax.plot(df['mask_ratio'], df['avg_accuracy_random'] * 100,
            color='#1f77b4', linewidth=2, linestyle='--', label='Random Mask (baseline)')
    ax.fill_between(df['mask_ratio'],
                    (df['avg_accuracy_random'] - df['std_accuracy_random']) * 100,
                    (df['avg_accuracy_random'] + df['std_accuracy_random']) * 100,
                    color='#1f77b4', alpha=0.15)
    
    ax.set_xlabel('Mask Ratio', fontsize=13)
    ax.set_ylabel('Physical Accuracy (%)', fontsize=13)
    ax.set_title('InsideCo Verification: Three Masking Strategies Comparison', fontsize=14)
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(MASK_RATIO_MIN, MASK_RATIO_MAX)
    
    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, "Exp1_InsideCo_sweep.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] 对比曲线图已保存至: {plot_path}")


if __name__ == "__main__":
    main()
