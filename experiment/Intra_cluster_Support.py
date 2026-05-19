# -*- coding: utf-8 -*-
"""
簇内互助 vs 跨簇推断 — 控制变量隔离验证实验 (v2)

固定掩码率，验证族内互相推断比外族推断正确率更高

核心改进：
    保持模型在合理掩码率(~0.4)下工作，仅对比"簇内成员是否可见"   对目标传感器重构精度的影响。

实验设计：
    固定可见传感器总数 = NUM_VISIBLE（约 37 个，对应 mask_ratio≈0.4）。
    对每一个非孤立簇 C 中的每个传感器 S_i：

    实验组（Intra-cluster Support）：
        - 掩码 S_i
        - 保持簇 C 内的其他成员可见（|C|-1 个）
        - 从外部传感器中随机补齐到 NUM_VISIBLE 个可见
        → S_i 的"家人"在可见集中

    对照组（Inter-cluster Support）：
        - 掩码 S_i
        - 簇 C 内的伙伴全部掩码
        - 从外部传感器中随机挑选 NUM_VISIBLE 个可见
        → S_i 的"家人"全部不可见，但总可见数相同

    指标：对 S_i 的重构 1-MSE 和物理准确率
    重复 NUM_REPEATS 次

输出：
    - results/Exp2_IntraCluster_Support.csv
    - results/Exp2_IntraCluster_Support.png

Usage:
    python experiment/Intra_cluster_Support.py
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

# ============ 配置参数 ============
EXCEL_PATH = os.path.join(parent_dir, "data", "SensorData.xlsx")
CHECKPOINT_PATH = os.path.join(parent_dir, "checkpoints", "best.pth")
CLUSTERS_PATH = os.path.join(parent_dir, "results", "inside_clusters.json")
SAVE_DIR = os.path.join(parent_dir, "results")

NUM_SENSORS = 61
D_MODEL = 128
L = 500
BATCH_SIZE = 64
MASK_RATIO_TRAIN = 0.7
LAMBDA_SIGNAL = 1

# 控制可见传感器总数，与训练时 mask_ratio=0.4 一致
NUM_VISIBLE = int(NUM_SENSORS * (1 - MASK_RATIO_TRAIN))  # 37

NUM_REPEATS = 50  # 每个 (簇, 目标传感器) 组合重复次数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============ 模型加载 ============
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


# ============ 单目标传感器推断 ============
def compute_target_metrics(model, data, visible_indices, mask_indices,
                           target_sensor_id, train_mean, train_std):
    """
    对一组 visible / mask 方案执行前向传播，
    仅返回 target_sensor_id 对应的重构指标。

    Returns:
        target_mse:  float, 目标传感器的标准化 MSE
        target_acc:  float, 目标传感器的物理准确率 (1 - relative_error)
    """
    B = BATCH_SIZE
    vis_idx = torch.from_numpy(visible_indices).unsqueeze(0).expand(B, -1).to(device)
    msk_idx = torch.from_numpy(mask_indices).unsqueeze(0).expand(B, -1).to(device)

    tokens, vis_idx_out, msk_idx_out, batch_X = \
        model.temporal_encoder.forward_with_mask(data, vis_idx, msk_idx)
    encoded = model.spatial_encoder(tokens, vis_idx_out)
    decoded = model.spatial_decoder(encoded, vis_idx_out, msk_idx_out)
    signal_out, sequence_out = model.projection_head(decoded)

    y_signal_true = batch_X[:, -1, :].unsqueeze(-1)
    batch_idx = torch.arange(B, device=device).unsqueeze(1)
    pred_signal = signal_out[batch_idx, msk_idx_out]
    true_signal = y_signal_true[batch_idx, msk_idx_out]

    # 找到 target_sensor_id 在 mask_indices 中的位置
    target_pos = int(np.where(mask_indices == target_sensor_id)[0][0])

    pred_target = pred_signal[:, target_pos, 0]  # [B]
    true_target = true_signal[:, target_pos, 0]  # [B]

    # 标准化空间 MSE
    target_mse = torch.nn.functional.mse_loss(pred_target, true_target).item()

    # 物理准确率
    mu = train_mean[target_sensor_id].to(device)
    sigma = train_std[target_sensor_id].to(device)
    pred_phys = pred_target * sigma + mu
    true_phys = true_target * sigma + mu
    error = torch.abs(pred_phys - true_phys) / (torch.abs(mu) + 1e-3)
    target_acc = 1 - error.mean().item()

    return target_mse, target_acc


# ============ 主实验逻辑 ============
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ---- 读取聚类 ----
    if not os.path.exists(CLUSTERS_PATH):
        print(f"[ERROR] 找不到聚类文件 {CLUSTERS_PATH}，请先运行 InsideCo.py")
        return

    with open(CLUSTERS_PATH, 'r') as f:
        clusters_raw = json.load(f)

    # 只保留非孤立簇（size > 1）
    clusters = {}
    for cname, cinfo in clusters_raw.items():
        if cname == "Orphans":
            continue
        if len(cinfo["sensors"]) > 1:
            clusters[cname] = {
                "sensors": cinfo["sensors"],
                "avg_lift": cinfo.get("avg_lift", 0),
            }

    print(f"[INFO] 共 {len(clusters)} 个非孤立簇")
    print(f"[INFO] 固定可见传感器数: {NUM_VISIBLE} (mask_ratio ≈ {1 - NUM_VISIBLE / NUM_SENSORS:.2f})")
    for cname, cinfo in clusters.items():
        sensors = cinfo["sensors"]
        print(f"  {cname}: {[f'S{s}' for s in sensors]}  "
              f"(size={len(sensors)}, lift={cinfo['avg_lift']:.4f})")

    # ---- 加载数据与模型 ----
    print("\n[INFO] 正在加载数据...")
    ds = DataSeperate(file_path=EXCEL_PATH)
    ds.process()
    train_mean = ds.train_mean
    train_std = ds.train_std
    val_data = ds.val_tensor.to(device)
    print(f"[INFO] 验证集: {val_data.shape}")

    model = load_model()

    # ---- 逐簇逐传感器实验 ----
    results = []
    rng = np.random.RandomState(42)

    total_targets = sum(len(c["sensors"]) for c in clusters.values())
    pbar = tqdm(total=total_targets, desc="Controlled Inference Test")

    for cluster_name, cinfo in clusters.items():
        cluster_sensors = cinfo["sensors"]
        cluster_set = set(cluster_sensors)
        cluster_size = len(cluster_sensors)
        num_mates = cluster_size - 1  # 簇内可提供的支援数量

        # 外部候选池：不属于本簇的所有传感器
        external_pool = [s for s in range(NUM_SENSORS) if s not in cluster_set]

        for target_sid in cluster_sensors:
            # 簇内伙伴（不含 target 自身）
            mates = [s for s in cluster_sensors if s != target_sid]

            # ========== 实验组：簇内伙伴可见 ==========
            # 可见 = mates + 从外部随机补齐到 NUM_VISIBLE
            num_external_needed_intra = NUM_VISIBLE - num_mates
            # 确保不超过外部候选池大小
            num_external_needed_intra = min(num_external_needed_intra, len(external_pool))

            mse_intra_list, acc_intra_list = [], []
            for _ in range(NUM_REPEATS):
                ext_sample = rng.choice(external_pool, size=num_external_needed_intra, replace=False)
                visible_intra = np.array(sorted(list(mates) + list(ext_sample)))
                masked_intra = np.array(sorted(
                    [s for s in range(NUM_SENSORS) if s not in visible_intra]
                ))
                with torch.no_grad():
                    mse_val, acc_val = compute_target_metrics(
                        model, val_data, visible_intra, masked_intra,
                        target_sid, train_mean, train_std
                    )
                mse_intra_list.append(mse_val)
                acc_intra_list.append(acc_val)

            # ========== 对照组：簇内伙伴不可见 ==========
            # 可见 = 从外部（排除簇内所有成员）随机挑 NUM_VISIBLE 个
            # 外部候选池此时必须排除整个簇（包括 target）
            external_pool_ctrl = [s for s in range(NUM_SENSORS) if s not in cluster_set]
            num_external_needed_inter = min(NUM_VISIBLE, len(external_pool_ctrl))

            mse_inter_list, acc_inter_list = [], []
            for _ in range(NUM_REPEATS):
                ext_sample = rng.choice(external_pool_ctrl, size=num_external_needed_inter, replace=False)
                visible_inter = np.array(sorted(list(ext_sample)))
                masked_inter = np.array(sorted(
                    [s for s in range(NUM_SENSORS) if s not in visible_inter]
                ))
                with torch.no_grad():
                    mse_val, acc_val = compute_target_metrics(
                        model, val_data, visible_inter, masked_inter,
                        target_sid, train_mean, train_std
                    )
                mse_inter_list.append(mse_val)
                acc_inter_list.append(acc_val)

            # ========== 汇总 ==========
            avg_mse_intra = np.mean(mse_intra_list)
            avg_mse_inter = np.mean(mse_inter_list)
            avg_acc_intra = np.mean(acc_intra_list)
            avg_acc_inter = np.mean(acc_inter_list)

            results.append({
                "cluster": cluster_name,
                "target_sensor": f"S{target_sid}",
                "target_sensor_id": target_sid,
                "cluster_size": cluster_size,
                "num_mates_visible": num_mates,
                "total_visible": NUM_VISIBLE,
                "avg_mse_intra": avg_mse_intra,
                "std_mse_intra": np.std(mse_intra_list),
                "avg_acc_intra": avg_acc_intra,
                "std_acc_intra": np.std(acc_intra_list),
                "avg_1mse_intra": 1 - avg_mse_intra,
                "avg_mse_inter": avg_mse_inter,
                "std_mse_inter": np.std(mse_inter_list),
                "avg_acc_inter": avg_acc_inter,
                "std_acc_inter": np.std(acc_inter_list),
                "avg_1mse_inter": 1 - avg_mse_inter,
                "mse_reduction_pct": (avg_mse_inter - avg_mse_intra) / (avg_mse_inter + 1e-9) * 100,
                "acc_delta": avg_acc_intra - avg_acc_inter,
            })

            pbar.update(1)
            pbar.set_postfix({
                "cluster": cluster_name,
                "target": f"S{target_sid}",
                "intra_1mse": f"{1 - avg_mse_intra:.4f}",
                "inter_1mse": f"{1 - avg_mse_inter:.4f}",
                "delta": f"{avg_mse_inter - avg_mse_intra:+.4f}",
            })

    pbar.close()

    # ---- 保存 CSV ----
    df = pd.DataFrame(results)
    csv_path = os.path.join(SAVE_DIR, "Exp2_IntraCluster_Support.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[INFO] 逐传感器结果已保存至: {csv_path}")

    # ---- 汇总统计 ----
    print_summary(df)

    # ---- 可视化 ----
    plot_results(df)


def print_summary(df):
    """打印汇总统计"""
    print("\n" + "=" * 70)
    print("  Exp2: Intra-cluster Support vs Inter-cluster Support")
    print("  (controlled: same total visible sensors)")
    print("=" * 70)

    # 按簇汇总
    cluster_summary = df.groupby("cluster").agg({
        "avg_mse_intra": "mean",
        "avg_mse_inter": "mean",
        "avg_acc_intra": "mean",
        "avg_acc_inter": "mean",
        "acc_delta": "mean",
        "mse_reduction_pct": "mean",
        "cluster_size": "first",
        "num_mates_visible": "first",
    }).sort_values("acc_delta", ascending=False)

    for idx, row in cluster_summary.iterrows():
        intra_1mse = 1 - row["avg_mse_intra"]
        inter_1mse = 1 - row["avg_mse_inter"]
        tag = "WIN" if row["acc_delta"] > 0 else "LOSE"
        print(f"\n  {idx} (size={int(row['cluster_size'])}, "
              f"mates_visible={int(row['num_mates_visible'])})  [{tag}]")
        print(f"    Intra(family):  1-MSE={intra_1mse:.4f}  PhysAcc={row['avg_acc_intra']:.4f}")
        print(f"    Inter(stranger):1-MSE={inter_1mse:.4f}  PhysAcc={row['avg_acc_inter']:.4f}")
        print(f"    MSE reduction = {row['mse_reduction_pct']:+.1f}%  "
              f"Acc delta = {row['acc_delta']:+.4f}")

    # 全局
    global_intra_1mse = 1 - df["avg_mse_intra"].mean()
    global_inter_1mse = 1 - df["avg_mse_inter"].mean()
    global_intra_acc = df["avg_acc_intra"].mean()
    global_inter_acc = df["avg_acc_inter"].mean()
    win_count = (df["acc_delta"] > 0).sum()

    print(f"\n  {'─' * 50}")
    print(f"  Global Average (n={len(df)} sensors):")
    print(f"    Intra:  1-MSE={global_intra_1mse:.4f}  PhysAcc={global_intra_acc:.4f}")
    print(f"    Inter:  1-MSE={global_inter_1mse:.4f}  PhysAcc={global_inter_acc:.4f}")
    print(f"    Intra wins: {win_count}/{len(df)} ({win_count / len(df) * 100:.1f}%)")
    print("=" * 70)


def plot_results(df):
    """生成可视化：簇级柱状对比 + 逐传感器散点图"""

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # ======== 图 1：簇级柱状对比（使用 1-MSE） ========
    ax = axes[0]

    cluster_summary = df.groupby("cluster").agg({
        "avg_mse_intra": "mean",
        "avg_mse_inter": "mean",
        "cluster_size": "first",
        "acc_delta": "mean",
    }).sort_values("acc_delta", ascending=False)

    cluster_names = cluster_summary.index.tolist()
    x = np.arange(len(cluster_names))
    bar_width = 0.35

    intra_1mse = (1 - cluster_summary["avg_mse_intra"]) * 100
    inter_1mse = (1 - cluster_summary["avg_mse_inter"]) * 100

    ax.bar(x - bar_width / 2, intra_1mse, bar_width,
           label="Intra-cluster (family visible)",
           color="#2ca02c", alpha=0.85, edgecolor="white")
    ax.bar(x + bar_width / 2, inter_1mse, bar_width,
           label="Inter-cluster (strangers only)",
           color="#d62728", alpha=0.85, edgecolor="white")

    # 标注 delta
    for i, cname in enumerate(cluster_names):
        delta = intra_1mse.iloc[i] - inter_1mse.iloc[i]
        color = "#2ca02c" if delta > 0 else "#d62728"
        y_pos = max(intra_1mse.iloc[i], inter_1mse.iloc[i]) + 0.3
        ax.annotate(f"{delta:+.1f}%", xy=(x[i], y_pos),
                    ha="center", fontsize=8, color=color, fontweight="bold")

    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("1 - MSE  (%)", fontsize=12)
    ax.set_title("Cluster-level: Family vs Strangers\n"
                 f"(total visible = {NUM_VISIBLE})", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={int(cluster_summary.loc[c, 'cluster_size'])})"
                        for c in cluster_names], fontsize=8, rotation=45, ha="right")
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # ======== 图 2：逐传感器散点 ========
    ax2 = axes[1]

    intra_scores = (1 - df["avg_mse_intra"]) * 100
    inter_scores = (1 - df["avg_mse_inter"]) * 100
    colors = ["#2ca02c" if d > 0 else "#d62728" for d in (intra_scores - inter_scores)]

    ax2.scatter(inter_scores, intra_scores, c=colors,
                s=60, alpha=0.7, edgecolors="white", linewidth=0.5)

    # 对角线
    all_vals = list(intra_scores) + list(inter_scores)
    lo, hi = min(all_vals) - 1, max(all_vals) + 1
    ax2.plot([lo, hi], [lo, hi], 'k--', alpha=0.4, linewidth=1, label="y = x (tie)")
    ax2.set_xlim(lo, hi)
    ax2.set_ylim(lo, hi)

    for _, row in df.iterrows():
        ax2.annotate(row["target_sensor"],
                     ((1 - row["avg_mse_inter"]) * 100,
                      (1 - row["avg_mse_intra"]) * 100),
                     fontsize=6, alpha=0.7, textcoords="offset points",
                     xytext=(3, 3))

    ax2.set_xlabel("Inter-cluster (strangers) 1-MSE (%)", fontsize=12)
    ax2.set_ylabel("Intra-cluster (family) 1-MSE (%)", fontsize=12)
    ax2.set_title("Per-Sensor: above diagonal = family wins", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, "Exp2_IntraCluster_Support.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] 可视化已保存至: {plot_path}")


if __name__ == "__main__":
    main()
