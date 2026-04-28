# -*- coding: utf-8 -*-
"""
模型对比评估脚本：原始模型 vs 微调模型

在完全相同的掩码组合下，使用相同的等权 MSE Loss 公式，
严格对比两个模型的重构性能差异。

Usage:
    python compare_models.py
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from models.dataset import DataSeperate
from models.cist_mae import CIST_MAE


def load_model(checkpoint_path, device, num_sensors=61, d_model=128, L=500,
               batch_size=64, mask_ratio=0.4):
    """加载模型并返回"""
    model = CIST_MAE(
        num_sensors=num_sensors, d_model=d_model, L=L,
        Batchsize=batch_size, mask_ratio=mask_ratio,
        lambda_signal=1,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def probe_with_mask(model, data, visible_indices, mask_indices, batch_size, num_repeats=10, seed=0):
    """用指定掩码探测模型 Loss（等权 MSE），固定随机种子确保数据窗口一致"""
    device = next(model.parameters()).device
    vis_idx = torch.from_numpy(visible_indices).unsqueeze(0).expand(batch_size, -1).to(device)
    msk_idx = torch.from_numpy(mask_indices).unsqueeze(0).expand(batch_size, -1).to(device)
    losses = []
    for i in range(num_repeats):
        # 固定种子：相同的 seed+i 保证原始模型和微调模型看到相同的数据窗口
        torch.manual_seed(seed + i)
        torch.cuda.manual_seed(seed + i)
        _, _, loss = model.forward_with_mask(data, vis_idx, msk_idx)
        losses.append(loss.item())
    return np.mean(losses)


def main():
    # ============ 配置 ============
    excel_path = os.path.join("data", "SensorData.xlsx")
    original_ckpt = os.path.join("checkpoints", "best.pth")
    finetuned_ckpt = os.path.join("checkpoints", "finetuned_best.pth")
    probe_results_path = os.path.join("results", "0.4mask_probe_results.csv")
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    num_sensors = 61
    batch_size = 64
    d_model = 128
    L = 500
    mask_ratio = 0.4
    num_repeats = 10  # 每组掩码重复探测次数

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    # ============ 加载数据 ============
    print("[INFO] 加载数据...")
    ds = DataSeperate(file_path=excel_path)
    ds.process()
    val_data = ds.val_tensor.to(device)
    print(f"[INFO] 验证集: {val_data.shape}")

    # ============ 加载两个模型 ============
    print(f"\n[INFO] 加载原始模型: {original_ckpt}")
    model_original = load_model(original_ckpt, device, num_sensors, d_model, L, batch_size, mask_ratio)

    print(f"[INFO] 加载微调模型: {finetuned_ckpt}")
    model_finetuned = load_model(finetuned_ckpt, device, num_sensors, d_model, L, batch_size, mask_ratio)

    # ============ 加载 mask_probe 结果，提取掩码组合 ============
    print(f"\n[INFO] 加载掩码探测结果: {probe_results_path}")
    probe_df = pd.read_csv(probe_results_path)
    probe_df = probe_df.sort_values('avg_loss', ascending=False).reset_index(drop=True)

    # 选取关键掩码组合进行对比
    # 1. Top-10 最高 Loss 的掩码（最难推断的情况）
    # 2. Top-10 最低 Loss 的掩码（最容易推断的情况）
    # 3. 中位数附近 10 组（典型情况）
    n_total = len(probe_df)
    mid_start = max(0, n_total // 2 - 5)

    groups = {
        "最难 Top-10 (原始Loss最高)": probe_df.iloc[:10],
        "中等难度 Mid-10": probe_df.iloc[mid_start:mid_start + 10],
        "最易 Bottom-10 (原始Loss最低)": probe_df.iloc[-10:],
    }

    # ============ 对比评估 ============
    print(f"\n{'='*70}")
    print(f"  模型对比评估（等权 MSE Loss，相同掩码组合）")
    print(f"{'='*70}")

    all_results = []

    for group_name, group_df in groups.items():
        print(f"\n--- {group_name} ---")
        original_losses = []
        finetuned_losses = []

        for _, row in tqdm(group_df.iterrows(), total=len(group_df), desc="  探测中"):
            mask_vector = np.array(json.loads(row['mask_vector']))
            visible_indices = np.where(mask_vector == 1)[0]
            mask_indices = np.where(mask_vector == 0)[0]

            with torch.no_grad():
                mask_seed = int(row['mask_id']) * 1000  # 基于 mask_id 生成种子
                loss_orig = probe_with_mask(
                    model_original, val_data, visible_indices, mask_indices,
                    batch_size, num_repeats, seed=mask_seed
                )
                loss_ft = probe_with_mask(
                    model_finetuned, val_data, visible_indices, mask_indices,
                    batch_size, num_repeats, seed=mask_seed  # 相同种子
                )

            original_losses.append(loss_orig)
            finetuned_losses.append(loss_ft)
            all_results.append({
                'group': group_name,
                'mask_id': row['mask_id'],
                'original_loss': loss_orig,
                'finetuned_loss': loss_ft,
                'improvement': loss_orig - loss_ft,
                'improvement_pct': (loss_orig - loss_ft) / loss_orig * 100,
            })

        # 打印该组的统计结果
        orig_mean = np.mean(original_losses)
        ft_mean = np.mean(finetuned_losses)
        orig_max = np.max(original_losses)
        ft_max = np.max(finetuned_losses)

        print(f"  原始模型  → 均值: {orig_mean:.6f}  最大: {orig_max:.6f}")
        print(f"  微调模型  → 均值: {ft_mean:.6f}  最大: {ft_max:.6f}")
        print(f"  均值变化: {orig_mean - ft_mean:+.6f} ({(orig_mean - ft_mean) / orig_mean * 100:+.1f}%)")
        print(f"  最大值变化: {orig_max - ft_max:+.6f} ({(orig_max - ft_max) / orig_max * 100:+.1f}%)")

    # ============ 全量对比（可选，耗时较长） ============
    print(f"\n{'='*70}")
    print(f"  是否进行全量 {n_total} 组掩码对比？(耗时较长)")
    print(f"{'='*70}")

    # 保存结果
    result_df = pd.DataFrame(all_results)
    save_path = os.path.join(save_dir, "model_comparison.csv")
    result_df.to_csv(save_path, index=False)

    # ============ 总结 ============
    print(f"\n{'='*70}")
    print(f"  对比总结")
    print(f"{'='*70}")

    improved = result_df[result_df['improvement'] > 0]
    degraded = result_df[result_df['improvement'] < 0]
    print(f"  总测试掩码组数: {len(result_df)}")
    print(f"  微调后提升的组数: {len(improved)} ({len(improved)/len(result_df)*100:.1f}%)")
    print(f"  微调后退步的组数: {len(degraded)} ({len(degraded)/len(result_df)*100:.1f}%)")
    print(f"  平均 Loss 变化: {result_df['improvement'].mean():+.6f}")
    print(f"  结果已保存至: {save_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
