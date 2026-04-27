# -*- coding: utf-8 -*-
"""
步骤 A：掩码探测与 Loss 评估 (Mask Probe)

Usage:
    python mask_probe.py
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from models.dataset import DataSeperate
from models.cist_mae import CIST_MAE


def generate_mask_vectors(N, K, mask_ratio_range=(0.2, 0.8), seed=42):
    rng = np.random.RandomState(seed)
    masks = []
    for k in range(K):
        # mask_ratio = rng.uniform(*mask_ratio_range)
        mask_ratio = 0.4# conform stable mask ratio
        num_masked = max(1, min(int(N * mask_ratio), N - 1))
        perm = rng.permutation(N)
        mask_vector = np.ones(N, dtype=np.int32)
        mask_vector[perm[:num_masked]] = 0
        masks.append({
            'mask_vector': mask_vector,
            'mask_ratio': num_masked / N,
            'visible_indices': np.where(mask_vector == 1)[0],
            'mask_indices': np.where(mask_vector == 0)[0],
        })
    return masks


def probe_single_mask(model, data, visible_indices, mask_indices, batch_size, num_repeats=10):
    device = next(model.parameters()).device
    vis_idx = torch.from_numpy(visible_indices).unsqueeze(0).expand(batch_size, -1).to(device)
    msk_idx = torch.from_numpy(mask_indices).unsqueeze(0).expand(batch_size, -1).to(device)
    losses = []
    for _ in range(num_repeats):
        _, _, loss = model.forward_with_mask(data, vis_idx, msk_idx)
        losses.append(loss.item())
    return np.mean(losses)


def run_mask_probe():
    excel_path = os.path.join("data", "SensorData.xlsx")
    checkpoint_path = os.path.join("checkpoints", "best.pth")
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    num_sensors = 61
    K = 5000
    R = 10
    batch_size = 64
    d_model = 128
    L = 500
    mask_ratio = 0.4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    print("[INFO] 加载数据...")
    ds = DataSeperate(file_path=excel_path)
    ds.process()
    val_data = ds.val_tensor.to(device)
    print(f"[INFO] 验证集: {val_data.shape}")

    print(f"[INFO] 加载预训练模型: {checkpoint_path}")
    model = CIST_MAE(
        num_sensors=num_sensors, d_model=d_model, L=L,
        Batchsize=batch_size, mask_ratio=mask_ratio,
        lambda_signal=1,  # 与 train.py 保持一致
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print("[INFO] 模型加载成功")

    print(f"[INFO] 生成 {K} 组随机掩码向量...")
    masks = generate_mask_vectors(N=num_sensors, K=K)

    print(f"[INFO] 开始掩码探测 (K={K}, R={R})...\n")
    results = []
    with torch.no_grad():
        for k, mask_info in enumerate(tqdm(masks, desc="掩码探测")):
            avg_loss = probe_single_mask(
                model, val_data,
                mask_info['visible_indices'], mask_info['mask_indices'],
                batch_size, R,
            )
            results.append({
                'mask_id': k,
                'mask_ratio': mask_info['mask_ratio'],
                'avg_loss': avg_loss,
                'mask_vector': json.dumps(mask_info['mask_vector'].tolist()),
            })

    df = pd.DataFrame(results)
    save_path = os.path.join(save_dir, "0.4mask_probe_results.csv")
    df.to_csv(save_path, index=False)

    print(f"\n{'='*50}")
    print(f"[INFO] 掩码探测完成!")
    print(f"  - 总掩码组数: {K}")
    print(f"  - Loss 范围: [{df['avg_loss'].min():.6f}, {df['avg_loss'].max():.6f}]")
    print(f"  - Loss 均值: {df['avg_loss'].mean():.6f}")
    print(f"  - 结果已保存至: {save_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    run_mask_probe()
