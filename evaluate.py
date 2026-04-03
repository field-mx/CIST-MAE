# -*- coding: utf-8 -*-
"""
CIST-MAE 评估 / 推理入口

Usage:
    python evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pth
"""

import argparse

import torch
import yaml
from torch.utils.data import DataLoader, random_split

from data.dataset import MockSensorDataset
from models import CIST_MAE
from utils.metrics import evaluate_reconstruction


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(model, test_loader, device):
    """在测试集上评估模型"""
    model.eval()
    all_metrics = {"MSE": 0, "RMSE": 0, "MAE": 0, "MAPE": 0, "R2": 0}
    num_batches = 0

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        y_hat, loss = model(x, y)

        # 获取掩码索引用于评估（需从 channel_masking 获取）
        _, mask_indices, _ = model.channel_masking(x)
        metrics = evaluate_reconstruction(y_hat, y, mask_indices)

        for k in all_metrics:
            all_metrics[k] += metrics[k]
        num_batches += 1

    # 平均指标
    for k in all_metrics:
        all_metrics[k] /= max(num_batches, 1)

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="CIST-MAE Evaluation")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="配置文件路径"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="模型权重路径"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 构建测试数据 ---
    data_cfg = config["data"]
    dataset = MockSensorDataset(
        num_samples=2000,
        num_sensors=data_cfg["num_sensors"],
        seq_len=data_cfg["seq_len"],
    )
    n_total = len(dataset)
    n_train = int(n_total * data_cfg["train_ratio"])
    n_val = int(n_total * data_cfg["val_ratio"])
    n_test = n_total - n_train - n_val
    _, _, test_set = random_split(dataset, [n_train, n_val, n_test])

    test_loader = DataLoader(
        test_set, batch_size=config["training"]["batch_size"], shuffle=False
    )

    # --- 加载模型 ---
    model = CIST_MAE(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    print(f"[INFO] 已加载模型权重: {args.checkpoint}")

    # --- 评估 ---
    metrics = evaluate(model, test_loader, device)
    print("\n" + "=" * 50)
    print("  CIST-MAE 测试集评估结果")
    print("=" * 50)
    for name, value in metrics.items():
        print(f"  {name:>6s}: {value:.6f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
