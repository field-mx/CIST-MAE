# -*- coding: utf-8 -*-
"""
CIST-MAE 训练入口

Usage:
    python train.py --config configs/default.yaml
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, random_split

from data.dataset import MockSensorDataset, SensorDataset
from models import CIST_MAE
from utils.metrics import evaluate_reconstruction


def set_seed(seed: int):
    """固定随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataloaders(config: dict):
    """构建训练/验证/测试 DataLoader"""
    data_cfg = config["data"]
    train_cfg = config["training"]

    # 使用模拟数据集进行开发调试
    dataset = MockSensorDataset(
        num_samples=2000,
        num_sensors=data_cfg["num_sensors"],
        seq_len=data_cfg["seq_len"],
    )

    # 划分数据集
    n_total = len(dataset)
    n_train = int(n_total * data_cfg["train_ratio"])
    n_val = int(n_total * data_cfg["val_ratio"])
    n_test = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(
        train_set, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_set, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader


def train_one_epoch(model, train_loader, optimizer, device, grad_clip=1.0):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, val_loader, device):
    """验证"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="CIST-MAE Training")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="配置文件路径"
    )
    args = parser.parse_args()

    # --- 加载配置 ---
    config = load_config(args.config)
    train_cfg = config["training"]
    log_cfg = config["logging"]

    set_seed(train_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    # --- 构建数据与模型 ---
    train_loader, val_loader, _ = build_dataloaders(config)
    model = CIST_MAE(config).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] 模型可训练参数量: {num_params:,}")

    # --- 优化器与调度器 ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    scheduler = None
    if train_cfg["lr_scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg["epochs"] - train_cfg["warmup_epochs"]
        )

    # --- 训练循环 ---
    os.makedirs(log_cfg["save_dir"], exist_ok=True)
    best_val_loss = float("inf")

    print(f"[INFO] 开始训练, 共 {train_cfg['epochs']} 个 epoch")
    for epoch in range(1, train_cfg["epochs"] + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, train_cfg["grad_clip"]
        )
        val_loss = validate(model, val_loader, device)

        if scheduler is not None and epoch > train_cfg["warmup_epochs"]:
            scheduler.step()

        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:03d}/{train_cfg['epochs']}] "
            f"Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}  "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}  Time: {elapsed:.1f}s"
        )

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": config,
                },
                os.path.join(log_cfg["save_dir"], "best.pth"),
            )
            print(f"  -> 保存最优模型 (val_loss={val_loss:.6f})")

        # 定期保存
        if epoch % log_cfg["save_every"] == 0:
            torch.save(
                model.state_dict(),
                os.path.join(log_cfg["save_dir"], f"epoch_{epoch:03d}.pth"),
            )

    print(f"[INFO] 训练完成! 最优验证损失: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
