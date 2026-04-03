# -*- coding: utf-8 -*-
"""
评估指标

提供传感器重构任务的常用评估指标。
"""

import torch
import numpy as np


def mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """均方误差 (Mean Squared Error)"""
    return torch.mean((y_pred - y_true) ** 2).item()


def rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """均方根误差 (Root Mean Squared Error)"""
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()


def mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """平均绝对误差 (Mean Absolute Error)"""
    return torch.mean(torch.abs(y_pred - y_true)).item()


def mape(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> float:
    """平均绝对百分比误差 (Mean Absolute Percentage Error)"""
    return torch.mean(torch.abs((y_true - y_pred) / (y_true.abs() + eps))).item() * 100


def r2_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """决定系数 R²"""
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
    return (1 - ss_res / (ss_tot + 1e-8)).item()


def evaluate_reconstruction(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask_indices: torch.Tensor = None,
) -> dict:
    """
    综合评估重构质量

    Args:
        y_pred: 预测值, shape (B, N, 1)
        y_true: 真实值, shape (B, N, 1)
        mask_indices: 被掩码传感器索引（仅评估掩码位置时使用）
    Returns:
        metrics: 包含各指标的字典
    """
    if mask_indices is not None:
        # 仅评估被掩码位置
        y_pred = torch.gather(
            y_pred, dim=1,
            index=mask_indices.unsqueeze(-1).expand(-1, -1, 1),
        )
        y_true = torch.gather(
            y_true, dim=1,
            index=mask_indices.unsqueeze(-1).expand(-1, -1, 1),
        )

    return {
        "MSE": mse(y_pred, y_true),
        "RMSE": rmse(y_pred, y_true),
        "MAE": mae(y_pred, y_true),
        "MAPE": mape(y_pred, y_true),
        "R2": r2_score(y_pred, y_true),
    }
