# -*- coding: utf-8 -*-
"""
物理量投射头 (ProjectionHead)

功能：单层无激活函数的 Linear 层，将隐维度 D 投影至物理量标量 1。

输入：(B, N, D)
输出：(B, N, 1)
"""

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """物理量投射头"""

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 解码器输出, shape (B, N, D)
        Returns:
            y_hat: 物理量预测值, shape (B, N, 1)
        """
        return self.proj(x)  # shape: (B, N, D) -> (B, N, 1)
