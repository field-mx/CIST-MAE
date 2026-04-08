# -*- coding: utf-8 -*-
"""
物理量投射头 (ProjectionHead)

功能：单层无激活函数的 Linear 层，将隐维度 d_model 投影至物理量标量 1。

输入：(B, num_sensors, d_model)
输出：(B, num_sensors, 1)
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
            x: 解码器输出, shape ((B, num_sensors, d_model))
        Returns:
            y_hat: 物理量预测值, shape (B, num_sensors, 1)
        """
        return self.proj(x)  # shape: (B, num_sensors, d_model) -> (B, num_sensors, 1)
