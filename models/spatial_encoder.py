# -*- coding: utf-8 -*-
"""
重型空间编码器 (SpatialEncoder)

功能：接收可见传感器的时序特征 Token，添加空间位置编码后，
     通过多层 Transformer Encoder 捕获传感器间的空间依赖关系。

输入：(B, M, D)
输出：(B, M, D)
"""

import torch
import torch.nn as nn


class SpatialEncoder(nn.Module):
    """重型空间 Transformer 编码器"""

    def __init__(
        self,
        num_sensors: int = 64,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.d_model = d_model

        # TODO: 构建空间位置编码、Transformer Encoder 层

    def forward(
        self, x: torch.Tensor, visible_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: 可见传感器特征, shape (B, M, D)
            visible_indices: 可见传感器索引, 用于选取对应位置编码
        Returns:
            encoded: 编码后的特征, shape (B, M, D)
        """
        # TODO: 实现前向传播
        raise NotImplementedError
