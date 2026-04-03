# -*- coding: utf-8 -*-
"""
占位符拼接与轻量解码器 (SpatialDecoder)

功能：使用可学习的 mask_token 填充被掩码位置，恢复完整 N 个传感器的特征，
     添加完整空间位置编码后，通过轻量级 Transformer 解码融合。

输入：编码器输出 (B, M, D)，可见/掩码索引
输出：(B, N, D)
"""

import torch
import torch.nn as nn


class SpatialDecoder(nn.Module):
    """轻量空间 Transformer 解码器"""

    def __init__(
        self,
        num_sensors: int = 64,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.d_model = d_model

        # TODO: 构建 mask_token、空间位置编码、Transformer Decoder 层

    def forward(
        self,
        x_encoded: torch.Tensor,
        visible_indices: torch.Tensor,
        mask_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_encoded: 编码器输出, shape (B, M, D)
            visible_indices: 可见传感器索引
            mask_indices: 被掩码传感器索引
        Returns:
            decoded: 恢复完整传感器的解码特征, shape (B, N, D)
        """
        # TODO: 实现前向传播
        raise NotImplementedError
