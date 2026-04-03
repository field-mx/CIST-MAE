# -*- coding: utf-8 -*-
"""
时序编码器 (TemporalEncoder)

功能：采用通道独立机制，通过因果一维卷积提取时序特征，
     再通过加权池化压缩时间维度，映射至隐空间维度 D。

架构：
  - 2-3 层因果一维卷积（Causal 1D-Conv），左侧非对称补零
  - 每层后接 RELU 激活
  - 时间轴加权池化（Attention Pooling / 指数加权平均）
  - 线性映射至隐维度 D

输入：(B, M, L)
输出：(B, M, D)
"""

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """通道独立时序编码器"""

    def __init__(
        self,
        d_model: int = 128,
        conv_channels: list = None,
        kernel_sizes: list = None,
        pool_type: str = "attention",
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [1, 32, 64]
        if kernel_sizes is None:
            kernel_sizes = [5, 3]

        self.d_model = d_model
        self.conv_channels = conv_channels
        self.kernel_sizes = kernel_sizes
        self.pool_type = pool_type

        # TODO: 构建因果卷积层、池化层、线性映射层

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 可见传感器数据, shape (B, M, L)
        Returns:
            tokens: 时序特征 Token, shape (B, M, D)
        """
        # TODO: 实现前向传播
        raise NotImplementedError
