# -*- coding: utf-8 -*-
"""
CIST-MAE 主模型
Channel-Independent Spatio-Temporal Masked Autoencoder

将各子模块组装，实现完整的前向传播与掩码重构损失计算。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .channel_masking import ChannelMasking
from .temporal_encoder import TemporalEncoder
from .spatial_encoder import SpatialEncoder
from .spatial_decoder import SpatialDecoder
from .projection_head import ProjectionHead


class CIST_MAE(nn.Module):
    """
    CIST-MAE 主模型

    数据流:
        (B, N, L) -> ChannelMasking -> (B, M, L)
                  -> TemporalEncoder -> (B, M, D)
                  -> SpatialEncoder  -> (B, M, D)
                  -> SpatialDecoder  -> (B, N, D)
                  -> ProjectionHead  -> (B, N, 1)
    """

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        d_model = model_cfg["d_model"]
        num_sensors = config["data"]["num_sensors"]

        # --- 子模块初始化 ---
        self.channel_masking = ChannelMasking(
            mask_ratio=model_cfg["mask_ratio"],
        )

        te_cfg = model_cfg["temporal_encoder"]
        self.temporal_encoder = TemporalEncoder(
            d_model=d_model,
            conv_channels=te_cfg["conv_channels"],
            kernel_sizes=te_cfg["kernel_sizes"],
            pool_type=te_cfg["pool_type"],
        )

        se_cfg = model_cfg["spatial_encoder"]
        self.spatial_encoder = SpatialEncoder(
            num_sensors=num_sensors,
            d_model=d_model,
            num_layers=se_cfg["num_layers"],
            num_heads=se_cfg["num_heads"],
            ffn_dim=se_cfg["ffn_dim"],
            dropout=se_cfg["dropout"],
        )

        sd_cfg = model_cfg["spatial_decoder"]
        self.spatial_decoder = SpatialDecoder(
            num_sensors=num_sensors,
            d_model=d_model,
            num_layers=sd_cfg["num_layers"],
            num_heads=sd_cfg["num_heads"],
            ffn_dim=sd_cfg["ffn_dim"],
            dropout=sd_cfg["dropout"],
        )

        self.projection_head = ProjectionHead(d_model=d_model)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        """
        Args:
            x: 输入时序数据, shape (B, N, L)
            y: 真实标签（当前时刻物理量）, shape (B, N, 1)，训练时需提供
        Returns:
            y_hat: 全部传感器的预测值, shape (B, N, 1)
            loss: 仅被掩码传感器的 MSE 损失（训练时返回，推理时为 None）
        """
        # 1. 通道级空间掩码
        x_visible, mask_indices, visible_indices = self.channel_masking(x)
        # shape: (B, N, L) -> x_visible: (B, M, L)

        # 2. 时序编码
        tokens = self.temporal_encoder(x_visible)
        # shape: (B, M, L) -> (B, M, D)

        # 3. 空间编码
        encoded = self.spatial_encoder(tokens, visible_indices)
        # shape: (B, M, D) -> (B, M, D)

        # 4. 解码融合（填充 mask_token + 轻量 Transformer）
        decoded = self.spatial_decoder(encoded, visible_indices, mask_indices)
        # shape: (B, M, D) -> (B, N, D)

        # 5. 投射至物理量
        y_hat = self.projection_head(decoded)
        # shape: (B, N, D) -> (B, N, 1)

        # --- 损失计算：仅计算被掩码传感器的重构误差 ---
        loss = None
        if y is not None:
            # 提取被掩码位置的预测与真实值
            y_hat_masked = torch.gather(
                y_hat, dim=1,
                index=mask_indices.unsqueeze(-1).expand(-1, -1, 1),
            )  # shape: (B, N-M, 1)
            y_masked = torch.gather(
                y, dim=1,
                index=mask_indices.unsqueeze(-1).expand(-1, -1, 1),
            )  # shape: (B, N-M, 1)
            loss = F.mse_loss(y_hat_masked, y_masked)

        return y_hat, loss
