# -*- coding: utf-8 -*-
"""
CIST-MAE 主模型
Channel-Independent Spatio-Temporal Masked Autoencoder

将各子模块组装，实现完整的前向传播与掩码重构损失计算。

数据流: B-batch; L-cut data length; N-total sensors; c-visible sensors; d-embedding dim
        (D, N) -> TemporalEncoder -> (B, c, d) + visible_indices + mask_indices
               -> SpatialEncoder  -> (B, c, d)
               -> SpatialDecoder  -> (B, N, d)
               -> ProjectionHead  -> (B, N, 1) and (B, N, L)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal_encoder import TemporalEncoder
from .spatial_encoder import SpatialEncoder
from .spatial_decoder import SpatialDecoder
from .projection_head import ProjectionHead


class CIST_MAE(nn.Module):

    def __init__(
        self,
        # 数据参数
        num_sensors: int = 61,
        # 时序编码器参数
        d_model: int = 128,
        L: int = 500,
        Batchsize: int = 32,
        mask_ratio: float = 0.5,
        # 空间编码器参数
        enc_heads: int = 4,
        enc_layers: int = 2,
        enc_ffn_dim: int = 256,
        enc_dropout: float = 0.2,
        # 空间解码器参数
        dec_heads: int = 8,
        dec_layers: int = 4,
        dec_ffn_dim: int = 512,
        dec_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.L = L

        # --- 子模块初始化 ---
        # 1. 时序编码器（内部包含 z-score、批次切分、通道掩码、因果卷积、池化）
        self.temporal_encoder = TemporalEncoder(
            d_model=d_model,
            L=L,
            Batchsize=Batchsize,
            mask_ratio=mask_ratio,
        )

        # 2. 轻量空间编码器
        self.spatial_encoder = SpatialEncoder(
            num_sensor=num_sensors,
            d_model=d_model,
            heads=enc_heads,
            layers=enc_layers,
            ffn_dim=enc_ffn_dim,
            dropout=enc_dropout,
        )

        # 3. 重量空间解码器
        self.spatial_decoder = SpatialDecoder(
            num_sensors=num_sensors,
            d_model=d_model,
            heads=dec_heads,
            layers=dec_layers,
            ffn_dim=dec_ffn_dim,
            dropout=dec_dropout,
        )

        # 4. 双分支投射头
        self.projection_head = ProjectionHead(d_model=d_model, L=L)

    def forward(self, x: torch.Tensor, y_signal: torch.Tensor = None, y_sequence: torch.Tensor = None):
        """
        Args:
            x: 输入时序数据, shape (D, C)  —— D 为总时间步数, C 为传感器总数
            y_signal: 真实标签（当前时刻物理量）, shape (B, N, 1)，训练时提供
            y_sequence: 真实历史时序, shape (B, N, L)，训练时提供
        Returns:
            signal_out: 全部传感器的预测值, shape (B, N, 1)
            sequence_out: 全部传感器的重构时序, shape (B, N, L)
            loss: 训练损失（训练时返回，推理时为 None）
        """
        # 1. 时序编码（内部完成 z-score、批次切分、掩码、卷积、池化）
        tokens, visible_indices, mask_indices = self.temporal_encoder(x)
        # tokens: [B, c, d_model]
        # visible_indices: [B, c]
        # mask_indices: [B, num_masked]

        # 2. 空间编码（注入位置编码 + Transformer 多头注意力）
        encoded = self.spatial_encoder(tokens, visible_indices)
        # encoded: [B, c, d_model]

        # 3. 解码融合（填充 mask_token + 恢复顺序 + Transformer 解码）
        decoded = self.spatial_decoder(encoded, visible_indices, mask_indices)
        # decoded: [B, N, d_model]

        # 4. 双分支投射
        signal_out, sequence_out = self.projection_head(decoded)
        # signal_out: [B, N, 1]
        # sequence_out: [B, N, L]

        # --- 损失计算：仅计算被掩码传感器的重构误差 ---
        loss = None
        if y_signal is not None or y_sequence is not None:
            total_loss = 0.0

            # 分支一损失：回归预测损失（仅被掩码传感器）
            if y_signal is not None:
                pred_signal = torch.gather(
                    signal_out, dim=1,
                    index=mask_indices.unsqueeze(-1).expand(-1, -1, 1),
                )  # [B, num_masked, 1]
                true_signal = torch.gather(
                    y_signal, dim=1,
                    index=mask_indices.unsqueeze(-1).expand(-1, -1, 1),
                )  # [B, num_masked, 1]
                loss_signal = F.mse_loss(pred_signal, true_signal)
                total_loss = total_loss + loss_signal

            # 分支二损失：历史重构损失（仅被掩码传感器）
            if y_sequence is not None:
                pred_seq = torch.gather(
                    sequence_out, dim=1,
                    index=mask_indices.unsqueeze(-1).expand(-1, -1, self.L),
                )  # [B, num_masked, L]
                true_seq = torch.gather(
                    y_sequence, dim=1,
                    index=mask_indices.unsqueeze(-1).expand(-1, -1, self.L),
                )  # [B, num_masked, L]
                loss_seq = F.mse_loss(pred_seq, true_seq)
                total_loss = total_loss + loss_seq

            loss = total_loss

        return signal_out, sequence_out, loss
