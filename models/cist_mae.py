# -*- coding: utf-8 -*-
"""
CIST-MAE 主模型
Channel-Independent Spatio-Temporal Masked Autoencoder

将各子模块组装，实现完整的前向传播与自监督掩码重构损失计算。

数据流: D-total time steps; N-total sensors; B-batch; L-window length; c-visible sensors; d-embedding dim
        (D, N) -> TemporalEncoder -> (B, c, d) + visible_indices + mask_indices + batch_X
               -> SpatialEncoder  -> (B, c, d)
               -> SpatialDecoder  -> (B, N, d)
               -> ProjectionHead  -> signal_out(B, N, 1) + sequence_out(B, N, L)
               -> Loss(仅被掩码传感器)
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
        enc_heads: int = 2,
        enc_layers: int = 1,
        enc_ffn_dim: int = 64,
        enc_dropout: float = 0.3,
        # 空间解码器参数
        dec_heads: int = 2,
        dec_layers: int = 2,
        dec_ffn_dim: int = 128,
        dec_dropout: float = 0.4,
        # 损失权重
        lambda_signal: float = 1.0,
        lambda_sequence: float = 1.0,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.L = L
        self.lambda_signal = lambda_signal
        self.lambda_sequence = lambda_sequence

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

    def forward(self, x: torch.Tensor):
        """
        自监督前向传播：标签来自数据本身，无需外部提供 y。
        
        Args:
            x: 原始时序数据, shape (D, C)  —— D 为总时间步数, C 为传感器总数
        Returns:
            signal_out: 全部传感器的预测值, shape (B, N, 1)
            sequence_out: 全部传感器的重构时序, shape (B, N, L)
            loss: 自监督训练损失
        """
        # 1. 时序编码（内部完成 z-score、批次切分、掩码、卷积、池化）
        tokens, visible_indices, mask_indices, batch_X = self.temporal_encoder(x)
        # tokens: [B, c, d_model]
        # visible_indices: [B, c]
        # mask_indices: [B, num_masked]
        # batch_X: [B, L, C] —— 标准化后的原始窗口数据（掩码前）

        # 2. 空间编码
        encoded = self.spatial_encoder(tokens, visible_indices)
        # [B, c, d_model]

        # 3. 解码融合
        decoded = self.spatial_decoder(encoded, visible_indices, mask_indices)
        # [B, N, d_model]

        # 4. 双分支投射
        signal_out, sequence_out = self.projection_head(decoded)
        # signal_out: [B, N, 1]       每个传感器的预测标量
        # sequence_out: [B, N, L]     每个传感器的重构时序

        # === 自监督损失计算 ===
        # 真实标签来自模型内部切割出的原始窗口数据 batch_X [B, L, C]
        B = batch_X.shape[0]

        # --- 分支一损失：回归预测 ---
        # 真实值：取每个传感器窗口的最后一个时间点作为预测目标
        # batch_X 形状 [B, L, C]，取最后一步 → [B, C, 1]
        y_signal_true = batch_X[:, -1, :].unsqueeze(-1)  # [B, C, 1]
        # 只取被掩码传感器位置的预测和真实值
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1)
        pred_signal = signal_out[batch_idx, mask_indices]  # [B, num_masked, 1]
        true_signal = y_signal_true[batch_idx, mask_indices]  # [B, num_masked, 1]
        loss_signal = F.mse_loss(pred_signal, true_signal)

        # --- 分支二损失：时序重构 ---
        # 真实值：batch_X 转置为 [B, C, L]，使通道在第二维
        y_seq_true = batch_X.permute(0, 2, 1)  # [B, C, L]
        # 只取被掩码传感器位置
        pred_seq = sequence_out[batch_idx, mask_indices]  # [B, num_masked, L]
        true_seq = y_seq_true[batch_idx, mask_indices]  # [B, num_masked, L]
        loss_sequence = F.mse_loss(pred_seq, true_seq)

        # --- 总损失 ---
        loss = self.lambda_signal * loss_signal + self.lambda_sequence * loss_sequence

        return signal_out, sequence_out, loss
