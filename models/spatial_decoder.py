# -*- coding: utf-8 -*-
"""
掩码空窗拼接与重量解码器 (SpatialDecoder)

功能：使用可学习的 mask_window[B, num_masked, d_model] 填充被掩码位置，恢复所有传感器的特征，
     添加完整空间位置编码后，通过重量 Transformer 解码融合。

输入：编码器输出 (B, c, d_model)，visible_indices (B, c)，mask_indices (B, num_masked)
输出：(B, num_sensors, d_model)
"""

import torch
import torch.nn as nn


class SpatialDecoder(nn.Module):
    def __init__(
        self,
        num_sensors = 61,
        d_model = 128,
        heads = 8,
        layers = 4,
        ffn_dim = 512,
        dropout = 0.1,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.d_model = d_model

        # 1. 可学习的掩码占位符 [MASK] token
        # 形状 [1, 1, d_model]，后续会自动广播到 [B, num_masked, d_model]
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # 2. 完整的空间位置编码（覆盖全部 61 个传感器）
        self.sensor_pos_embedding = nn.Parameter(torch.randn(1, num_sensors, d_model) * 0.02)

        # 3. 轻量 Transformer Decoder 层（比编码器少，仅 2 层）
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=layers)

    def forward(
        self,
        x_encoded: torch.Tensor,
        visible_indices: torch.Tensor,
        mask_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_encoded: 编码器输出, shape (B, c, d_model)
            visible_indices: 可见传感器索引, shape (B, c)
            mask_indices: 被掩码传感器索引, shape (B, num_masked)
        Returns:
            decoded: 恢复完整传感器的解码特征, shape (B, num_sensors, d_model)
        """
        B, c, D = x_encoded.shape
        num_masked = mask_indices.shape[1]

        # 1. 生成掩码占位符
        #   mask_token [1,1,128] 广播扩展为 [B, num_masked, 128]
        mask_tokens = self.mask_token.expand(B, num_masked, D)

        # 2. 拼接：把编码器输出（可见部分）和掩码占位符拼在一起
        #   x_full 形状: [B, c + num_masked, d_model] = [B, 61, 128]
        x_full = torch.cat([x_encoded, mask_tokens], dim=1)

        # 3. 恢复原始传感器顺序
        #   目前 x_full 中前 c 列是可见传感器，后 num_masked 列是占位符
        #   需要按照原始的传感器编号重新排列回正确的物理位置
        #   按照该批次的掩码矩阵拼接完整索引: [B, 61]，前半是可见编号，后半是掩码编号
        restore_indices = torch.cat([visible_indices, mask_indices], dim=1)
        #   argsort 按照索引顺序(正确传感器位置01234),在C维度中是什么位置[4,0,1,2,3]
        restore_order = torch.argsort(restore_indices, dim=1)
        #   用高级索引把每一行按照 restore_order 重排
        #   生成批次索引[B,1]
        batch_idx = torch.arange(B, device=x_encoded.device).unsqueeze(1)
        x_full = x_full[batch_idx, restore_order]

        # 4. 添加完整的空间位置编码（全部 61 个传感器都加）
        x_full = x_full + self.sensor_pos_embedding

        # 5. 解码
        decoded = self.decoder(x_full)

        return decoded
