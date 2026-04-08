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

        # 1. 空间位置编码 (为每个可能的传感器分配固定的空间身份标识)
        # 采用正态分布小幅噪声初始化，利于训练起步
        self.pos_embed = nn.Parameter(torch.randn(1, num_sensors, d_model) * 0.02)
        
        # 2. Transformer Encoder 网络层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",  # Transformer架构中 Gelu 比 Relu 表现更好
            batch_first=True    # 保证张量接受 [B, SeqLength, Feature] 的格式
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

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
        # 1. 形状自适应对齐防错
        # 上一级的输出如果没来得及转为 [B, c, d_model] 而是 [B, d_model, c]，在这里帮它倒序过来
        # 因为 Transformer 是序列模型，它要求最后一位必须是打散的特征维度
        if x.shape[1] == self.d_model and x.shape[-1] != self.d_model:
            x = x.transpose(1, 2)
            
        B, c, D_dim = x.shape
        
        # 2. 依据掩码矩阵，精准提取保留下来的传感器的专有位置编码
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1)
        b_pos_embed = self.pos_embed.expand(B, -1, -1)
        
        # [B_dim, 1] 和 [B_dim, c_dim] 联手，从位置编码典籍中把留下的那几号抽出来
        visible_pos_embed = b_pos_embed[batch_idx, visible_indices, :]
        
        # 注入身份证明
        x = x + visible_pos_embed
        
        # 3. 跑重型的 Transformer 空间多头注意力提取
        encoded = self.encoder(x)
        
        return encoded
