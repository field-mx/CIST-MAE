# -*- coding: utf-8 -*-
"""
轻量空间编码器 (SpatialEncoder)
输入张量: [B, c, d_model],batchsize,channel,embedding_dim,编码后的特征维度,匹配编码器输入,暂定128
功能：接收可见传感器的时序特征 Token，添加空间位置编码后，
     通过多层 Transformer Encoder 捕获传感器间的空间依赖关系。

输出：[B, c, d_model],batchsize,channel,embedding_dim
"""

import torch
import torch.nn as nn

class SpatialEncoder(nn.Module):
    def __init__(
        self, 
        num_sensor = 61,
        d_model = 128,
        heads = 4,
        layers = 2,
        ffn_dim = 256,
        dropout = 0.2
    ):
        super().__init__()
        self.num_sensor = num_sensor
        self.d_model = d_model
        self.heads = heads
        self.layers = layers
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        
        # 1.传感器位置编码,掩码矩阵在这里面选取位置编码,叠加时序特征输入进网络[1, 61, 128]
        self.sensor_pos_embedding = nn.Parameter(torch.randn(1, num_sensor, d_model))

        # 2.Transformer Encoder层
        # 2.1 定义单层 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        # 2.2 堆叠多层
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    # 3.前向传播过程
    def forward(self, x: torch.Tensor, visible_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量, shape (B, c, d_model)
            visible_indices: 可见传感器索引, shape (B, c)
        Returns:
            x_encoded: 编码器输出, shape (B, c, d_model)
        """

        # 1. 获取三维数据大小 [B, c, d_model]
        B, c, d_model = x.shape
        # 2. 根据掩码矩阵,提取已知传感器的编码
        # batch目录,用于visible矩阵查阅[B,1]
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1)
        # 扩展到[B,61,128]
        sensor_pos_embedding = self.sensor_pos_embedding.expand(B, -1, -1)
        # 提取可见传感器的位置编码
        sensor_pos_embedding = sensor_pos_embedding[batch_idx, visible_indices]
        # 3. 时间特征叠加位置编码
        x = x + sensor_pos_embedding
        # 4.transformer 前向传播
        x = self.encoder(x)
        return x

        
        
        