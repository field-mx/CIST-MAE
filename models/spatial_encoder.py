# -*- coding: utf-8 -*-
"""
重型空间编码器 (SpatialEncoder)
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
        heads = 8,
        layers = 4,
        ffn_dim = 512,
        dropout = 0.1
    ):
        super().__init__()
        self.num_sensor = num_sensor
        self.d_model = d_model
        self.heads = heads
        self.layers = layers
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        
        # 1.传感器位置编码,掩码矩阵在这里面选取位置编码,叠加时序特征输入进网络
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
        # 3. 时间特征叠加位置编码
        # 4.transformer 前向传播
        x = self.encoder(x)
        return x

        
        
        