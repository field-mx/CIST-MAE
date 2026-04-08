# -*- coding: utf-8 -*-
"""
时序编码器 (TemporalEncoder)

功能：采用通道独立机制，通过因果一维卷积提取时序特征，
     再通过加权池化压缩时间维度，映射至隐空间维度 D。

架构：
  - 输入分割好的训练集[D,C]
  - 对D维度的数据进行标准化[D,C]
  - 随机窗口进行长度为L的batchsize分割[B, L, C]
  - 对每个batch：
    - 进行随机掩码[B, L, c]
    - 对未掩码的通道进行因果卷积[B, d, l, c]
    - 对未掩码的通道激活[B, d, l, c]
    - 对未掩码的通道进行加权池化[B, d, c]
    - 对未掩码的通道进行线性映射[B, d', c]
  - 线性映射至隐维度 d',为后续输入维度
  - d为卷积特征维度
  - l为卷积后的时序长度
  - c为可见通道数


输入：( D, C),L,Batchsize
输出：(B, c, d'),batchsize,channel,embedding_dim
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .dataset import DataSeperate
from .channel_masking import ChannelMasking
from .spatial_encoder import SpatialEncoder
from .spatial_decoder import SpatialDecoder
from .projection_head import ProjectionHead

# 定义左填充卷积
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride)
        
    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)

class TemporalEncoder(nn.Module):

    def __init__(
        self, 
        # 最终特征维度
        d_model = 128,
        # 敏感历史数据长度
        L = 500,
        
        Batchsize = 32,
        mask_ratio = 0.5
    ):
        super().__init__()
        self.d_model = d_model
        self.L = L
        self.Batchsize = Batchsize
        self.mask_ratio = mask_ratio

        # 卷积层
        self.conv1 = CausalConv1d(in_channels=1, out_channels=16, kernel_size=8, stride = 4)
        self.conv2 = CausalConv1d(in_channels=16, out_channels=64, kernel_size=5, stride = 2)
        self.conv3 = CausalConv1d(in_channels=64, out_channels=128, kernel_size=3, stride = 2)
        # 激活函数
        self.relu = nn.ReLU()
        # 加权池化层(32个浓缩时序序列)
        self.weighted_pool = nn.Linear(32, 1)
        # 线性映射层（接收维度，输出维度，用于输入vit维度统一）
        self.linear = nn.Linear(128, d_model)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        """
        Args:
            x: 输入张量, shape (D, C)
        Returns:
            x_encoded: 编码器输出, shape (B, d', c)
        """
        # get shape of data
        D, C = x.shape
        B = self.Batchsize
        L = self.L
        
        # 1. data z-score
        mask_module = ChannelMasking(mask_ratio=self.mask_ratio)
        x = mask_module.z_score(x)

        # 2. Batchsize cut
        # conform start location of batch
        if D<=L:
            starts = torch.zeros(B, dtype=torch.long)
        else:
            starts = torch.randint(0, D - L + 1, (B,))
        batch_X = torch.stack([x[s : s + L, :] for s in starts], dim=0)
        
        # 3. Mask the data(ai help)
        visible_indices, mask_indices = mask_module(batch_X)  # visible_indices: [B, c]
        B_dim, L_dim, C_dim = batch_X.shape
        c_dim = visible_indices.shape[1]
        # 生成形状为 [B, 1] 的批次序列号
        batch_idx = torch.arange(B_dim).unsqueeze(1)
        # 根据掩码矩阵的索引,不同批次提取不同的传感器,重新拼接[B, c, L]
        x_visible = batch_X[batch_idx, :, visible_indices]

        # 4. causal convolution 
        # 压缩,一维时序卷积
        x_conv_in = x_visible.reshape(B_dim * c_dim, 1, L_dim)
        x = self.conv1(x_conv_in)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        # 卷积完后，x 的形状是 [B*c, 128, L_final]，例如 [B*c, 128, 32]
        d_conv = x.shape[1]      # 特征厚度 (128)
        L_final = x.shape[2]     # 最终的序列长度 (32)
        # 解压复原
        # 把 B 和 c 重新拆开，形状恢复并对准 [B, c, 特征厚度128, 时序长度32]
        x = x.view(B_dim, c_dim, d_conv, L_final)
        
        # 5. 加权池化聚合与线性映射统一维度
        # nn.Linear 作用在最后维度，将 32 个时序步长通过打分融合为 1 个数值
        x = self.weighted_pool(x)       # -> [B, c, 128, 1]
        x = x.squeeze(-1)               # -> [B, c, 128]
        
        # 应用投影层，将 128 维变为规定的 d_model 维度
        x = self.linear(x)              # -> [B, c, d_model]
        return x
