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
输出：(B, d', c)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .dataset import DataSeperate
from .channel_masking import ChannelMasking
from .temporal_encoder import TemporalEncoder
from .spatial_encoder import SpatialEncoder
from .spatial_decoder import SpatialDecoder
from .projection_head import ProjectionHead

# 定义左填充卷积
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        
    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)

class TemporalEncoder(nn.Module):
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
      - c为可见通道数,由掩码率计算得到


    输入：(D, C),L,Batchsize
    输出：(B, d', c)
    """

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
        self.conv_channels = conv_channels
        self.Batchsize = Batchsize

        self.conv1 = CausalConv1d(in_channels=1, out_channels=16, kernel_size=8)
        self.conv2 = CausalConv1d(in_channels=16, out_channels=64, kernel_size=5)
        self.conv3 = CausalConv1d(in_channels=64, out_channels=128, kernel_size=3)
        
        