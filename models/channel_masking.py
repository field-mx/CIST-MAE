# -*- coding: utf-8 -*-
"""
通道级空间掩码模块 (ChannelMasking)

功能：根据 mask_ratio 随机选取可见传感器通道，
     分离可见/被掩码传感器的时序数据，以节省后续时序计算算力。

输入：(data, channel)
输出：可见数据 (D, C_Visible)、不可见数据(D, C_INvisible)

掩码索引代表通道位置
总通道数*掩码率=不可见通道数量，剩余通道为可见通道
掩码通道每次随机选择
"""

import torch
import torch.nn as nn
from .dataset import DataSeperate


class ChannelMasking(nn.Module):  
    """通道级空间掩码模块"""

    def __init__(self, mask_ratio: float = 0.5):
        super().__init__()
        self.mask_ratio = mask_ratio
        
    def z_score(self, x:torch.Tensor)->torch.Tensor:
        """
        Args:
            x: 输入张量, shape (data, channel)
        Returns:
            x_zscore: 标准化后的张量, shape (data, channel)
        """
        # get shape of data
        D, C = x.shape
        # calculate mean and std
        mean = x.mean(dim=0)
        std = x.std(dim=0)
        # z-score normalization
        x_zscore = (x - mean) / std
        return x_zscore
    # 随机掩码函数
    # input：(data, channel),mask_ratio
    # output：(data, channel_masked)        
            
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: 输入张量, shape (batch, data, channel)
        Returns:
            mask_matrix: 掩码矩阵 形状 (batch, channel_unmasked)
        """
        # 1. 获取三维数据大小 [B, L, C]
        B, L, C = x.shape
        
        # calculate number of masked sensors
        num_masked = int(C * self.mask_ratio)
        c = C - num_masked  # c 为处在 0-C 之间的可见通道数
        
        # 2. 根据掩码率生成掩码矩阵 [B, c] 和 [B, num_masked]
        # (因为你需要 [B, c] 维度的不重复随机索引，最简洁的写法就是用 argsort)
        noise = torch.rand(B, C, device=x.device)
        shuffle_indices = torch.argsort(noise, dim=1)
        self.mask_indices = shuffle_indices[:, :num_masked]       # [B, num_masked]
        self.visible_indices = shuffle_indices[:, num_masked:]    # [B, c]
        # 返回
        return self.visible_indices, self.mask_indices

#module 功能测试
if __name__ == "__main__":
    # 直接运行本文件时执行
    import os
    # 定位到项目根目录下的 data 文件夹
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    excel_path = os.path.join(project_root, "data", "SensorData.xlsx")
    
    dataset = DataSeperate(file_path=excel_path)  # 创建一个实例
    dataset.process()  

    # get train data
    ChannelMasking = ChannelMasking() 
    ChannelMasking.forward(dataset.train_tensor)
    print("x_visible_shape:", ChannelMasking.x_visible.shape)
                  
    
    
