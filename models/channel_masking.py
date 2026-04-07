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
from dataset import DataSeperate


class ChannelMasking(nn.Module):  
    """通道级空间掩码模块"""

    def __init__(self, mask_ratio: float = 0.5):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: 输入张量, shape (data, channel)
        Returns:
            x_visible: 可见传感器数据, 形状 (M, L)
            mask_indices: 被掩码的传感器索引, 一维张量
            visible_indices: 可见传感器索引, 一维张量
        """
        # get number of sensors
        num_sensors = x.shape[1]
        # calculate number of masked sensors
        num_masked = int(num_sensors * self.mask_ratio)
        # generate mask indices
        self.mask_indices = torch.randperm(num_sensors)[:num_masked]
        # generate visible indices
        self.visible_indices = torch.randperm(num_sensors)[num_masked:]
        # get visible data
        self.x_visible = x[:, self.visible_indices]
        # get masked data
        self.x_masked = x[:, self.mask_indices]

""" module 功能测试
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
"""                    
    
    
