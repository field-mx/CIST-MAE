# -*- coding: utf-8 -*-
"""
双分支物理量投射头 (ProjectionHead)

采用多任务学习双分支（Dual-Head）架构：
  分支一（回归预测）：全连接层 + 加权池化 → 预测单一物理量标量
  分支二（历史重构）：反卷积序列 → 还原时序波形

输入：(B, num_sensors, d_model)
输出：
  regression_out: (B, num_sensors, 1)       回归预测值
  recon_out:      (B, num_sensors, L)       重构的历史时序
"""

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """双分支物理量投射头"""

    def __init__(self, d_model: int = 128, L: int = 500):
        super().__init__()
        self.d_model = d_model
        self.L = L

        # ============ 分支一：回归预测分支 (Regression Head) ============
        # 全连接层 + 激活 + 加权池化，将 d_model 维特征压缩为 1 个标量
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),   # 128 -> 64
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),          # 64 -> 1
        )

        # ============ 分支二：历史演化重构分支 (Reconstruction Head) ============
        # 与 TemporalEncoder 的卷积序列对称的反卷积序列
        # 编码路径: L=500 --conv1(k=8,s=4)--> 125 --conv2(k=5,s=2)--> 63 --conv3(k=3,s=2)--> 32
        # 解码路径: 反向还原
        
        # 步骤 1: 将 d_model 维特征映射回卷积特征空间 [128维, 长度1]
        self.recon_fc = nn.Linear(d_model, 128)
        
        # 步骤 2: 反卷积序列，逐步还原时序长度
        # 128@1 -> 64@3（对应 conv3 的逆操作）
        self.deconv3 = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=0)
        # 64@3 -> 16@7（对应 conv2 的逆操作）
        self.deconv2 = nn.ConvTranspose1d(64, 16, kernel_size=5, stride=2, padding=0)
        # 16@7 -> 1@L（对应 conv1 的逆操作）
        self.deconv1 = nn.ConvTranspose1d(16, 1, kernel_size=8, stride=4, padding=0)
        
        self.Relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: 解码器输出, shape (B, num_sensors, d_model)
        Returns:
            regression_out: 回归预测值, shape (B, num_sensors, 1)
            recon_out: 重构的历史时序, shape (B, num_sensors, L)
        """
        B, N, D = x.shape

        # ============ 分支一：回归预测 ============
        signal_out = self.regression_head(x)  # [B, N, d_model] -> [B, N, 1]

        # ============ 分支二：历史演化重构 ============
        # 2.1 将每个传感器的特征映射为反卷积的起始信号
        recon = self.recon_fc(x)                  # [B, N, 128]
        
        # 2.2 将 B 和 N 合并，按通道独立方式处理（与编码器中的策略一致）
        recon = recon.reshape(B * N, 128)         # [B*N, 128]
        recon = recon.unsqueeze(2)                 # [B*N, 128, 1] — 反卷积输入格式
        
        # 2.3 反卷积序列：逐步还原时序长度
        recon = self.Relu(self.deconv3(recon))     # [B*N, 64, 3]
        recon = self.Relu(self.deconv2(recon))     # [B*N, 16, 9]
        recon = self.deconv1(recon)                # [B*N, 1, L_raw] 最后一层不加激活
        
        # 2.4 去掉通道维度并截断/填充到精确的 L
        recon = recon.squeeze(1)                   # [B*N, L_raw]
        # 反卷积可能产生的长度与 L 略有偏差，需要裁剪或填充对齐
        L_raw = recon.shape[1]
        if L_raw >= self.L:
            recon = recon[:, :self.L]              # 截断多余部分
        else:
            recon = nn.functional.pad(recon, (0, self.L - L_raw))  # 右侧零填充
        
        # 2.5 恢复 B 和 N 维度
        sequence_out = recon.reshape(B, N, self.L)    # [B, num_sensors, L]

        return signal_out, sequence_out
