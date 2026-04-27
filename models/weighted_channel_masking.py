# -*- coding: utf-8 -*-
"""
加权通道掩码模块 (WeightedChannelMasking)

基于 Apriori 挖掘得到的传感器重要性先验，实现加权随机掩码。
"黄金传感器" 保留概率高（仅 ~10% 被遮蔽），冗余传感器保留概率低（~90% 被遮蔽）。

与原有 ChannelMasking 接口完全兼容，可直接替换。

输入：(B, L, C)
输出：visible_indices (B, c), mask_indices (B, num_masked)
"""

import torch
import torch.nn as nn


class WeightedChannelMasking(nn.Module):
    """基于传感器重要性先验的加权随机掩码"""

    def __init__(self, retain_probs: torch.Tensor, mask_ratio: float = 0.4):
        """
        Args:
            retain_probs: 每个传感器的保留概率, shape [N]，值域 [0, 1]
                          例如 [0.9, 0.9, 0.1, 0.5, ...]
                          - 0.9 = 黄金传感器，90% 概率被保留（10% 被遮蔽）
                          - 0.1 = 冗余传感器，10% 概率被保留（90% 被遮蔽）
                          - 0.5 = 普通传感器，均匀随机
            mask_ratio: 目标掩码率，用于控制总掩码传感器数量
        """
        super().__init__()
        self.mask_ratio = mask_ratio
        # 注册为 buffer，不参与梯度计算但随模型移动到 GPU
        self.register_buffer('retain_probs', retain_probs)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: 输入张量, shape (B, L, C)
        Returns:
            visible_indices: 可见传感器索引, shape (B, c)
            mask_indices: 被掩码传感器索引, shape (B, num_masked)
        """
        B, L, C = x.shape
        num_masked = int(C * self.mask_ratio)
        c = C - num_masked  # 可见通道数

        # 加权采样策略：
        # 用 retain_probs 作为权重，通过 Gumbel-top-k 技巧实现加权不重复采样
        # 原理：给每个传感器加上 Gumbel 噪声后排序，retain_probs 越高的传感器
        #       越倾向于排在前面（被保留），越低的越倾向于排在后面（被遮蔽）
        
        # 将 retain_probs 扩展到 [B, C]
        log_probs = torch.log(self.retain_probs.clamp(min=1e-6)).unsqueeze(0).expand(B, -1)
        
        # Gumbel 噪声
        gumbel_noise = -torch.log(-torch.log(torch.rand(B, C, device=x.device) + 1e-8) + 1e-8)
        
        # 加权打分：log(保留概率) + Gumbel噪声
        scores = log_probs + gumbel_noise
        
        # 按分数排序：分数高的保留，分数低的遮蔽
        shuffle_indices = torch.argsort(scores, dim=1, descending=True)
        
        visible_indices = shuffle_indices[:, :c]          # [B, c]
        mask_indices = shuffle_indices[:, c:]              # [B, num_masked]

        return visible_indices, mask_indices
