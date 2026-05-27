# CIST-MAE: Channel-Independent Spatio-Temporal Masked Autoencoder

面向复杂工业物理系统的多传感器时空规律学习模型。  
本项目以工业制氢场景下的多通道传感器数据为对象，构建 CIST-MAE（Channel-Independent Spatio-Temporal Masked Autoencoder）模型，用于在稀疏观测条件下学习传感器间的时空依赖关系，实现被掩码传感器状态预测与历史序列重构。

## 1. 项目背景

在工业制氢、能源装备、流程工业等复杂物理系统中，传感器数据通常具有以下特点：

- 通道数量多，不同传感器之间存在复杂空间耦合；
- 时间序列具有明显的动态演化规律；
- 真实工业场景中可能存在传感器缺失、采样稀疏、噪声扰动等问题；
- 系统状态难以被所有传感器完整、实时、低成本地观测。

因此，本项目尝试通过自监督掩码建模方法，让模型仅依靠部分可见传感器通道，恢复被掩码通道的当前状态和历史变化规律，从而为工业数字孪生、稀疏感知、状态预测和故障预警提供算法基础。

## 2. 方法概述

CIST-MAE 的核心思想是：

> 随机掩码部分传感器通道，仅输入可见通道的历史序列，通过时序编码和空间建模学习多传感器物理场的时空规律，最终恢复被掩码通道的状态。

整体流程如下：

```text
Input Sensor Data
        │
        ▼
Channel Masking
        │
        ├── Visible Sensors
        │
        ▼
Temporal Encoder
因果 Conv1D + 加权池化
        │
        ▼
Spatial Encoder
位置编码 + Transformer Encoder
        │
        ▼
Spatial Decoder
恢复完整传感器拓扑
        │
        ▼
Projection Head
单点预测 + 历史序列重构
        │
        ▼
Self-supervised Reconstruction Loss
