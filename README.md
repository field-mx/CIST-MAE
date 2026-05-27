# CIST-MAE: Channel-Independent Spatio-Temporal Masked Autoencoder

面向复杂工业物理系统的多传感器时空规律学习模型。

本项目以工业制氢场景下的多通道传感器数据为研究对象，构建 **CIST-MAE（Channel-Independent Spatio-Temporal Masked Autoencoder）** 模型，用于在稀疏观测条件下学习多传感器系统的时空依赖关系，实现被掩码传感器状态预测与历史序列重构。

---

## 1. 项目背景

在工业制氢、能源装备、流程工业等复杂物理系统中，传感器数据通常具有以下特点：

- 传感器通道数量多，不同通道之间存在复杂耦合关系；
- 时间序列具有明显的动态演化规律；
- 实际工业场景中可能出现传感器缺失、采样稀疏、噪声扰动等问题；
- 系统状态难以通过全部传感器进行完整、实时、低成本观测。

因此，本项目希望解决的问题是：

> 在仅观测部分传感器通道的条件下，模型能否学习工业物理系统中的时空关联，并恢复被掩码传感器的当前状态与历史变化规律？

该问题可服务于以下应用场景：

- 工业数字孪生；
- 多传感器稀疏感知；
- 系统状态预测；
- 传感器缺失恢复；
- 异常检测与故障预警；
- 工业边缘智能部署。

---

## 2. 核心思想

CIST-MAE 的核心思想是：

> 随机掩码部分传感器通道，仅利用可见传感器的历史序列，通过时序编码和空间建模学习多传感器物理场的隐含规律，最终恢复被掩码通道的状态。

整体流程如下：

```text
Input Multi-sensor Time Series
        │
        ▼
Channel Masking
        │
        ├── Visible Sensor Channels
        │
        ▼
Temporal Encoder
Causal Conv1D + Weighted Pooling
        │
        ▼
Spatial Encoder
Sensor Positional Encoding + Transformer Encoder
        │
        ▼
Spatial Decoder
Mask Token + Full Sensor Topology Recovery
        │
        ▼
Projection Head
Point Prediction + Sequence Reconstruction
        │
        ▼
Self-supervised Reconstruction Loss
```

---

## 3. 模型结构

CIST-MAE 主要由以下模块组成：

```text
CIST-MAE
├── Channel Masking
├── Temporal Encoder
├── Spatial Encoder
├── Spatial Decoder
└── Projection Head
```

---

### 3.1 Channel Masking

Channel Masking 用于随机掩码部分传感器通道，模拟工业场景中的稀疏观测和传感器缺失情况。

原始输入数据为二维传感器时间序列表格：

```text
[D, C]
```

其中：

- `D` 表示时间步长度；
- `C` 表示传感器通道数，本项目中为 61。

经过滑动窗口切分后，模型输入为：

```text
[B, L, C]
```

其中：

- `B` 表示 batch size；
- `L` 表示历史窗口长度；
- `C` 表示传感器通道数。

训练过程中，模型随机选择一部分传感器作为可见通道，其余通道作为被掩码通道。

---

### 3.2 Temporal Encoder

Temporal Encoder 用于提取每个传感器通道的历史动态特征。

本项目采用 **通道独立时序编码机制**，即对每个传感器通道分别进行时序建模，避免不同物理量纲、不同波动尺度的传感器信号在输入阶段直接混合。

主要结构包括：

- Z-score 标准化；
- 通道独立建模；
- Causal Conv1D；
- ReLU 激活；
- 加权池化；
- 线性映射。

Temporal Encoder 的输出为：

```text
[B, c, d_model]
```

其中：

- `c` 表示可见传感器数量；
- `d_model` 表示隐空间特征维度。

采用因果卷积的原因是：  
在状态预测任务中，模型只能利用当前时刻之前的历史信息，不能引入未来信息泄露。

---

### 3.3 Spatial Encoder

Spatial Encoder 用于建模不同传感器通道之间的空间依赖关系。

主要结构包括：

- 可学习传感器位置编码；
- Transformer Encoder；
- Multi-head Self-Attention；
- Feed Forward Network；
- Dropout 正则化。

传感器位置编码用于区分不同传感器在系统中的物理角色，使模型能够学习不同通道之间的耦合关系。

Spatial Encoder 的输入为可见传感器 token：

```text
[B, c, d_model]
```

输出仍为：

```text
[B, c, d_model]
```

---

### 3.4 Spatial Decoder

Spatial Decoder 用于将可见传感器的编码结果恢复到完整传感器拓扑。

对于被掩码传感器，模型引入可学习的 **Mask Token** 表示缺失通道，并结合传感器位置编码进行空间解码。

Spatial Decoder 的输出为：

```text
[B, N, d_model]
```

其中：

- `N` 表示完整传感器数量，本项目中为 61。

该模块的目标是：  
通过可见通道的时空表示，推断被掩码通道在完整传感器系统中的状态。

---

### 3.5 Projection Head

Projection Head 采用双分支输出结构，同时完成两个任务：

#### 1. Signal Prediction Head

用于预测每个传感器窗口末端的单点状态值。

输出形状：

```text
[B, N, 1]
```

#### 2. Sequence Reconstruction Head

用于重构每个传感器的历史时序波形。

输出形状：

```text
[B, N, L]
```

双分支设计的目的在于同时约束模型学习：

- 当前状态预测能力；
- 历史动态重构能力。

---

## 4. 自监督训练目标

本项目不依赖人工标签，而是利用原始传感器数据自身构造监督信号。

训练时，模型随机掩码部分传感器通道，仅使用可见通道作为输入，并要求模型恢复被掩码通道的状态。

损失函数由两部分组成：

```text
Loss = λ × Loss_signal + (1 - λ) × Loss_sequence
```

其中：

- `Loss_signal`：单点状态预测损失；
- `Loss_sequence`：历史序列重构损失；
- `λ`：单点预测损失权重。

训练过程中，损失主要在被掩码传感器通道上计算，使模型学习如何从部分可见传感器推断完整系统状态。

---

## 5. 项目特点

### 5.1 通道独立时序编码

模型对每个传感器通道独立提取历史动态特征，减少不同物理量纲和信号尺度之间的直接干扰。

### 5.2 因果时序建模

采用 Causal Conv1D 提取时间因果特征，避免未来信息泄露，适合实际状态预测场景。

### 5.3 Transformer 空间建模

通过传感器位置编码和 Transformer Encoder 捕获跨通道依赖关系，建模复杂工业系统中的空间耦合。

### 5.4 自监督掩码重构

无需人工标注，直接基于原始传感器数据进行自监督学习，适合工业场景中标签稀缺的问题。

### 5.5 双分支预测输出

同时完成单点状态预测和历史序列重构，使模型兼具状态估计和动态规律学习能力。

### 5.6 轻量化模型设计

模型参数量约为 39 万，具有进一步部署到工业边缘端的潜力。

---

## 6. 仓库结构

```text
CIST-MAE/
├── checkpoints/              # 模型权重保存目录
├── configs/                  # 配置文件
│   └── default.yaml
├── data/                     # 数据目录
├── models/                   # 模型模块
│   ├── channel_masking.py    # 通道掩码模块
│   ├── cist_mae.py           # CIST-MAE 主模型
│   ├── dataset.py            # 数据处理模块
│   ├── projection_head.py    # 双分支预测头
│   ├── spatial_decoder.py    # 空间解码器
│   ├── spatial_encoder.py    # 空间编码器
│   └── temporal_encoder.py   # 时序编码器
├── utils/                    # 工具函数
├── train.py                  # 训练入口
├── evaluate.py               # 评估入口
├── test_pipeline.py          # 测试脚本
├── requirements.txt          # Python 依赖
└── README.md                 # 项目说明文档
```

---

## 7. 环境依赖

建议使用 Python 3.9 或更高版本。

安装依赖：

```bash
pip install -r requirements.txt
```

主要依赖包括：

```text
torch
numpy
pandas
pyyaml
matplotlib
tensorboard
tqdm
scikit-learn
```

---

## 8. 数据格式

默认数据文件路径：

```text
data/SensorData.xlsx
```

输入数据应为二维传感器时间序列表格：

```text
[D, C]
```

其中：

- 每一行表示一个时间步；
- 每一列表示一个传感器通道；
- 本项目默认传感器通道数为 61。

示例：

| Time | Sensor 1 | Sensor 2 | ... | Sensor 61 |
|---|---:|---:|---:|---:|
| t1 | 0.12 | 1.32 | ... | 0.45 |
| t2 | 0.15 | 1.29 | ... | 0.47 |
| t3 | 0.14 | 1.31 | ... | 0.46 |

---

## 9. 训练方法

运行训练脚本：

```bash
python train.py
```

默认训练配置示例：

```text
num_sensors = 61
d_model = 128
window_length = 500
batch_size = 64
mask_ratio = 0.3
epochs = 500
learning_rate = 1e-3
weight_decay = 1e-2
grad_clip = 1.0
```

训练流程包括：

1. 加载多传感器时间序列数据；
2. 进行数据标准化与窗口切分；
3. 随机掩码部分传感器通道；
4. 构建 CIST-MAE 模型；
5. 使用 AdamW 优化器训练；
6. 使用 CosineAnnealingLR 调整学习率；
7. 保存验证集损失最优模型；
8. 使用 Early Stopping 防止过拟合。

最优模型默认保存为：

```text
checkpoints/best.pth
```

---

## 10. 评估方法

运行评估脚本：

```bash
python evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pth
```

评估指标包括：

- MSE；
- RMSE；
- MAE；
- MAPE；
- R²。

---

## 11. 实验结果

当前模型在工业制氢 61 通道传感器数据上进行了稀疏观测重建实验。

| Mask Ratio | Sensors | Parameters | Task | Result |
|---:|---:|---:|---|---:|
| 0.3 | 61 | ~0.39M | 单点预测 + 序列重构 | 待补充 |
| 0.5 | 61 | ~0.39M | 单点预测 + 序列重构 | 待补充 |
| 0.7 | 61 | ~0.39M | 单点预测 + 序列重构 | 87% |

> 注：请根据实际实验日志补充不同掩码率下的 MAE、RMSE、R²、预测精度等指标。

---

## 12. 可视化结果

建议在仓库中补充以下可视化结果：

1. 真实值与预测值对比曲线；
2. 不同传感器通道的重构误差热图；
3. 不同 mask ratio 下的性能变化曲线；
4. 被掩码通道的历史序列重构结果；
5. 消融实验对比图。

推荐目录结构：

```text
assets/
├── architecture.png
├── reconstruction_curve.png
├── mask_ratio_result.png
└── ablation_result.png
```

在 README 中可按照以下方式插入图片：

```markdown
![Architecture](assets/architecture.png)

![Reconstruction Curve](assets/reconstruction_curve.png)

![Mask Ratio Result](assets/mask_ratio_result.png)
```

---

## 13. 消融实验建议

为了进一步验证各模块有效性，建议补充以下消融实验：

| Variant | Temporal Encoder | Spatial Encoder | Mask Token | Dual Head | Result |
|---|---|---|---|---|---:|
| Full CIST-MAE | Causal Conv1D | Transformer | Yes | Yes | 待补充 |
| w/o Temporal Encoder | - | Transformer | Yes | Yes | 待补充 |
| w/o Spatial Encoder | Causal Conv1D | - | Yes | Yes | 待补充 |
| w/o Mask Token | Causal Conv1D | Transformer | - | Yes | 待补充 |
| w/o Sequence Head | Causal Conv1D | Transformer | Yes | - | 待补充 |

---

## 14. 后续工作

后续可进一步开展以下研究：

- 与 LSTM、GRU、TCN、Transformer、MAE 等基线模型进行系统对比；
- 增加不同工况下的跨场景泛化实验；
- 引入传感器物理拓扑先验，提升空间建模解释性；
- 研究在线推理与工业边缘端部署；
- 将重构误差用于异常检测、健康评估和故障预警；
- 构建面向工业数字孪生的多传感器自监督预训练模型。

---

## 15. 项目应用价值

CIST-MAE 可应用于：

- 工业设备状态预测；
- 多传感器稀疏感知；
- 数字孪生状态重建；
- 传感器缺失恢复；
- 异常检测与故障预警；
- 工业边缘智能部署。

---

## 16. 作品描述

CIST-MAE 是一个面向复杂工业物理系统的多传感器时空掩码自编码模型。项目通过随机掩码部分传感器通道，仅利用可见通道的历史序列学习系统时空关联，并恢复被掩码传感器的当前状态与历史波形。模型采用因果 Conv1D 提取单通道时序特征，结合传感器位置编码和 Transformer Encoder 建模跨通道空间依赖，最终通过双分支预测头完成单点预测与序列重构。该方法适用于工业数字孪生、稀疏感知、状态预测、异常检测和故障预警等任务。



---

## 17. Citation

如需引用本项目，可使用：

```bibtex
@misc{cistmae2026,
  title  = {CIST-MAE: Channel-Independent Spatio-Temporal Masked Autoencoder for Multi-sensor Physical Field Learning},
  author = {Xiang Mu},
  year   = {2026},
  url    = {https://github.com/field-mx/CIST-MAE}
}
```

---

## 18. License

本项目目前主要用于科研学习与实验验证。  
如需用于商业或工程部署，请联系作者。
