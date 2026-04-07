import torch
from models.dataset import DataSeperate
from models.channel_masking import ChannelMasking
from models.temporal_encoder import TemporalEncoder

if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(script_dir, "data", "SensorData.xlsx")
    
    # 1. 数据预处理
    print("Testing DataSeperate...")
    ds = DataSeperate(file_path=excel_path)
    ds.process()
    x_input = ds.train_tensor
    print(f"Input shape (D, C): {x_input.shape}")
    
    # 2. 通道掩码
    print("\nTesting ChannelMasking...")
    masking = ChannelMasking(mask_ratio=0.5)
    x_visible, mask_idx, vis_idx = masking(x_input)
    print(f"Visible Data (D, c): {x_visible.shape}")
    
    # 3. 时序特征提取（自带原生地划窗分Batch）
    print("\nTesting TemporalEncoder...")
    encoder = TemporalEncoder(d_model=128, L=500, step=100)
    tokens = encoder(x_visible)
    print(f"Output tokens (BatchSize, c, D_model): {tokens.shape}")
