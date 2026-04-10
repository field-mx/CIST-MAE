# 数据集划分 
# 输入：初始excel 划分比例
# 输出：标准化的训练集 验证集 测试集 
import pandas as pd
import torch
import numpy as np

class DataSeperate:
    def __init__(
        self,
        file_path:str = ".\data\SensorData.xlsx",
        spilt_ratio:tuple = (0.8, 0.1, 0.1),
        
    ):
        self.file_path = file_path
        self.spilt_ratio = spilt_ratio
        
        # def real data scope
        self.row_start = 3
        self.row_end = 24237
        self.col_start = 5
        self.col_end = 66
        # def empty data tensor
        self.train_tensor = None
        self.val_tensor = None
        self.test_tensor = None

    # data process :z-score initial
    def process(self) -> tuple:
        # read data
        df = pd.read_excel(self.file_path, header = None)
        # cut data from file
        raw_data = df.iloc[self.row_start:self.row_end, self.col_start:self.col_end].values.astype(np.float32)
        data_len = raw_data.shape[0]
        
        # 区间块分配：将时间序列切成连续块，随机分配给 train/val/test
        # 每个块内部保持时间连续性（卷积需要），块之间随机分配（消除时间漂移）
        block_size = 1000  # 每个块的时间步长度
        num_blocks = data_len // block_size
        # 将数据切成若干完整的块
        blocks = [raw_data[i * block_size : (i + 1) * block_size] for i in range(num_blocks)]
        # 随机打散块的顺序
        np.random.seed(42)
        perm = np.random.permutation(num_blocks)
        blocks = [blocks[i] for i in perm]
        # 按比例分配块
        train_n = int(num_blocks * self.spilt_ratio[0])
        val_n = int(num_blocks * self.spilt_ratio[1])
        # 拼接各集合的块
        train_data = np.concatenate(blocks[:train_n], axis=0)
        val_data = np.concatenate(blocks[train_n:train_n + val_n], axis=0)
        test_data = np.concatenate(blocks[train_n + val_n:], axis=0)
        # z-score normalization（用训练集的统计量标准化全部数据）
        train_mean = train_data.mean(axis=0)
        train_std = train_data.std(axis=0)
        train_std[train_std == 0] = 1
        train_data = (train_data - train_mean) / train_std
        val_data = (val_data - train_mean) / train_std
        test_data = (test_data - train_mean) / train_std

        # 保存标准化参数（部署时反标准化需要）
        self.train_mean = torch.from_numpy(train_mean)  # [C]
        self.train_std = torch.from_numpy(train_std)    # [C]

        # change to tensor and save new tensor
        self.train_tensor = torch.from_numpy(train_data)
        self.val_tensor = torch.from_numpy(val_data)
        self.test_tensor = torch.from_numpy(test_data)

"""
if __name__ == "__main__":
    # 直接运行本文件时执行
    import os
    # 定位到项目根目录下的 data 文件夹
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    excel_path = os.path.join(project_root, "data", "SensorData.xlsx")

    ds = DataSeperate(file_path=excel_path)  # 创建一个实例
    ds.process()                              # 处理数据
    print("train:", ds.train_tensor.shape)    # 用同一个实例获取结果
    print("val:  ", ds.val_tensor.shape)
    print("test: ", ds.test_tensor.shape)  
    print("hello")     
"""