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
        spilt_ratio:tuple = (0.7, 0.2, 0.1),
        
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
        # caculate len of train, val, test
        train_len = int(data_len * self.spilt_ratio[0])
        val_len = int(data_len * self.spilt_ratio[1])
        test_len = int(data_len * self.spilt_ratio[2])
        # cut data from train, val, test
        train_data = raw_data[:train_len]
        val_data = raw_data[train_len:train_len+val_len]
        test_data = raw_data[train_len+val_len:]
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