# -*- coding: utf-8 -*-
"""
数据和日志保存模块
用于在训练过程中和训练结束后保存各项数据。
"""
import os
import pandas as pd
from datetime import datetime

class DataSaver:
    def __init__(self, base_dir="results"):
        """
        初始化数据保存类
        :param base_dir: 数据保存的根目录
        """
        self.base_dir = base_dir
        # 自动创建保存日志的目录
        os.makedirs(self.base_dir, exist_ok=True)
        
    def get_experiment_name(self, **kwargs):
        """
        根据传入的参数动态生成前缀，加上时间戳返回唯一且具有标识性的名字。
        例如传入 mask_ratio=0.75, d_model=256
        生成: m0.75_d256_20260413_201049
        """
        parts = []
        for key, value in kwargs.items():
            if key == "mask_ratio":
                parts.append(f"m{value}")
            elif key == "d_model":
                parts.append(f"d{value}")
            elif key == "lambda_signal":
                parts.append(f"lsig{value}")
            elif key == "epochs":
                parts.append(f"ep{value}")
            else:
                parts.append(f"{key}{value}")
                
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "_".join(parts)
        if prefix:
            return f"{prefix}_{timestamp}"
        return f"exp_{timestamp}"

    def save_loss_to_csv(self, loss_history: dict, filename: str):
        """
        将训练损失历史保存为 CSV 文件。
        :param loss_history: 字典，键通常为 ['epoch', 'train_loss', 'val_loss', 'lr']，值为对应列表
        :param filename: CSV 文件的名称（不包含后缀也会自动补全）
        """
        if not filename.endswith(".csv"):
            filename += ".csv"
            
        save_path = os.path.join(self.base_dir, filename)
        df = pd.DataFrame(loss_history)
        df.to_csv(save_path, index=False)
        print(f"\n[DataSaver] 损失历史已保存至: {save_path}")

    # ================= 预留接口 ================= #

    def save_metrics_plot(self, loss_history: dict, filename: str):
        """
        [预留] 绘制并保存训练指标曲线(如 Loss 曲线图)
        """
        pass

    def save_model_config(self, config: dict, filename: str):
        """
        [预留] 保存模型超参数配置结构(如保存为 JSON)
        """
        pass
        
    def save_predictions(self, y_true, y_pred, filename: str):
        """
        [预留] 保存测试集的预测结果与真实值，以便后续绘制对比图
        """
        pass
