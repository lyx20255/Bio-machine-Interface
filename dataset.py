import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SignalDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        一维信号数据集构造
        :param data_dir: 数据目录，按类别存储.npy文件 (如class_0/, class_1/)
        :param transform: 数据增强函数
        """
        self.samples = []
        self.labels = []
        
        # 遍历类别文件夹加载数据‌:ml-citation{ref="4,6" data="citationList"}
        for label_idx, class_dir in enumerate(sorted(data_dir.glob('class_*'))):
            for signal_file in class_dir.glob('*.npy'):
                signal = np.load(signal_file)  # 加载一维numpy数组
                self.samples.append(signal)
                self.labels.append(label_idx)
                
        self.transform = transform

    def __len__(self):
        return len(self.samples)  # 必须实现的方法‌:ml-citation{ref="5" data="citationList"}

    def __getitem__(self, idx):
        signal = self.samples[idx].astype(np.float32)
        label = self.labels[idx]
        
        # 数据预处理：归一化到[-1,1]‌:ml-citation{ref="4" data="citationList"}
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        
        if self.transform:
            signal = self.transform(signal)  # 应用数据增强
            
        return torch.from_numpy(signal).unsqueeze(0), label  # 输出形状(1, seq_len)‌:ml-citation{ref="1,2" data="citationList"}
