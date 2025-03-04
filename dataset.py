import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SignalDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        :param data_dir: 
        :param transform: 
        """
        self.samples = []
        self.labels = []
        
        for label_idx, class_dir in enumerate(sorted(data_dir.glob('class_*'))):
            for signal_file in class_dir.glob('*.npy'):
                signal = np.load(signal_file)  
                self.samples.append(signal)
                self.labels.append(label_idx)
                
        self.transform = transform

    def __len__(self):
        return len(self.samples)  

    def __getitem__(self, idx):
        signal = self.samples[idx].astype(np.float32)
        label = self.labels[idx]
        
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        
        if self.transform:
            signal = self.transform(signal) 
            
        return torch.from_numpy(signal).unsqueeze(0), label  
