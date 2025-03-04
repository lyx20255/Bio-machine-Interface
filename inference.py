
%% 环境配置
import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

 随机种子设置
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True

%% 数据管道
class BioSignalDataset(Dataset):
    """生物医学信号数据集加载器
    
    Args:
        data_path (str): 信号数据.npy文件路径
        label_path (str): 标签数据.npy文件路径
        seq_length (int): 时间序列长度 (默认: 128)
        normalize (bool): 是否进行Z-score标准化 (默认: True)
    """
    
    def __init__(self, data_path, label_path, seq_length=128, normalize=True):
        self.signals = np.load(data_path).astype(np.float32)
        self.labels = np.load(label_path).astype(np.int64)
        
         数据预处理
        if normalize:
            self.signals = (self.signals - np.mean(self.signals)) / np.std(self.signals)
            
         转换为PyTorch张量
        self.signals = torch.from_numpy(self.signals).unsqueeze(1)  # (N, 1, L)
        self.labels = torch.from_numpy(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]

#%% 模型架构
class BioCNN1D(nn.Module):
    """1D-CNN生物信号分类器
    
    Args:
        input_channels (int): 输入通道数 (默认: 1)
        seq_length (int): 输入序列长度 (默认: 128)
        num_classes (int): 输出类别数 (默认: 5)
        dropout_prob (float): Dropout概率 (默认: 0.3)
    """
    
    def __init__(self, input_channels=1, seq_length=128, num_classes=5, dropout_prob=0.3):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            # Block 1
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_prob),
            
            # Block 2
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_prob)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * (seq_length//4), 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

%% 实验配置
class ExperimentConfig:
    """实验参数配置类"""
    
    def __init__(self, mode='BMI'):
        # 基础配置
        self.batch_size = 32
        self.num_workers = 4
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        self.patience = 5
        
        # 模式相关配置
        if mode == 'BMI':
            self.seq_length = 128
            self.num_classes = 5
            self.epochs = 25
        elif mode == 'Decouple-GC':
            self.seq_length = 20
            self.num_classes = 20
            self.epochs = 20
        else:
            raise ValueError("支持模式: 'BMI' 或 'Decouple-GC'")

%% 训练引擎
class BioTrainer:
    """模型训练验证引擎"""
    
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=config.patience
        )
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        
        for signals, labels in train_loader:
            signals = signals.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(signals)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for signals, labels in data_loader:
            signals = signals.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(signals)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            all_preds.append(outputs.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())
            
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        accuracy = (all_preds == all_labels).float().mean()
        
        return total_loss / len(data_loader), accuracy.numpy()

%% 主程序
def main(data_root, output_dir, mode='BMI'):
    # 初始化配置
    config = ExperimentConfig(mode)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 准备数据集
    full_dataset = BioSignalDataset(
        os.path.join(data_root, 'signals.npy'),
        os.path.join(data_root, 'labels.npy'),
        seq_length=config.seq_length
    )
    
    # 数据集划分
    train_size = int(0.8 * len(full_dataset))
    val_size = (len(full_dataset) - train_size) // 2
    test_size = len(full_dataset) - train_size - val_size
    
    train_set, val_set, test_set = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # 数据加载器
    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers
    )
    
    # 初始化模型
    model = BioCNN1D(
        seq_length=config.seq_length,
        num_classes=config.num_classes
    )
    trainer = BioTrainer(model, config, device)
    
     训练循环
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    
    for epoch in range(config.epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.evaluate(val_loader)
        
        trainer.scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{config.epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    结果可视化
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'training_curve.png'))
    
     最终测试
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    
     生成分类报告
    _, test_preds = trainer.evaluate(test_loader)
    print("\nClassification Report:")
    print(classification_report(test_set.dataset.labels[test_set.indices], test_preds))

if __name__ == "__main__":
     BMI传感实验
    main(
        data_root='path/to/bmi_data',
        output_dir='results/bmi',
        mode='BMI'
    )
    
     Decouple-GC实验
    main(
        data_root='path/to/decouple_data',
        output_dir='results/decouple',
        mode='Decouple-GC'
)
