# 参数设置
BATCH_SIZE = 64
NUM_WORKERS = 4  # 多进程加载‌:ml-citation{ref="1,2" data="citationList"}

# 创建数据集实例
train_dataset = SignalDataset(data_dir=Path('data/train'))
test_dataset = SignalDataset(data_dir=Path('data/test'))

# 构建数据加载器‌:ml-citation{ref="3,4" data="citationList"}
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,        # 训练集需打乱顺序
    num_workers=NUM_WORKERS,
    pin_memory=True,      # 加速GPU传输
    drop_last=True        # 丢弃最后不完整批次‌:ml-citation{ref="1,2" data="citationList"}
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)
