BATCH_SIZE = 64
NUM_WORKERS = 4  

train_dataset = SignalDataset(data_dir=Path('data/train'))
test_dataset = SignalDataset(data_dir=Path('data/test'))

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,     
    num_workers=NUM_WORKERS,
    pin_memory=True,     
    drop_last=True       
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)
