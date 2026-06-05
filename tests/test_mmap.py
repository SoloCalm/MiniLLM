"""独立测试脚本：验证 PretrainDatasetMmap 是否正常工作"""
import sys
sys.path.insert(0, ".")

import numpy as np
import torch
from pathlib import Path

print("Step 1: import done")

from training.data_loader import PretrainDatasetMmap, create_dataloader
print("Step 2: import data_loader done")

# 加载小数据集
ds = PretrainDatasetMmap(Path("data/pretrain_tokenized_small/train_ids.npy"), max_length=256)
print(f"Step 3: Dataset size = {len(ds)}")

# 取一个样本
sample = ds[0]
print(f"Step 4: input_ids shape = {sample['input_ids'].shape}")

# 创建 DataLoader
dl = create_dataloader(ds, batch_size=2)
print("Step 5: DataLoader created")

batch = next(iter(dl))
print(f"Step 6: batch shape = {batch['input_ids'].shape}")

print("ALL PASSED!")
