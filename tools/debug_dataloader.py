import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader
from datasets.dataset_2p5d import MRI2p5DDataset

DATASET_ROOT = r"./data/dataset_5fold_final_v4"
FOLD = 4

TRAIN_PATH = os.path.join(DATASET_ROOT, f"fold_{FOLD}", "train")

dataset = MRI2p5DDataset(TRAIN_PATH)

loader = DataLoader(dataset, batch_size=2)

x, y, name = next(iter(loader))

print("INPUT SHAPE :", x.shape)
print("TARGET SHAPE:", y.shape)

print("INPUT MIN MAX:", x.min().item(), x.max().item())
print("TARGET MIN MAX:", y.min().item(), y.max().item())

print("FILENAME:", name)
