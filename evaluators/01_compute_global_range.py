import os
import numpy as np
from tqdm import tqdm

ROOT = r"./data/dataset_5fold_final_v2"

global_min = float("inf")
global_max = float("-inf")

for fold in os.listdir(ROOT):

    if not fold.startswith("fold_"):
        continue

    train_path = os.path.join(ROOT, fold, "train", "T1CE")

    if not os.path.exists(train_path):
        continue

    for f in tqdm(os.listdir(train_path), desc=f"{fold}"):

        if not f.endswith(".npy"):
            continue

        arr = np.load(os.path.join(train_path, f))

        assert arr.shape == (3,512,512)

        center = arr[1]

        global_min = min(global_min, center.min())
        global_max = max(global_max, center.max())

data_range = global_max - global_min

print("\n===== GLOBAL RANGE (TRAIN ONLY) =====")
print("GLOBAL MIN:", global_min)
print("GLOBAL MAX:", global_max)
print("FIXED DATA_RANGE:", data_range)