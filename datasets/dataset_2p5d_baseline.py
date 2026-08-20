import os
import numpy as np
import torch
from torch.utils.data import Dataset


class MRI2p5DDataset(Dataset):
    def __init__(self, root_dir):
        self.root = root_dir

        self.t1_dir = os.path.join(root_dir, "T1")
        self.t1ce_dir = os.path.join(root_dir, "T1CE")

        self.files = sorted(f for f in os.listdir(self.t1_dir) if f.endswith(".npy"))

        for f in self.files:
            assert os.path.exists(os.path.join(self.t1ce_dir, f)), f"Missing pair for {f}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        fname = self.files[idx]

        t1 = np.load(os.path.join(self.t1_dir, fname))
        t1ce = np.load(os.path.join(self.t1ce_dir, fname))

        assert t1.shape == (3,512,512)
        assert t1ce.shape == (3,512,512)

        center = 1
        t1ce_center = t1ce[center:center+1]

        #t1 = torch.from_numpy(t1).float()
        #t1ce_center = torch.from_numpy(t1ce_center).float()

        t1 = torch.from_numpy(t1).float().contiguous()
        t1ce_center = torch.from_numpy(t1ce_center).float().contiguous()

        return t1, t1ce_center, fname