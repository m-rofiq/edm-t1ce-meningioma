import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MRIIdentityDataset(Dataset):

    def __init__(self, root_dir):

        self.root = root_dir
        self.t1_dir = os.path.join(root_dir, "T1")

        self.files = sorted(
            f for f in os.listdir(self.t1_dir) if f.endswith(".npy")
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        fname = self.files[idx]

        t1 = np.load(os.path.join(self.t1_dir, fname))

        assert t1.shape == (3,512,512)

        center = 1
        target = t1[center:center+1]

        t1 = torch.from_numpy(t1).float()
        target = torch.from_numpy(target).float()

        return t1, target, fname