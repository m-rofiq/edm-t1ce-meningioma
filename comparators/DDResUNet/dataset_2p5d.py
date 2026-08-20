import os
import numpy as np
import torch
from torch.utils.data import Dataset
from config_ddresunet_sota import CONFIG


class MRI2p5DDataset(Dataset):

    def __init__(self, root_dir):

        self.root = root_dir
        self.modalities = CONFIG["modalities"]

        self.mod_dirs = {
            m: os.path.join(root_dir, m) for m in self.modalities
        }

        self.target_dir = os.path.join(root_dir, "T1CE")

        first_mod = self.modalities[0]

        self.files = sorted(
            f for f in os.listdir(self.mod_dirs[first_mod])
            if f.endswith(".npy")
        )

        for f in self.files:
            assert os.path.exists(os.path.join(self.target_dir, f)), f"Missing pair for {f}"


    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):

        fname = self.files[idx]

        inputs = []

        for m in self.modalities:

            vol = np.load(os.path.join(self.mod_dirs[m], fname))

            assert vol.shape == (3,512,512)

            inputs.append(vol)

        x = np.concatenate(inputs, axis=0)

        t1ce = np.load(os.path.join(self.target_dir, fname))

        assert t1ce.shape == (3,512,512)

        center = 1
        y = t1ce[center:center+1]

        x = torch.from_numpy(x).float().contiguous()
        y = torch.from_numpy(y).float().contiguous()

        return x, y, fname