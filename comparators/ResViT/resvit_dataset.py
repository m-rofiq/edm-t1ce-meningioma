# Dataset khusus untuk ResViT baseline.
#
# BERBEDA dari MRI2p5DDataset (dataset_2p5d.py) Anda: hanya mengambil SLICE
# TENGAH per modalitas (bukan triplet 3-slice), sesuai keputusan Anda untuk
# mengikuti desain native ResViT (1 channel = 1 kontras, bukan 2.5D).
#
# RESCALE ke [-1,1]: generator ResViT punya nn.Tanh() di layer output (bagian
# integral arsitektur, TIDAK diubah -- lihat penjelasan sebelumnya). Data Anda
# z-score (rentang global_min..global_max dari metric_config.json), jadi perlu
# di-rescale affine ke [-1,1] SEBELUM masuk model, dan di-inverse-transform
# KEMBALI ke skala z-score asli saat evaluasi (supaya psnr_roi dkk. tetap
# dihitung di ruang yang sama dengan DD-Res U-Net/SynDiff -- protokol
# evaluasi tetap konsisten, cuma representasi input model yang beda).

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

MODALITIES_INPUT = ["T1", "T2", "FLAIR"]
MODALITY_TARGET = "T1CE"
CENTER_SLICE_IDX = 1

# WAJIB 256, BUKAN parameter bebas: arsitektur ART_block ResViT punya grid
# transformer tetap 16x16 + upsample tetap 2x stride-2 (spasial selalu 64x64
# apapun img_size-nya), sedangkan cabang CNN downsampling 2x stride-2
# (spasial = img_size/4). Keduanya baru cocok untuk di-concat kalau
# img_size/4 == 64, yaitu img_size == 256. Ini hyperparameter arsitektural
# yang dipatok paper asli (dataset IXI mereka native 256x256), BUKAN
# sekadar ukuran yang bisa diubah bebas ke resolusi MRI native Anda (512).
RESVIT_IMG_SIZE = 256

METRIC_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "metric_config.json")


def _load_global_range():
    with open(METRIC_CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    return float(cfg["global_min"]), float(cfg["global_max"])


def to_tanh_range(x, global_min, global_max):
    """z-score -> [-1, 1], affine, dari statistik global dataset (BUKAN
    per-sample min/max -- supaya konsisten antar slice/pasien)."""
    x = (x - global_min) / (global_max - global_min)  # -> [0,1]
    return x * 2.0 - 1.0  # -> [-1,1]


def from_tanh_range(x, global_min, global_max):
    """Inverse transform: [-1,1] -> z-score asli, dipakai saat evaluasi."""
    x = (x + 1.0) / 2.0  # -> [0,1]
    return x * (global_max - global_min) + global_min


class ResViTDataset(Dataset):
    def __init__(self, root_dir):
        self.root = root_dir
        self.global_min, self.global_max = _load_global_range()

        self.mod_dirs = {m: os.path.join(root_dir, m) for m in MODALITIES_INPUT}
        self.target_dir = os.path.join(root_dir, MODALITY_TARGET)

        first_mod = MODALITIES_INPUT[0]
        self.files = sorted(
            f for f in os.listdir(self.mod_dirs[first_mod]) if f.endswith(".npy")
        )
        for f in self.files:
            for m in MODALITIES_INPUT:
                assert os.path.exists(os.path.join(self.mod_dirs[m], f)), f"Missing {m} pair for {f}"
            assert os.path.exists(os.path.join(self.target_dir, f)), f"Missing T1CE pair for {f}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        inputs = []
        for m in MODALITIES_INPUT:
            vol = np.load(os.path.join(self.mod_dirs[m], fname))  # (3, H, W)
            center = vol[CENTER_SLICE_IDX]  # (H, W) -- slice tengah saja
            inputs.append(center)

        x = np.stack(inputs, axis=0)  # (3, H, W) -- 1 channel per modalitas
        x = to_tanh_range(x, self.global_min, self.global_max)

        t1ce = np.load(os.path.join(self.target_dir, fname))
        y_native = t1ce[CENTER_SLICE_IDX:CENTER_SLICE_IDX + 1]  # (1, H, W) -- native res, disimpan untuk evaluasi
        y = to_tanh_range(y_native, self.global_min, self.global_max)

        x = torch.from_numpy(x).float().contiguous()
        y = torch.from_numpy(y).float().contiguous()

        # Resize ke 256x256 -- WAJIB untuk arsitektur ResViT (lihat catatan
        # RESVIT_IMG_SIZE di atas). bilinear untuk citra kontinu (bukan mask).
        if x.shape[-1] != RESVIT_IMG_SIZE:
            x = F.interpolate(x.unsqueeze(0), size=(RESVIT_IMG_SIZE, RESVIT_IMG_SIZE),
                               mode="bilinear", align_corners=False).squeeze(0)
            y = F.interpolate(y.unsqueeze(0), size=(RESVIT_IMG_SIZE, RESVIT_IMG_SIZE),
                               mode="bilinear", align_corners=False).squeeze(0)

        return x, y, fname
