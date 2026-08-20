import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import json
import os


CONFIG_PATH = os.path.join(
    (os.path.dirname(__file__)),
    "metric_config.json"
)


def load_data_range():
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    return float(cfg["data_range"])


def _force_2d(x_np):
    """
    Force input to 2D (H,W) safely.
    Handles:
    (B,C,H,W)
    (C,H,W)
    (H,W,C)
    (D,H,W)
    (H,W,D)
    (H,W)
    """

    # Remove batch if exists
    if x_np.ndim == 4:
        x_np = x_np[0]

    # If still 3D → determine which axis is depth/channel
    if x_np.ndim == 3:
        dims = x_np.shape

        # Case (C,H,W) like (3,512,512)
        if dims[0] <= 5:
            center = dims[0] // 2
            x_np = x_np[center]

        # Case (H,W,C)
        elif dims[-1] <= 5:
            center = dims[-1] // 2
            x_np = x_np[..., center]

        # Case (D,H,W)
        else:
            center = dims[0] // 2
            x_np = x_np[center]

    if x_np.ndim != 2:
        raise ValueError(f"SSIM input still not 2D. Current shape: {x_np.shape}")

    return x_np


def ssim_roi(pred, target, mask, data_range=None):

    if data_range is None:
        data_range = load_data_range()

    pred_np = pred.detach().cpu().numpy().astype(np.float64)
    target_np = target.detach().cpu().numpy().astype(np.float64)
    mask_np = mask.detach().cpu().numpy()

    pred_np = _force_2d(pred_np)
    target_np = _force_2d(target_np)
    mask_np = _force_2d(mask_np)

    mask_bool = mask_np.astype(bool)

    # Ensure win_size <= min image dimension and odd
    min_dim = min(pred_np.shape)
    win_size = min(11, min_dim if min_dim % 2 == 1 else min_dim - 1)
    if win_size < 3:
        win_size = 3

    ssim_val, ssim_map = ssim(
        target_np,
        pred_np,
        data_range=data_range,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
        win_size=win_size,
        full=True
    )

    masked_ssim = ssim_map[mask_bool]

    if masked_ssim.size == 0:
        return torch.tensor(0.0)

    return torch.tensor(masked_ssim.mean(), dtype=torch.float32)