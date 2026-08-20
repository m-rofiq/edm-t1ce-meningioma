"""
Holdout Evaluator — bisa digunakan untuk semua model:
  - BL-Pix2Pix
  - BL-SimpleUNet
  - EDMSynth
"""

import argparse
import torch
import numpy as np
import os
import sys
import json
import lpips
import scipy.stats as stats

# ======================================================
# [PERBAIKAN PATH]
# Mundur 1 level ke root project (04_training) agar Python 
# bisa menemukan folder 'datasets' dan 'models'
# ======================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR) 

# Pastikan yang dimasukkan ke sys.path adalah _PROJECT_ROOT, bukan _SCRIPT_DIR
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from torch.utils.data import DataLoader
from datasets.dataset_2p5d import MRI2p5DDataset
from evaluators.psnr_fixed_range import psnr_roi
from pytorch_msssim import ssim
import torch.nn.functional as F

# Dataset root dan fold
DATASET_ROOT = r"./data/dataset_5fold_final_v4"
FOLD         = 0
BATCH_SIZE   = 4
NUM_WORKERS  = 2
SEED         = 42

# ======================================================
# METRIC HELPERS
# ======================================================
def psnr_fn(pred, target, mask):
    return psnr_roi(pred, target, mask).item()

def ssim_fn(pred, target, mask):
    pred_m   = torch.clamp(pred   * mask, 0.0, 1.0)
    target_m = torch.clamp(target * mask, 0.0, 1.0)
    with torch.amp.autocast('cuda', enabled=False):
        val = ssim(pred_m.float(), target_m.float(),
                   data_range=1.0, size_average=True)
    return val.item() if torch.isfinite(val) else float('nan')

def mae_fn(pred, target, mask):
    mask_bool = mask.bool()
    if not mask_bool.any():
        return float('nan')
    return torch.abs(pred - target)[mask_bool].mean().item()

def calibration_bias_fn(pred, target, mask):
    mask_f = mask.float()
    n      = mask_f.sum().clamp(min=1.0)
    return ((pred - target) * mask_f).sum().item() / n.item()

def variance_ratio_fn(pred, target, mask):
    mask_bool = mask.bool()
    if mask_bool.sum() < 2:
        return float('nan')
    pred_v   = pred[mask_bool].var().item()
    target_v = target[mask_bool].var().item()
    if target_v < 1e-8:
        return float('nan')
    return pred_v / target_v

def psnr_enh_fn(pred, target, diff):
    enh_mask = diff > 0.025
    if not enh_mask.any():
        return float('nan')
    p = pred[enh_mask]
    t = target[enh_mask]
    mse = torch.mean((p - t) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(1.0 / mse)

def ssim_enh_fn(pred, target, diff):
    enh_mask = (diff > 0.025).float()
    if enh_mask.sum() < 10:
        return float('nan')
    pred_m   = torch.clamp(pred   * enh_mask, 0.0, 1.0)
    target_m = torch.clamp(target * enh_mask, 0.0, 1.0)
    with torch.amp.autocast('cuda', enabled=False):
        val = ssim(pred_m.float(), target_m.float(),
                   data_range=1.0, size_average=True)
    return val.item() if torch.isfinite(val) else float('nan')

def sharpness_fn(x, mask):
    kernel = torch.tensor(
        [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
        dtype=x.dtype, device=x.device
    ).view(1, 1, 3, 3)
    lap  = F.conv2d(x, kernel, padding=1)
    mb   = mask.bool()
    if not mb.any():
        return float('nan')
    return lap[mb].var().item()

def ci_95(data):
    data = [d for d in data if not np.isnan(d)]
    if len(data) < 2:
        return float('nan'), float('nan')
    n    = len(data)
    mean = np.mean(data)
    se   = stats.sem(data)
    h    = se * stats.t.ppf(0.975, df=n-1)
    return mean - h, mean + h


# ======================================================
# LOAD MODEL
# ======================================================
def load_model(model_type, model_path, device):
    in_ch = 9   # 3 modalities × 3 slices

    if model_type in ("Pix2Pix", "SimpleUNet"):
        # Import diarahkan ke folder models
        from models.baseline_models import BASELINE_REGISTRY
        model = BASELINE_REGISTRY[model_type](
            in_channels=in_ch, out_channels=1, base_ch=64
        )
    elif model_type == "EDMSynth":
        from models.model_registry import MODEL_REGISTRY
        model = MODEL_REGISTRY["EDMSynth"](
            base_channels=32, in_channels=in_ch
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    state = torch.load(model_path, map_location=device)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    print(f"  Loaded: {model_path}")
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {total:,}\n")
    return model


# ======================================================
# MAIN EVALUATION
# ======================================================
def evaluate(model_type, model_path, exp_name):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MENDETEKSI FOLDER EXPERIMENT SECARA OTOMATIS
    target_dir = os.path.dirname(os.path.abspath(model_path))

    print(f"\n{'='*60}")
    print(f"  Evaluating: {exp_name} | {model_type}")
    print(f"{'='*60}")

    model = load_model(model_type, model_path, device)

    print("[EagerInit] LPIPS...", end=" ", flush=True)
    lpips_model = lpips.LPIPS(net='alex').to(device).eval()
    print("OK")

    holdout_split = "val"
    holdout_loader = DataLoader(
        MRI2p5DDataset(os.path.join(DATASET_ROOT, f"fold_{FOLD}", holdout_split)),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    print(f"  Using split: {holdout_split}")

    psnr_list, ssim_list, mae_list, lpips_list       = [], [], [], []
    bias_list, var_list                               = [], []
    psnr_enh_list, ssim_enh_list                     = [], []
    sharp_pred_list, sharp_gt_list, sharp_ratio_list  = [], [], []

    print(f"\nRunning holdout evaluation on fold {FOLD} test set...")

    with torch.no_grad():
        for x, y, _ in holdout_loader:
            x = x.to(device)
            y = y.to(device)

            mask    = (y != 0).float()
            mask_b  = mask.bool()

            if not mask_b.any():
                continue

            with torch.amp.autocast(device_type=device.type, enabled=True):
                output = model(x)

            if isinstance(output, tuple):
                pred = output[0].float()
            else:
                pred = output.float()

            if not torch.isfinite(pred).all():
                continue

            t1   = x[:, 1:2, :, :]
            diff = y - t1

            for b in range(pred.shape[0]):
                p = pred[b:b+1]
                t = y[b:b+1]
                m = mask[b:b+1]
                d = diff[b:b+1]

                if not m.bool().any():
                    continue

                psnr_list.append(psnr_fn(p, t, m))
                ssim_list.append(ssim_fn(p, t, m))
                mae_list.append(mae_fn(p, t, m))
                bias_list.append(calibration_bias_fn(p, t, m))
                var_list.append(variance_ratio_fn(p, t, m))
                psnr_enh_list.append(psnr_enh_fn(p, t, d))
                ssim_enh_list.append(ssim_enh_fn(p, t, d))

                sp = sharpness_fn(p, m)
                sg = sharpness_fn(t, m)
                sharp_pred_list.append(sp)
                sharp_gt_list.append(sg)
                if sg > 1e-8 and not np.isnan(sp) and not np.isnan(sg):
                    sharp_ratio_list.append(sp / sg)

                pred_lp = (torch.clamp(p, 0.0, 1.0) * 2.0) - 1.0
                y_lp    = (torch.clamp(t, 0.0, 1.0) * 2.0) - 1.0
                if pred_lp.shape[1] == 1:
                    pred_lp = pred_lp.repeat(1, 3, 1, 1)
                    y_lp    = y_lp.repeat(1, 3, 1, 1)
                lpips_list.append(lpips_model(pred_lp, y_lp).mean().item())

    def agg(lst):
        lst_clean = [v for v in lst if not np.isnan(v)]
        mean = np.mean(lst_clean) if lst_clean else float('nan')
        lo, hi = ci_95(lst_clean)
        return mean, lo, hi

    results = {
        "exp_name"    : exp_name,
        "model_type"  : model_type,
        "model_path"  : model_path,
        "n_samples"   : len(psnr_list),
        "psnr_roi"    : agg(psnr_list),
        "ssim_roi"    : agg(ssim_list),
        "mae"         : agg(mae_list),
        "lpips"       : agg(lpips_list),
        "cal_bias"    : agg(bias_list),
        "var_ratio"   : agg(var_list),
        "psnr_enh"    : agg(psnr_enh_list),
        "ssim_enh"    : agg(ssim_enh_list),
        "sharp_pred"  : agg(sharp_pred_list),
        "sharp_gt"    : agg(sharp_gt_list),
        "sharp_ratio" : agg(sharp_ratio_list),
    }

    # --- print summary ---
    print(f"\n{'─'*50}")
    print(f"  {exp_name} — Holdout Results (n={results['n_samples']})")
    print(f"{'─'*50}")
    
    # PERBAIKAN: Unpacking Dictionary
    for k, v in results.items():
        if k in ("exp_name","model_type","model_path","n_samples"):
            continue
        mean, lo, hi = v
        print(f"  {k:<16}: {mean:.4f}  [{lo:.4f}, {hi:.4f}]")
    print(f"{'─'*50}\n")

    # --- save JSON ---
    out_path = os.path.join(target_dir, f"{exp_name}_holdout.json")

    save_dict = {"exp_name": exp_name, "model_type": model_type,
                 "n_samples": results["n_samples"]}
                 
    # PERBAIKAN: Unpacking Dictionary untuk JSON
    for k, v in results.items():
        if k in ("exp_name","model_type","model_path","n_samples"):
            continue
        mean, lo, hi = v
        save_dict[f"{k}_mean"] = mean
        save_dict[f"{k}_ci_low"] = lo
        save_dict[f"{k}_ci_high"] = hi

    with open(out_path, "w") as f:
        json.dump(save_dict, f, indent=2)
    print(f"  Saved: {out_path}")

    return results

# ======================================================
# CLI
# ======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True,
                        choices=["Pix2Pix", "SimpleUNet", "EDMSynth"],
                        help="Jenis model yang akan dievaluasi")
    parser.add_argument("--model_path", required=True,
                        help="Path ke best_model.pth")
    parser.add_argument("--exp_name", required=True,
                        help="Nama eksperimen (untuk output file)")
    
    # Argumen --output_dir DIBUANG agar tidak mengganggu sistem otomatis
    
    args = parser.parse_args()

    evaluate(
        model_type = args.model_type,
        model_path = args.model_path,
        exp_name   = args.exp_name
    )