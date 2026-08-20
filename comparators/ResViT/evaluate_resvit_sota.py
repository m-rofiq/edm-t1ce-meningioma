# Evaluation script untuk ResViT -- ADAPTASI dari evaluate_experiment_paper.py
# (versi DD-Res U-Net). SEMUA fungsi metrik (psnr_roi, ssim_roi, calibration_bias,
# variance_ratio, gradient_difference_tumor, sharpness, lpips, bootstrap_ci)
# di-reuse IDENTIK dari modul yang sama -- supaya definisi metrik tetap
# sebanding dengan DD-Res U-Net/EDMSynth. Yang diadaptasi HANYA:
#   1. Loading data: native 3-channel (1 slice tengah per modalitas), bukan
#      9-channel triplet 2.5D.
#   2. Resize 512->256 sebelum masuk model (arsitektur ResViT wajib 256).
#   3. Inverse-transform output dari [-1,1] (Tanh) balik ke skala z-score asli,
#      LALU upsample 256->512 (bilinear) SEBELUM dihitung metrik apa pun --
#      supaya semua metrik dihitung di resolusi & ruang nilai yang identik
#      dengan DD-Res U-Net (native 512x512, z-score).

import os
import sys
import lpips
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

from bootstrap_ci import bootstrap_ci
from model_registry import MODEL_REGISTRY
from config_resvit_sota import CONFIG

from psnr_fixed_range import (
    psnr_roi,
    calibration_bias,
    variance_ratio,
    gradient_difference_tumor
)

from ssim_masked import ssim_roi
from build_enhancement_mask import build_enhancement_mask
from resvit_dataset import to_tanh_range, from_tanh_range, _load_global_range, RESVIT_IMG_SIZE

import time
from torchinfo import summary as torchinfo_summary
from fvcore.nn import FlopCountAnalysis, flop_count_table


MODALITIES_INPUT = ["T1", "T2", "FLAIR"]
MODALITY_TARGET = "T1CE"
CENTER_SLICE_IDX = 1


# ==================================================
# SHARPNESS METRICS -- identik dengan evaluate_experiment_paper.py
# ==================================================
def sharpness_tenengrad(img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img.dtype, device=img.device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=img.dtype, device=img.device
    ).view(1, 1, 3, 3)
    gx = F.conv2d(img, sobel_x, padding=1)
    gy = F.conv2d(img, sobel_y, padding=1)
    grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    mask_bool = mask.bool()
    if mask_bool.sum() == 0:
        return img.new_tensor(0.0)
    return grad_mag[mask_bool].mean()


def sharpness_ratio(pred, gt, mask):
    s_pred = sharpness_tenengrad(pred, mask)
    s_gt = sharpness_tenengrad(gt, mask)
    return s_pred / (s_gt + 1e-8)


def profile_model(model, sample_x, device, n_warmup=10, n_runs=50):
    """Identik dengan evaluate_experiment_paper.py -- profiling di resolusi
    NATIVE model (256x256), bukan 512, karena itu resolusi asli forward pass."""
    if n_warmup < 5 or n_warmup > 10:
        print(f"[profile_model] WARNING: n_warmup={n_warmup} di luar rentang rekomendasi 5-10.")
    if n_runs < 50:
        print(f"[profile_model] WARNING: n_runs={n_runs} < 50, hasil mungkin kurang stabil.")

    info = torchinfo_summary(model, input_data=sample_x, verbose=0)
    total_params = info.total_params
    trainable_params = info.trainable_params

    model.eval()
    with torch.no_grad():
        flop_analyzer = FlopCountAnalysis(model, sample_x)
        flop_analyzer.unsupported_ops_warnings(False)
        flop_analyzer.uncalled_modules_warnings(False)
        total_flops = flop_analyzer.total()
    print("\n" + flop_count_table(flop_analyzer))

    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        for _ in range(n_warmup):
            _ = model(sample_x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = model(sample_x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

    avg_time_ms = (end - start) / n_runs * 1000.0
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "flops": total_flops,
        "gflops": total_flops / 1e9,
        "n_warmup": n_warmup,
        "n_runs": n_runs,
        "avg_inference_time_ms_per_slice": avg_time_ms,
        "throughput_slices_per_sec": 1000.0 / avg_time_ms,
        "device": str(device),
        "note": "profiled at native ResViT resolution (256x256), not 512x512",
    }


def load_center_slice(path):
    """Native ResViT: ambil 1 slice tengah saja dari triplet (3,H,W) -> (1,H,W)."""
    vol = np.load(path)
    return vol[CENTER_SLICE_IDX:CENTER_SLICE_IDX + 1]


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_min, global_max = _load_global_range()

    FOLD = CONFIG["fold"]
    EXP_DIR = os.path.join(
        CONFIG["experiment_root"],
        f'{CONFIG["experiment_id"]}_{CONFIG["experiment_name"]}_fold{FOLD}'
    )
    CHECKPOINT_PATH = os.path.join(EXP_DIR, "best_model.pth")

    DATASET_ROOT = CONFIG["dataset_root"]
    HOLDOUT_PATH = os.path.join(DATASET_ROOT, "holdout_test")

    modalities = CONFIG["modalities"]
    mod_dirs = {m: os.path.join(HOLDOUT_PATH, m) for m in modalities}
    t1ce_dir = os.path.join(HOLDOUT_PATH, MODALITY_TARGET)

    files = sorted(f for f in os.listdir(t1ce_dir) if f.endswith(".npy"))

    # =========================
    # LOAD MODEL (ResViT -- in_channels = len(modalities), BUKAN 3x)
    # =========================
    model_name = CONFIG["model"]
    model = MODEL_REGISTRY[model_name](
        base_channels=CONFIG["base_channels"],
        in_channels=len(modalities),
        img_size=CONFIG["img_size"],
        vit_name=CONFIG["vit_name"],
        pretrained_vit_path=None,  # skip load ViT pretrained -- checkpoint sendiri akan menimpa semua bobot di bawah ini
    ).to(device)

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    entri = list(model.named_parameters(remove_duplicate=False))
    unik  = {p.data_ptr(): p for _, p in entri}
    print("[SHARE] entri:", len(entri), "unik:", len(unik))
    print("[SHARE] jumlah entri:", sum(p.numel() for _, p in entri))
    print("[SHARE] jumlah unik :", sum(p.numel() for p in unik.values()))

    #print("[PARAM]", sum(p.numel() for p in model.parameters()))
    #print("[BUFFER]", sum(b.numel() for b in model.buffers()))

    #from ptflops import get_model_complexity_info
    #macs, n_par = get_model_complexity_info(
    #    model, (3, 256, 256), as_strings=False, print_per_layer_stat=False)
    #print(f"[FLOPS] GFLOPs {2*macs/1e9:.2f}   params {n_par:,d}")

    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()

    # =========================
    # PROFILING (di resolusi native 256x256)
    # =========================
    sample_file = files[0]
    sample_inputs = [
        to_tanh_range(load_center_slice(os.path.join(mod_dirs[m], sample_file)),
                      global_min, global_max)
        for m in modalities
    ]
    sample_x = torch.from_numpy(np.concatenate(sample_inputs, axis=0)).unsqueeze(0).float()
    sample_x = F.interpolate(sample_x, size=(RESVIT_IMG_SIZE, RESVIT_IMG_SIZE),
                              mode="bilinear", align_corners=False).to(device)

    profile_records = profile_model(
        model, sample_x, device,
        n_warmup=CONFIG.get("profile_n_warmup", 10),
        n_runs=CONFIG.get("profile_n_runs", 50),
    )
    profile_path = os.path.join(EXP_DIR, "holdout_model_profile.csv")
    pd.DataFrame([profile_records]).to_csv(profile_path, index=False)
    print("\n===== MODEL COMPLEXITY =====\n")
    for k, v in profile_records.items():
        print(f"{k}: {v}")
    print(f"\nSaved: {profile_path}")

    # =========================
    # EVALUATION (native 512x512 -- upsample balik setelah inference)
    # =========================
    records = []

    for f in tqdm(files):
        patient_id = f.split("_slice")[0]

        # Load native 512x512 center slice per modalitas, rescale ke [-1,1]
        inputs = [
            to_tanh_range(load_center_slice(os.path.join(mod_dirs[m], f)),
                          global_min, global_max)
            for m in modalities
        ]
        x_native = torch.from_numpy(np.concatenate(inputs, axis=0)).unsqueeze(0).float().to(device)

        # Resize -> 256 (wajib untuk ResViT)
        x_256 = F.interpolate(x_native, size=(RESVIT_IMG_SIZE, RESVIT_IMG_SIZE),
                               mode="bilinear", align_corners=False)

        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                pred_256 = model(x_256)
        pred_256 = pred_256.float()

        # Upsample balik -> 512 SEBELUM inverse-transform & metrik
        pred_native_tanh = F.interpolate(pred_256, size=x_native.shape[-2:],
                                          mode="bilinear", align_corners=False)
        pred = from_tanh_range(pred_native_tanh, global_min, global_max)

        # Ground truth native 512x512, di skala z-score asli (TIDAK di-tanh-kan)
        gt_native = load_center_slice(os.path.join(t1ce_dir, f))
        gt = torch.from_numpy(gt_native).unsqueeze(0).float().to(device)

        mask = (gt != 0).float()

        # =========================
        # BASIC METRICS (identik definisi dengan DD-Res U-Net)
        # =========================
        psnr = psnr_roi(pred, gt, mask).item()
        ssim = ssim_roi(pred, gt, mask).item()
        mae = torch.mean(torch.abs(pred - gt)[mask.bool()]).item()

        with torch.no_grad():
            pred_lp = torch.clamp(pred, 0.0, 1.0)
            gt_lp = torch.clamp(gt, 0.0, 1.0)
            pred_lp = (pred_lp * 2.0) - 1.0
            gt_lp = (gt_lp * 2.0) - 1.0
            if pred_lp.shape[1] == 1:
                pred_lp = pred_lp.repeat(1, 3, 1, 1)
                gt_lp = gt_lp.repeat(1, 3, 1, 1)
            lp = lpips_model(pred_lp, gt_lp).mean().item()

        # =========================
        # ADVANCED METRICS
        # =========================
        bias = calibration_bias(pred, gt, mask).item()
        var_ratio = variance_ratio(pred, gt, mask).item()
        enh_mask = build_enhancement_mask(gt, mask)

        sharp_pred = sharpness_tenengrad(pred, mask).item()
        sharp_gt = sharpness_tenengrad(gt, mask).item()
        sharp_ratio = sharpness_ratio(pred, gt, mask).item()

        if torch.sum(enh_mask) > 10:
            psnr_enh = psnr_roi(pred, gt, enh_mask).item()
            ssim_enh = ssim_roi(pred, gt, enh_mask).item()
            grad_enh = gradient_difference_tumor(pred, gt, enh_mask).item()

            with torch.no_grad():
                enh_mask_f = enh_mask.float()
                pred_enh_lp = torch.clamp(pred * enh_mask_f, 0.0, 1.0)
                gt_enh_lp = torch.clamp(gt * enh_mask_f, 0.0, 1.0)
                pred_enh_lp = (pred_enh_lp * 2.0) - 1.0
                gt_enh_lp = (gt_enh_lp * 2.0) - 1.0
                if pred_enh_lp.shape[1] == 1:
                    pred_enh_lp = pred_enh_lp.repeat(1, 3, 1, 1)
                    gt_enh_lp = gt_enh_lp.repeat(1, 3, 1, 1)
                lpips_enh = lpips_model(pred_enh_lp, gt_enh_lp).mean().item()

            sharp_ratio_enh = sharpness_ratio(pred, gt, enh_mask).item()
        else:
            psnr_enh = ssim_enh = grad_enh = lpips_enh = sharp_ratio_enh = np.nan

        records.append({
            "patient_id": patient_id,
            "psnr_roi": psnr, "ssim_roi": ssim, "mae": mae, "lpips": lp,
            "calibration_bias": bias, "variance_ratio": var_ratio,
            "psnr_enh": psnr_enh, "ssim_enh": ssim_enh, "grad_enh": grad_enh,
            "lpips_enh": lpips_enh,
            "sharpness_pred": sharp_pred, "sharpness_gt": sharp_gt,
            "sharpness_ratio": sharp_ratio, "sharpness_ratio_enh": sharp_ratio_enh,
        })

    df = pd.DataFrame(records)
    patient_metrics = df.groupby("patient_id").mean().reset_index()
    patient_path = os.path.join(EXP_DIR, "holdout_patient_metrics.csv")
    patient_metrics.to_csv(patient_path, index=False)

    metrics = [
        "psnr_roi", "ssim_roi", "mae", "calibration_bias", "variance_ratio",
        "psnr_enh", "ssim_enh", "grad_enh", "lpips",
        "lpips_enh", "sharpness_pred", "sharpness_gt",
        "sharpness_ratio", "sharpness_ratio_enh",
    ]

    summary_records = {}
    for m in metrics:
        vals = patient_metrics[m].dropna().values
        if len(vals) == 0:
            summary_records[m + "_mean"] = np.nan
            summary_records[m + "_ci_low"] = np.nan
            summary_records[m + "_ci_high"] = np.nan
            continue
        mean, lower, upper = bootstrap_ci(vals)
        summary_records[m + "_mean"] = mean
        summary_records[m + "_ci_low"] = lower
        summary_records[m + "_ci_high"] = upper

    summary_df = pd.DataFrame([summary_records])
    summary_path = os.path.join(EXP_DIR, "holdout_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n===== HOLDOUT RESULTS (Bootstrap 95% CI) -- native 512x512 -====\n")
    for m in metrics:
        mean = summary_records[m + "_mean"]
        lo = summary_records[m + "_ci_low"]
        hi = summary_records[m + "_ci_high"]
        print(f"{m}: {mean:.4f}  (95% CI {lo:.4f} - {hi:.4f})")

    print("\nSaved:")
    print(patient_path)
    print(summary_path)


if __name__ == "__main__":
    import sys, json

    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            new_cfg = json.load(f)
        CONFIG.clear()
        CONFIG.update(new_cfg)

    main()
