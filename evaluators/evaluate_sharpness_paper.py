import os
import sys
import lpips

# =========================
# ADD PROJECT ROOT
# =========================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

from evaluators.bootstrap_ci import bootstrap_ci
from models.model_registry import MODEL_REGISTRY
from configs.config import CONFIG

from evaluators.psnr_fixed_range import (
    psnr_roi,
    calibration_bias,
    variance_ratio,
    gradient_difference_tumor
)

from evaluators.ssim_masked import ssim_roi
from evaluators.build_enhancement_mask import build_enhancement_mask


# ==================================================
# SHARPNESS METRICS
# --------------------------------------------------
# Mengukur ketajaman prediksi secara langsung.
# Kritis untuk membuktikan/mengukur blur antar
# iterasi eksperimen dan sebagai pendukung kuantitatif
# untuk diskusi grad_enh (arah bias, bukan cuma selisih).
#
# sharpness_tenengrad:
#   Rata-rata gradient magnitude (Sobel) di dalam
#   brain mask. Nilai lebih tinggi = lebih tajam.
#   Referensi: Krotkov & Martin, IJCV 1993.
#
# sharpness_ratio:
#   sharpness(pred) / sharpness(gt)
#   Nilai ideal = 1.0. < 1.0 berarti pred lebih
#   blur dari GT. Ini yang paling informatif untuk
#   monitoring progress antar eksperimen.
# ==================================================
def sharpness_tenengrad(img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Tenengrad sharpness: mean Sobel gradient magnitude inside mask.
    img  : [1, 1, H, W]  float, range [0, 1]
    mask : [1, 1, H, W]  float (0/1) atau bool
    """
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=img.dtype, device=img.device
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=img.dtype, device=img.device
    ).view(1, 1, 3, 3)

    gx = F.conv2d(img, sobel_x, padding=1)
    gy = F.conv2d(img, sobel_y, padding=1)
    grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    mask_bool = mask.bool()
    if mask_bool.sum() == 0:
        return img.new_tensor(0.0)

    return grad_mag[mask_bool].mean()


def sharpness_ratio(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    pred sharpness / gt sharpness.
    Ideal = 1.0. Nilai < 1.0 → pred lebih blur dari GT.
    """
    s_pred = sharpness_tenengrad(pred, mask)
    s_gt   = sharpness_tenengrad(gt, mask)
    return s_pred / (s_gt + 1e-8)


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================
    # PATH
    # =========================

    FOLD = CONFIG["fold"]

    EXP_DIR = os.path.join(
        CONFIG["experiment_root"],
        f'{CONFIG["experiment_id"]}_{CONFIG["experiment_name"]}_fold{FOLD}'
    )

    CHECKPOINT_PATH = os.path.join(EXP_DIR, "best_model.pth")

    DATASET_ROOT = CONFIG["dataset_root"]

    HOLDOUT_PATH = os.path.join(DATASET_ROOT, "holdout_test")

    modalities = CONFIG["modalities"]

    mod_dirs = {
        m: os.path.join(HOLDOUT_PATH, m) for m in modalities
    }

    t1ce_dir = os.path.join(HOLDOUT_PATH, "T1CE")

    files = sorted([
        f for f in os.listdir(t1ce_dir)
        if f.endswith(".npy")
    ])

    # =========================
    # LOAD MODEL
    # =========================
    model_name = CONFIG["model"]

    if model_name in ["DMECNetStep1", "DMECNetStep2", "DMECNetStep3"]:
        model = MODEL_REGISTRY[model_name](
            use_t2    = "T2"    in CONFIG["modalities"],
            use_flair = "FLAIR" in CONFIG["modalities"],
            base_ch   = CONFIG["base_channels"]
        ).to(device)
    else:
        model = MODEL_REGISTRY[model_name](
            base_channels = CONFIG["base_channels"],
            in_channels   = 3 * len(CONFIG["modalities"])
        ).to(device)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    # [FIX] model.eval() dipindah keluar dari percabangan.
    # Sebelumnya hanya dipanggil di cabang 'else', sehingga kalau
    # checkpoint disimpan sebagai dict berisi key "model_state_dict",
    # model tetap dalam mode training saat inferensi (mempengaruhi
    # layer seperti InstanceNorm/Dropout) - membuat hasil sharpness/
    # psnr/ssim tidak konsisten dengan script evaluasi lain yang
    # sudah benar memanggil eval() tanpa syarat.
    model.eval()

    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()

    # =========================
    # EVALUATION
    # =========================

    records = []

    for f in tqdm(files):

        patient_id = f.split("_slice")[0]

        inputs = []

        for m in modalities:
            vol = np.load(os.path.join(mod_dirs[m], f))
            inputs.append(vol)

        x_np = np.concatenate(inputs, axis=0)

        t1ce = np.load(os.path.join(t1ce_dir, f))

        gt_slice = t1ce[1]

        x  = torch.from_numpy(x_np).unsqueeze(0).float().to(device)
        gt = torch.from_numpy(gt_slice).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda" and CONFIG["amp"])):
                output = model(x)

        if isinstance(output, tuple):
            pred = output[0]
        else:
            pred = output

        pred = pred.float()

        mask      = (gt != 0).float()
        mask_bool = mask.bool()

        # =========================
        # BASIC METRICS
        # =========================

        psnr_val = psnr_roi(pred, gt, mask).item()
        ssim_val = ssim_roi(pred, gt, mask).item()
        mae_val  = torch.mean(torch.abs(pred - gt)[mask_bool]).item()

        with torch.no_grad():
            pred_lp = torch.clamp(pred, 0.0, 1.0)
            gt_lp   = torch.clamp(gt,   0.0, 1.0)

            pred_lp = (pred_lp * 2.0) - 1.0
            gt_lp   = (gt_lp   * 2.0) - 1.0

            if pred_lp.shape[1] == 1:
                pred_lp = pred_lp.repeat(1, 3, 1, 1)
                gt_lp   = gt_lp.repeat(1,   3, 1, 1)

            lp = lpips_model(pred_lp, gt_lp).mean().item()

        # =========================
        # ADVANCED METRICS
        # =========================

        bias      = calibration_bias(pred, gt, mask).item()
        var_ratio = variance_ratio(pred, gt, mask).item()

        # Sharpness metrics (global, inside brain mask)
        sharp_pred  = sharpness_tenengrad(pred, mask).item()
        sharp_gt    = sharpness_tenengrad(gt,   mask).item()
        sharp_ratio = sharpness_ratio(pred, gt, mask).item()

        enh_mask = build_enhancement_mask(gt, mask)

        if torch.sum(enh_mask) > 10:

            psnr_enh_val = psnr_roi(pred, gt, enh_mask).item()
            ssim_enh_val = ssim_roi(pred, gt, enh_mask).item()
            grad_enh_val = gradient_difference_tumor(pred, gt, enh_mask).item()

            # LPIPS di enhancement region secara spesifik
            with torch.no_grad():
                enh_mask_f = enh_mask.float()

                pred_enh_lp = torch.clamp(pred * enh_mask_f, 0.0, 1.0)
                gt_enh_lp   = torch.clamp(gt   * enh_mask_f, 0.0, 1.0)

                pred_enh_lp = (pred_enh_lp * 2.0) - 1.0
                gt_enh_lp   = (gt_enh_lp   * 2.0) - 1.0

                if pred_enh_lp.shape[1] == 1:
                    pred_enh_lp = pred_enh_lp.repeat(1, 3, 1, 1)
                    gt_enh_lp   = gt_enh_lp.repeat(1,   3, 1, 1)

                lpips_enh_val = lpips_model(pred_enh_lp, gt_enh_lp).mean().item()

            # Sharpness ratio di enhancement region
            sharp_ratio_enh = sharpness_ratio(pred, gt, enh_mask).item()

        else:
            psnr_enh_val    = np.nan
            ssim_enh_val    = np.nan
            grad_enh_val    = np.nan
            lpips_enh_val   = np.nan
            sharp_ratio_enh = np.nan

        records.append({
            "patient_id": patient_id,

            # Basic
            "psnr_roi"          : psnr_val,
            "ssim_roi"          : ssim_val,
            "mae"               : mae_val,
            "lpips"             : lp,

            # Calibration
            "calibration_bias"  : bias,
            "variance_ratio"    : var_ratio,

            # Enhancement region
            "psnr_enh"          : psnr_enh_val,
            "ssim_enh"          : ssim_enh_val,
            "grad_enh"          : grad_enh_val,
            "lpips_enh"         : lpips_enh_val,

            # Sharpness
            "sharpness_pred"    : sharp_pred,
            "sharpness_gt"      : sharp_gt,
            "sharpness_ratio"   : sharp_ratio,
            "sharpness_ratio_enh": sharp_ratio_enh,
        })

    # =========================
    # SAVE PATIENT METRICS
    # =========================

    df = pd.DataFrame(records)

    patient_metrics = df.groupby("patient_id").mean().reset_index()

    patient_path = os.path.join(EXP_DIR, "holdout_patient_metrics.csv")

    patient_metrics.to_csv(patient_path, index=False)

    # =========================
    # BOOTSTRAP SUMMARY
    # =========================

    metrics = [
        # Basic
        "psnr_roi",
        "ssim_roi",
        "mae",
        "lpips",
        # Calibration
        "calibration_bias",
        "variance_ratio",
        # Enhancement region
        "psnr_enh",
        "ssim_enh",
        "grad_enh",
        "lpips_enh",
        # Sharpness
        "sharpness_pred",
        "sharpness_gt",
        "sharpness_ratio",
        "sharpness_ratio_enh",
    ]

    summary_records = {}

    for m in metrics:

        vals = patient_metrics[m].dropna().values

        if len(vals) == 0:
            summary_records[m + "_mean"]     = np.nan
            summary_records[m + "_ci_low"]   = np.nan
            summary_records[m + "_ci_high"]  = np.nan
            continue

        mean, lower, upper = bootstrap_ci(vals)

        summary_records[m + "_mean"]    = mean
        summary_records[m + "_ci_low"]  = lower
        summary_records[m + "_ci_high"] = upper

    summary_df = pd.DataFrame([summary_records])

    summary_path = os.path.join(EXP_DIR, "holdout_summary.csv")

    summary_df.to_csv(summary_path, index=False)

    # =========================
    # PRINT RESULTS
    # =========================

    print("\n===== HOLDOUT RESULTS (Bootstrap 95% CI) =====\n")

    groups = {
        "── Basic": [
            "psnr_roi", "ssim_roi", "mae", "lpips"
        ],
        "── Calibration": [
            "calibration_bias", "variance_ratio"
        ],
        "── Enhancement Region": [
            "psnr_enh", "ssim_enh", "grad_enh", "lpips_enh"
        ],
        "── Sharpness": [
            "sharpness_pred", "sharpness_gt",
            "sharpness_ratio", "sharpness_ratio_enh"
        ],
    }

    for group_label, group_metrics in groups.items():
        print(group_label)
        for m in group_metrics:
            mean = summary_records[m + "_mean"]
            lo   = summary_records[m + "_ci_low"]
            hi   = summary_records[m + "_ci_high"]
            if np.isnan(mean):
                print(f"  {m:<25}: N/A")
            else:
                print(f"  {m:<25}: {mean:.4f}  (95% CI {lo:.4f} – {hi:.4f})")
        print()

    print("Saved:")
    print(patient_path)
    print(summary_path)


if __name__ == "__main__":
    import sys
    import json
    from configs.config import CONFIG

    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            new_cfg = json.load(f)
        CONFIG.clear()
        CONFIG.update(new_cfg)

    main()
