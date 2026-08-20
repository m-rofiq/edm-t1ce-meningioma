import os
import sys

# =========================
# ADD PROJECT ROOT
# =========================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
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
            use_t2 = "T2" in CONFIG["modalities"],
            use_flair = "FLAIR" in CONFIG["modalities"],
            base_ch = CONFIG["base_channels"]
        ).to(device)

    else:
        model = MODEL_REGISTRY[model_name](
            base_channels=CONFIG["base_channels"],
            in_channels = 3 * len(CONFIG["modalities"])
        ).to(device)

    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=device)
    )

    model.eval()


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

        x = torch.from_numpy(x_np).unsqueeze(0).float().to(device)

        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=(device.type=="cuda" and CONFIG["amp"])):
                output = model(x)

        if isinstance(output, tuple):
            pred = output[0]
        else:
            pred = output

        pred = pred.float()

        gt = torch.from_numpy(gt_slice).unsqueeze(0).unsqueeze(0).float().to(device)

        mask = (gt != 0).float()

        # =========================
        # BASIC METRICS
        # =========================

        psnr = psnr_roi(pred, gt, mask).item()
        ssim = ssim_roi(pred, gt, mask).item()
        mae = torch.mean(torch.abs(pred - gt)[mask == 1]).item()

        # =========================
        # ADVANCED METRICS
        # =========================

        bias = calibration_bias(pred, gt, mask).item()
        var_ratio = variance_ratio(pred, gt, mask).item()

        enh_mask = build_enhancement_mask(gt, mask)

        if torch.sum(enh_mask) > 10:

            psnr_enh = psnr_roi(pred, gt, enh_mask).item()
            ssim_enh = ssim_roi(pred, gt, enh_mask).item()
            grad_enh = gradient_difference_tumor(pred, gt, enh_mask).item()

        else:

            psnr_enh = np.nan
            ssim_enh = np.nan
            grad_enh = np.nan


        records.append({

            "patient_id": patient_id,

            "psnr_roi": psnr,
            "ssim_roi": ssim,
            "mae": mae,

            "calibration_bias": bias,
            "variance_ratio": var_ratio,

            "psnr_enh": psnr_enh,
            "ssim_enh": ssim_enh,
            "grad_enh": grad_enh
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
        "psnr_roi",
        "ssim_roi",
        "mae",
        "calibration_bias",
        "variance_ratio",
        "psnr_enh",
        "ssim_enh",
        "grad_enh"
    ]

    summary_records = {}

    for m in metrics:

        vals = patient_metrics[m].dropna().values

        if len(vals) == 0:
            summary_records[m+"_mean"] = np.nan
            summary_records[m+"_ci_low"] = np.nan
            summary_records[m+"_ci_high"] = np.nan
            continue

        mean, lower, upper = bootstrap_ci(vals)

        summary_records[m+"_mean"] = mean
        summary_records[m+"_ci_low"] = lower
        summary_records[m+"_ci_high"] = upper


    summary_df = pd.DataFrame([summary_records])

    summary_path = os.path.join(EXP_DIR, "holdout_summary.csv")

    summary_df.to_csv(summary_path, index=False)


    print("\n===== HOLDOUT RESULTS (Bootstrap 95% CI) =====\n")

    for m in metrics:

        mean = summary_records[m+"_mean"]
        lo = summary_records[m+"_ci_low"]
        hi = summary_records[m+"_ci_high"]

        print(f"{m}: {mean:.4f}  (95% CI {lo:.4f} – {hi:.4f})")


    print("\nSaved:")
    print(patient_path)
    print(summary_path)

    pass

if __name__ == "__main__":
    import sys, json
    from configs.config import CONFIG

    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            new_cfg = json.load(f)
        CONFIG.clear()
        CONFIG.update(new_cfg)

    main()