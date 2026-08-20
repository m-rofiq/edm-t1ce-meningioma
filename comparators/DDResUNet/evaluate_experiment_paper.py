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
from config_ddresunet_sota import CONFIG

from psnr_fixed_range import (
    psnr_roi,
    calibration_bias,
    variance_ratio,
    gradient_difference_tumor
)

from ssim_masked import ssim_roi
from build_enhancement_mask import build_enhancement_mask

import time
from torchinfo import summary as torchinfo_summary
from fvcore.nn import FlopCountAnalysis, flop_count_table


# ==================================================
# SHARPNESS METRICS
# --------------------------------------------------
# [MERGED] Dipindahkan dari evaluate_sharpness_paper.py — evaluasi
# sharpness sekarang dihitung menyatu dengan evaluasi experiment ini,
# tidak lagi lewat script terpisah.
#
# sharpness_tenengrad:
#   Rata-rata gradient magnitude (Sobel) di dalam
#   brain mask. Nilai lebih tinggi = lebih tajam.
#   Referensi: Krotkov & Martin, IJCV 1993.
#
# sharpness_ratio:
#   sharpness(pred) / sharpness(gt)
#   Nilai ideal = 1.0. < 1.0 berarti pred lebih
#   blur dari GT.
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


def profile_model(model, sample_x, device, n_warmup=10, n_runs=50):
    """
    Ukur jumlah parameter, FLOPs, dan inference time model pada satu sample input
    (1 slice).

    Konvensi mengikuti rekomendasi evaluasi paper (point 2):
      - warm-up 5-10 iterasi dibuang untuk menghindari cold-start bias
        (CUDA context init, cuDNN autotuning, memory allocator warm-up, dsb.)
      - waktu inferensi dirata-ratakan dari >= 50 forward pass setelah warm-up
      - dilaporkan sebagai ms/slice (karena sample_x berisi satu slice 2D)

    n_warmup: jumlah iterasi warm-up yang dibuang, disarankan 5-10.
    n_runs:   jumlah forward pass yang diukur & dirata-ratakan, minimal 50.
    """

    if n_warmup < 5 or n_warmup > 10:
        print(f"[profile_model] WARNING: n_warmup={n_warmup} di luar rentang "
              f"rekomendasi 5-10 iterasi.")
    if n_runs < 50:
        print(f"[profile_model] WARNING: n_runs={n_runs} < 50, "
              f"hasil rata-rata inference time mungkin kurang stabil.")

    # ---- Params (torchinfo) ----
    info = torchinfo_summary(model, input_data=sample_x, verbose=0)
    total_params = info.total_params
    trainable_params = info.trainable_params

    # ---- FLOPs (fvcore) ----
    model.eval()
    with torch.no_grad():
        flop_analyzer = FlopCountAnalysis(model, sample_x)
        flop_analyzer.unsupported_ops_warnings(False)
        flop_analyzer.uncalled_modules_warnings(False)
        total_flops = flop_analyzer.total()

    print("\n" + flop_count_table(flop_analyzer))

    # ---- Inference time (warm-up dibuang, lalu dirata-ratakan >= n_runs) ----
    with torch.no_grad():

        # Sinkronisasi sebelum warm-up supaya operasi CUDA dari load
        # model/FLOP-counting sebelumnya tidak ikut tercampur ke pengukuran.
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

    avg_time_ms_per_slice = (end - start) / n_runs * 1000.0
    throughput_slices_per_sec = 1000.0 / avg_time_ms_per_slice

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "flops": total_flops,
        "gflops": total_flops / 1e9,
        "n_warmup": n_warmup,
        "n_runs": n_runs,
        "avg_inference_time_ms_per_slice": avg_time_ms_per_slice,
        "throughput_slices_per_sec": throughput_slices_per_sec,
        "device": str(device),
    }


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

    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()

    # =========================
    # MODEL COMPLEXITY PROFILING
    # (params, FLOPs, inference time)
    # =========================

    sample_file = files[0]
    sample_inputs = [
        np.load(os.path.join(mod_dirs[m], sample_file)) for m in modalities
    ]
    sample_x = torch.from_numpy(
        np.concatenate(sample_inputs, axis=0)
    ).unsqueeze(0).float().to(device)

    # n_warmup/n_runs dapat dioverride lewat CONFIG (mis. untuk CPU-only run
    # yang ingin n_runs lebih kecil); default mengikuti rekomendasi
    # warm-up 5-10 iterasi & >= 50 forward pass terukur.
    profile_records = profile_model(
        model,
        sample_x,
        device,
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
        #mae = torch.mean(torch.abs(pred - gt)[mask == 1]).item()
        mae = torch.mean(torch.abs(pred - gt)[mask.bool()]).item()

        with torch.no_grad():
            pred_lp = torch.clamp(pred, 0.0, 1.0)
            gt_lp = torch.clamp(gt, 0.0, 1.0)

            pred_lp = (pred_lp * 2.0) - 1.0
            gt_lp = (gt_lp * 2.0) - 1.0

            if pred_lp.shape[1] == 1:
                pred_lp = pred_lp.repeat(1, 3, 1, 1)
                gt_lp = gt_lp.repeat(1, 3, 1, 1)

            # detach() opsional jika di dalam no_grad, tapi lebih aman tetap ada
            lp = lpips_model(pred_lp, gt_lp).mean().item()

        # =========================
        # ADVANCED METRICS
        # =========================

        bias = calibration_bias(pred, gt, mask).item()
        var_ratio = variance_ratio(pred, gt, mask).item()

        enh_mask = build_enhancement_mask(gt, mask)

        # [MERGED] Sharpness (global, di dalam brain mask) — tidak
        # bergantung pada enh_mask, jadi dihitung di luar percabangan
        # psnr_enh/ssim_enh/grad_enh seperti aslinya di
        # evaluate_sharpness_paper.py.
        sharp_pred  = sharpness_tenengrad(pred, mask).item()
        sharp_gt    = sharpness_tenengrad(gt,   mask).item()
        sharp_ratio = sharpness_ratio(pred, gt, mask).item()

        if torch.sum(enh_mask) > 10:

            psnr_enh = psnr_roi(pred, gt, enh_mask).item()
            ssim_enh = ssim_roi(pred, gt, enh_mask).item()
            grad_enh = gradient_difference_tumor(pred, gt, enh_mask).item()

            # [MERGED] LPIPS di enhancement region secara spesifik
            with torch.no_grad():
                enh_mask_f = enh_mask.float()

                pred_enh_lp = torch.clamp(pred * enh_mask_f, 0.0, 1.0)
                gt_enh_lp   = torch.clamp(gt   * enh_mask_f, 0.0, 1.0)

                pred_enh_lp = (pred_enh_lp * 2.0) - 1.0
                gt_enh_lp   = (gt_enh_lp   * 2.0) - 1.0

                if pred_enh_lp.shape[1] == 1:
                    pred_enh_lp = pred_enh_lp.repeat(1, 3, 1, 1)
                    gt_enh_lp   = gt_enh_lp.repeat(1,   3, 1, 1)

                lpips_enh = lpips_model(pred_enh_lp, gt_enh_lp).mean().item()

            # [MERGED] Sharpness ratio di enhancement region
            sharp_ratio_enh = sharpness_ratio(pred, gt, enh_mask).item()

        else:

            psnr_enh = np.nan
            ssim_enh = np.nan
            grad_enh = np.nan
            lpips_enh = np.nan
            sharp_ratio_enh = np.nan


        records.append({

            "patient_id": patient_id,

            "psnr_roi": psnr,
            "ssim_roi": ssim,
            "mae": mae,
            "lpips": lp,

            "calibration_bias": bias,
            "variance_ratio": var_ratio,

            "psnr_enh": psnr_enh,
            "ssim_enh": ssim_enh,
            "grad_enh": grad_enh,

            # [MERGED] Metrik tambahan dari evaluate_sharpness_paper.py,
            # diletakkan di bawah metrik yang sudah ada, urutan lama
            # tidak diubah.
            "lpips_enh": lpips_enh,
            "sharpness_pred": sharp_pred,
            "sharpness_gt": sharp_gt,
            "sharpness_ratio": sharp_ratio,
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
        "psnr_roi",
        "ssim_roi",
        "mae",
        "calibration_bias",
        "variance_ratio",
        "psnr_enh",
        "ssim_enh",
        "grad_enh",
        "lpips",

        # [MERGED] Metrik tambahan dari evaluate_sharpness_paper.py
        "lpips_enh",
        "sharpness_pred",
        "sharpness_gt",
        "sharpness_ratio",
        "sharpness_ratio_enh",
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
    from config_ddresunet_sota import CONFIG

    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            new_cfg = json.load(f)
        CONFIG.clear()
        CONFIG.update(new_cfg)

    main()
