import os
import sys
import numpy as np
import pandas as pd

# =========================
# ADD PROJECT ROOT
# =========================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from configs.config import CONFIG


EXP_ROOT = CONFIG["experiment_root"]

# metrik yang mau dimasukkan ke tabel per-fold mentah
# (samakan dengan daftar "metrics" di evaluate_experiment_paper.py)
METRICS = [
    "psnr_roi",
    "ssim_roi",
    "mae",
    "calibration_bias",
    "variance_ratio",
    "psnr_enh",
    "ssim_enh",
    "grad_enh",
    "lpips",

    # [MERGED] metrik tambahan dari evaluate_experiment_paper.py (hasil
    # merge dengan evaluate_sharpness_paper.py) — ditambahkan di bawah,
    # urutan 9 metrik lama tidak diubah.
    "lpips_enh",
    "sharpness_pred",
    "sharpness_gt",
    "sharpness_ratio",
    "sharpness_ratio_enh",
]

N_FOLDS = 5


# =========================
# KUMPULKAN holdout_summary.csv PER FOLD
# (tanpa dirata-rata -> nilai mentah per fold dipertahankan)
# =========================

records = {}  # base_exp -> {fold_num: row}

for exp_dir in os.listdir(EXP_ROOT):

    if "_fold" not in exp_dir:
        continue

    base_exp, fold_str = exp_dir.rsplit("_fold", 1)

    if not fold_str.isdigit():
        continue

    fold_num = int(fold_str)

    summary_path = os.path.join(EXP_ROOT, exp_dir, "holdout_summary.csv")

    if not os.path.exists(summary_path):
        continue

    df = pd.read_csv(summary_path)
    row = df.iloc[0]

    records.setdefault(base_exp, {})[fold_num] = row


# =========================
# PIVOT: satu baris per (Experiment, Fold)
# kolom: psnr_roi, ssim_roi, mae, ... (metrik jadi kolom, bukan baris)
# =========================

rows = []

for exp, fold_dict in records.items():

    for fold_num in range(N_FOLDS):

        entry = {"Experiment": exp, "Fold": fold_num}

        row = fold_dict.get(fold_num)

        for m in METRICS:

            col = f"{m}_mean"
            entry[m] = row[col] if (row is not None and col in row) else np.nan

        rows.append(entry)

result_df = pd.DataFrame(rows)

save_path = os.path.join(EXP_ROOT, "perfold_raw_results.csv")
result_df.to_csv(save_path, index=False)

print("\n===== PER-FOLD RAW RESULTS =====\n")
print(result_df)

print("\nSaved:", save_path)
print("\nTips: untuk isi Table S2a/S2b/..., pivot kolom metrik yang dibutuhkan")
print("(mis. psnr_roi) melawan kolom Fold, dengan Experiment sebagai baris.")
