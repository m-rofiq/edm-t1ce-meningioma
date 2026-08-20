import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# =========================
# ADD PROJECT ROOT
# =========================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from configs.config import CONFIG


EXP_ROOT = CONFIG["experiment_root"]

RAW_PATH = os.path.join(EXP_ROOT, "perfold_raw_results.csv")

# =========================
# EKSPERIMEN YANG DIBANDINGKAN
# ganti sesuai kebutuhan
# =========================

CHAMPION = "EXP-602_EDM" # "EXP-602_EDM" "EXP-703_EDM_GAN_VGG_LPIPS"

BASELINES = [
    # isi nama Experiment persis seperti di kolom "Experiment"
    # perfold_raw_results.csv, mis:
    "EXP-100_UNET_L1",
    "EXP-202_ATTENTION_UNET_L1",
    "EXP-204_DENSE_UNET_L1",
    "EXP-303_T1_T2_FLAIR_TO_T1CE",
    "EXP-402_MULTI_ENCODER_STEP1_T1_T2_F",
    "EXP-500_GUIDED_RESATTENTION_T1T2F",
    "EXP-601_EDM",
    #"EXP-602_EDM",
    "EXP-703_EDM_GAN_VGG_LPIPS",
    "EXP-713_MULTISCALE_CROSS_ATTENTION",
    "EXP-813_StratifiedEnh_Sharpness",
    "EXP-816_CalibratedGate_BalancedSeg"
]

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
    "lpips_enh",
    "sharpness_pred",
    "sharpness_gt",
    "sharpness_ratio",
    "sharpness_ratio_enh",
]


def stars(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def main():

    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"{RAW_PATH} tidak ditemukan. Jalankan build_perfold_raw_table.py dulu."
        )

    if not BASELINES:
        raise ValueError(
            "BASELINES masih kosong. Isi daftar Experiment yang mau dibandingkan "
            "dengan CHAMPION di bagian atas script."
        )

    df = pd.read_csv(RAW_PATH)

    champ_df = df[df["Experiment"] == CHAMPION].sort_values("Fold")

    if champ_df.empty:
        raise ValueError(f"CHAMPION '{CHAMPION}' tidak ada di {RAW_PATH}")

    records = []

    for baseline in BASELINES:

        base_df = df[df["Experiment"] == baseline].sort_values("Fold")

        if base_df.empty:
            print(f"[SKIP] '{baseline}' tidak ditemukan di perfold_raw_results.csv")
            continue

        # =========================
        # PASANGKAN BY FOLD (bukan by urutan baris)
        # -> jaga-jaga kalau ada fold yang hilang/tidak lengkap
        # =========================

        merged = pd.merge(
            champ_df, base_df,
            on="Fold",
            suffixes=("_champ", "_base")
        )

        for m in METRICS:

            col_champ = f"{m}_champ"
            col_base = f"{m}_base"

            x = merged[col_champ].values
            y = merged[col_base].values

            valid = ~(np.isnan(x) | np.isnan(y))
            x = x[valid]
            y = y[valid]

            n_pairs = len(x)

            if n_pairs < 3 or np.all(x == y):
                # wilcoxon butuh minimal beberapa selisih tak-nol
                stat, p = np.nan, np.nan
            else:
                try:
                    stat, p = wilcoxon(x, y)
                except ValueError:
                    stat, p = np.nan, np.nan

            records.append({
                "Champion": CHAMPION,
                "Baseline": baseline,
                "Metric": m,
                "n_pairs": n_pairs,
                "champion_mean": np.mean(x) if n_pairs else np.nan,
                "baseline_mean": np.mean(y) if n_pairs else np.nan,
                "wilcoxon_stat": stat,
                "p_value": p,
                "significance": stars(p),
            })

    result_df = pd.DataFrame(records)

    save_path = os.path.join(EXP_ROOT, "wilcoxon_significance.csv")
    result_df.to_csv(save_path, index=False)

    print("\n===== WILCOXON SIGNED-RANK TEST (per-fold paired, n=5) =====\n")
    print(result_df.to_string(index=False))

    print("\nSaved:", save_path)
    print("\nCatatan: n=5 (jumlah fold) kecil -> power uji rendah, 'ns' tidak")
    print("otomatis berarti 'tidak ada beda', bisa juga karena sampel kurang.")


if __name__ == "__main__":
    main()
