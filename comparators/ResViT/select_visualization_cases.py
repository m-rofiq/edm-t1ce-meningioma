import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from config_resvit_sota import CONFIG  # <-- satu-satunya perubahan vs versi DD-Res U-Net


def main():
    FOLD = CONFIG["fold"]
    EXP_DIR = os.path.join(
        CONFIG["experiment_root"],
        f"{CONFIG['experiment_id']}_{CONFIG['experiment_name']}_fold{FOLD}"
    )
    metrics_path = os.path.join(EXP_DIR, "holdout_patient_metrics.csv")

    df = pd.read_csv(metrics_path)
    print("\nLoaded metrics:", metrics_path)
    print("Patients:", len(df))

    best_case = df.sort_values("psnr_roi", ascending=False).iloc[0]

    median_psnr = df["psnr_roi"].median()
    df["median_dist"] = np.abs(df["psnr_roi"] - median_psnr)
    median_case = df.nsmallest(1, "median_dist").iloc[0]

    df["difficulty"] = (-df["psnr_roi"] + df["grad_enh"].fillna(0))
    hard_case = df.sort_values("difficulty", ascending=False).iloc[0]

    print("\n===== SELECTED VISUALIZATION CASES =====\n")
    print("BEST CASE"); print(best_case[["patient_id", "psnr_roi", "ssim_roi", "grad_enh"]])
    print("\nMEDIAN CASE"); print(median_case[["patient_id", "psnr_roi", "ssim_roi", "grad_enh"]])
    print("\nHARD CASE"); print(hard_case[["patient_id", "psnr_roi", "ssim_roi", "grad_enh"]])

    out_path = os.path.join(EXP_DIR, "selected_visualization_cases.csv")
    selected = pd.DataFrame([
        {"type": "best", "patient_id": best_case["patient_id"]},
        {"type": "median", "patient_id": median_case["patient_id"]},
        {"type": "hard", "patient_id": hard_case["patient_id"]},
    ])
    selected.to_csv(out_path, index=False)
    print("\nSaved:", out_path)


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            new_cfg = json.load(f)
        CONFIG.clear()
        CONFIG.update(new_cfg)
    main()
