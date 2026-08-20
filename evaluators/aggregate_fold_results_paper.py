import os
import sys
import pandas as pd
import numpy as np

# =========================
# ADD PROJECT ROOT
# =========================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from configs.config import CONFIG


EXP_ROOT = CONFIG["experiment_root"]

# kolom dari holdout_model_profile.csv yang ikut diagregasi lintas fold
PROFILE_COLS = [
    "total_params",
    "trainable_params",
    "gflops",
    "avg_inference_time_ms_per_slice",
    "throughput_slices_per_sec",
]

records = {}

for exp_dir in os.listdir(EXP_ROOT):

    if "_fold" not in exp_dir:
        continue

    base_exp = exp_dir.split("_fold")[0]

    summary_path = os.path.join(
        EXP_ROOT,
        exp_dir,
        "holdout_summary.csv"
    )

    if not os.path.exists(summary_path):
        continue

    df = pd.read_csv(summary_path)
    row = df.iloc[0].to_dict()

    # =========================
    # GABUNGKAN DATA PROFILING (params/FLOPs/inference time)
    # =========================

    profile_path = os.path.join(
        EXP_ROOT,
        exp_dir,
        "holdout_model_profile.csv"
    )

    if os.path.exists(profile_path):

        profile_df = pd.read_csv(profile_path)
        profile_row = profile_df.iloc[0]

        for c in PROFILE_COLS:
            if c in profile_row:
                row[c] = profile_row[c]

    else:
        # fold ini belum punya holdout_model_profile.csv (mis. dijalankan
        # sebelum profiling ditambahkan ke script evaluasi) -> isi NaN
        for c in PROFILE_COLS:
            row[c] = np.nan

    if base_exp not in records:
        records[base_exp] = []

    records[base_exp].append(row)


final_records = []

for exp in records:

    df = pd.DataFrame(records[exp])

    result = {"Experiment": exp, "n_folds": len(df)}

    # ---- metrik evaluasi (kolom *_mean dari holdout_summary.csv) ----
    for col in df.columns:

        if col.endswith("_mean"):
            vals = df[col].values
            metric = col.replace("_mean", "")

            result[f"{metric}_mean"] = np.mean(vals)
            #result[f"{metric}_std"] = np.std(vals)
            result[f"{metric}_std"] = np.std(vals, ddof=1)

    # ---- params/FLOPs/inference time (dari holdout_model_profile.csv) ----
    # total_params/trainable_params/gflops semestinya konstan lintas fold
    # (arsitektur sama), std-nya dilaporkan untuk verifikasi (idealnya ~0).
    # avg_inference_time_ms_per_slice & throughput bisa sedikit bervariasi
    # antar fold tergantung kondisi runtime saat profiling dijalankan.
    for col in PROFILE_COLS:

        if col not in df.columns:
            continue

        vals = df[col].dropna().values

        if len(vals) == 0:
            result[f"{col}_mean"] = np.nan
            result[f"{col}_std"] = np.nan
            continue

        result[f"{col}_mean"] = np.mean(vals)
        # n=1 (eksperimen supplementary fold0-only) -> std tidak terdefinisi,
        # NaN supaya konsisten dengan perilaku np.std(ddof=1) di kolom metrik
        result[f"{col}_std"] = np.std(vals, ddof=1) if len(vals) > 1 else np.nan

    final_records.append(result)


result_df = pd.DataFrame(final_records)

save_path = os.path.join(EXP_ROOT, "fold_aggregated_results.csv")

result_df.to_csv(save_path, index=False)

print("\n===== FOLD AGGREGATED RESULTS =====\n")
print(result_df)

print("\nSaved:", save_path)
