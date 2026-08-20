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
    row = df.iloc[0]

    if base_exp not in records:
        records[base_exp] = []

    records[base_exp].append(row)


final_records = []

for exp in records:

    df = pd.DataFrame(records[exp])

    result = {"Experiment": exp}

    for col in df.columns:

        if col.endswith("_mean"):
            vals = df[col].values
            metric = col.replace("_mean","")

            result[f"{metric}_mean"] = np.mean(vals)
            #result[f"{metric}_std"] = np.std(vals)
            result[f"{metric}_std"] = np.std(vals, ddof=1)

    final_records.append(result)


result_df = pd.DataFrame(final_records)

save_path = os.path.join(EXP_ROOT, "fold_aggregated_results.csv")

result_df.to_csv(save_path, index=False)

print("\n===== FOLD AGGREGATED RESULTS =====\n")
print(result_df)

print("\nSaved:", save_path)