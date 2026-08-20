import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("step5_per_case_metrics.csv")

modalities = df["modality"].unique()

outlier_records = []
summary_records = []

# ======================
# ANALYSIS PER MODALITY
# ======================
for mod in modalities:

    df_mod = df[df["modality"] == mod]
    delta = df_mod["delta_NCC"].values

    # ---------- HISTOGRAM ----------
    plt.figure()
    plt.hist(delta, bins=15)
    plt.title(f"Delta NCC Distribution - {mod}")
    plt.xlabel("Delta NCC")
    plt.ylabel("Frequency")
    plt.savefig(f"step6_hist_deltaNCC_{mod}.png")
    plt.close()

    # ---------- IQR OUTLIER ----------
    Q1 = np.percentile(delta, 25)
    Q3 = np.percentile(delta, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df_mod[(delta < lower_bound) | (delta > upper_bound)]

    # Simpan detail outlier
    for _, row in outliers.iterrows():
        outlier_records.append({
            "patient": row["patient"],
            "modality": mod,
            "delta_NCC": row["delta_NCC"]
        })

    summary_records.append({
        "modality": mod,
        "mean_delta": np.mean(delta),
        "std_delta": np.std(delta),
        "min_delta": np.min(delta),
        "max_delta": np.max(delta),
        "outlier_count": len(outliers),
        "total_cases": len(delta)
    })

# ======================
# SAVE RESULTS
# ======================
df_outliers = pd.DataFrame(outlier_records)
df_summary = pd.DataFrame(summary_records)

df_outliers.to_csv("step6_outliers_deltaNCC.csv", index=False)
df_summary.to_csv("step6_distribution_summary.csv", index=False)

print("Saved: step6_hist_deltaNCC_*.png")
print("Saved: step6_outliers_deltaNCC.csv")
print("Saved: step6_distribution_summary.csv")
print("\nSummary:")
print(df_summary)
