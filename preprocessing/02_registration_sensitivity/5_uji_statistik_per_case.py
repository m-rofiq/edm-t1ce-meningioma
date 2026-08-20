import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
from scipy.stats import shapiro, ttest_rel, wilcoxon

# ======================
# CONFIG
# ======================
before_root = r"./work/temp_resampled_fixed"
after_root  = r"./work/temp_registered"
modalities = ["FLAIR", "T1CE", "T2"]

per_case_records = []
summary_records = []

# ======================
# METRIC FUNCTIONS
# ======================
def compute_ncc(img1, img2):
    a = sitk.GetArrayFromImage(img1).astype(np.float32).ravel()
    b = sitk.GetArrayFromImage(img2).astype(np.float32).ravel()
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return np.mean(a * b)

def compute_mi(img1, img2, bins=64):
    a = sitk.GetArrayFromImage(img1).ravel()
    b = sitk.GetArrayFromImage(img2).ravel()
    hist_2d, _, _ = np.histogram2d(a, b, bins=bins)
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nz = pxy > 0
    return np.sum(pxy[nz] * np.log(pxy[nz] / px_py[nz]))

# ======================
# MAIN LOOP
# ======================
for mod in modalities:

    before_vals_ncc = []
    after_vals_ncc = []
    before_vals_mi = []
    after_vals_mi = []
    patient_list = []

    for patient in os.listdir(before_root):

        before_path = os.path.join(before_root, patient, f"{mod}.nii.gz")
        after_path  = os.path.join(after_root, patient, f"{mod}.nii.gz")

        if not os.path.exists(before_path) or not os.path.exists(after_path):
            continue

        fixed = sitk.ReadImage(before_path)
        moved = sitk.ReadImage(after_path)

        ncc_before = compute_ncc(fixed, fixed)
        ncc_after  = compute_ncc(fixed, moved)

        mi_before = compute_mi(fixed, fixed)
        mi_after  = compute_mi(fixed, moved)

        before_vals_ncc.append(ncc_before)
        after_vals_ncc.append(ncc_after)
        before_vals_mi.append(mi_before)
        after_vals_mi.append(mi_after)
        patient_list.append(patient)

        per_case_records.append({
            "patient": patient,
            "modality": mod,
            "NCC_before": ncc_before,
            "NCC_after": ncc_after,
            "delta_NCC": ncc_after - ncc_before,
            "MI_before": mi_before,
            "MI_after": mi_after,
            "delta_MI": mi_after - mi_before
        })

    before_vals_ncc = np.array(before_vals_ncc)
    after_vals_ncc  = np.array(after_vals_ncc)

    delta_ncc = after_vals_ncc - before_vals_ncc

    # Normality test
    p_normal = shapiro(delta_ncc).pvalue

    if p_normal > 0.05:
        stat, p_value = ttest_rel(after_vals_ncc, before_vals_ncc)
        test_used = "Paired t-test"
    else:
        stat, p_value = wilcoxon(after_vals_ncc, before_vals_ncc)
        test_used = "Wilcoxon"

    effect_size = np.mean(delta_ncc) / (np.std(delta_ncc) + 1e-8)
    percent_improved = np.sum(delta_ncc > 0) / len(delta_ncc) * 100

    summary_records.append({
        "modality": mod,
        "mean_delta_NCC": np.mean(delta_ncc),
        "std_delta_NCC": np.std(delta_ncc),
        "p_value": p_value,
        "effect_size": effect_size,
        "percent_improved": percent_improved,
        "test_used": test_used
    })

# ======================
# SAVE CSV
# ======================
df_per_case = pd.DataFrame(per_case_records)
df_summary = pd.DataFrame(summary_records)

df_per_case.to_csv("step5_per_case_metrics.csv", index=False)
df_summary.to_csv("step5_summary_stats.csv", index=False)

print("Saved: step5_per_case_metrics.csv")
print("Saved: step5_summary_stats.csv")
print("\nSummary:")
print(df_summary)
