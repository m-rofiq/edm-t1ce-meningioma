import pandas as pd
a = pd.read_csv("experiments/wilcoxon_significance_patientlevel.csv")
b = pd.read_csv("experiments/wilcoxon_significance_patientlevel_reference.csv")
print("bentuk", a.shape, b.shape)
print("kolom sama :", list(a.columns) == list(b.columns))
k = ["Comparison_Family","Champion","Baseline","Metric"]
m = a.merge(b, on=k, suffixes=("_baru","_ref"))
print("tercocokkan:", len(m))
for c in ["p_value","p_value_corrected","wilcoxon_stat","champion_mean","baseline_mean","n_pairs"]:
    print(f"{c:20s} beda maks = {(m[c+'_baru']-m[c+'_ref']).abs().max():.3e}")
print("tanda beda:", (m["significance_corrected_baru"] != m["significance_corrected_ref"]).sum())
