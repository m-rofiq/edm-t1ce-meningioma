import os
import sys
import pandas as pd

# =========================
# ADD PROJECT ROOT
# =========================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from evaluators.fold_anova import fold_anova

CSV_PATH = r"./results/fold_results/patient_level_results.csv"

df = pd.read_csv(CSV_PATH)

fold_results = {}

for fold_id in df["fold"].unique():
    fold_results[int(fold_id)] = df[df["fold"] == fold_id]["psnr"].values

F_stat = fold_anova(fold_results)

print("F-statistic (real folds):", F_stat)