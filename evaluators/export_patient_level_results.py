import os
import sys
import numpy as np
import pandas as pd
import torch

# =========================
# ADD PROJECT ROOT
# =========================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from evaluators.psnr_fixed_range import psnr_fixed
from evaluators.ssim_masked import ssim_roi

DATASET_ROOT = r"./data/dataset_5fold_final_v4"
RESULT_SAVE_PATH = r"./results/fold_results"

os.makedirs(RESULT_SAVE_PATH, exist_ok=True)

records = []

for fold_name in os.listdir(DATASET_ROOT):

    if not fold_name.startswith("fold_"):
        continue

    fold_id = int(fold_name.split("_")[1])
    val_root = os.path.join(DATASET_ROOT, fold_name, "val")

    t1ce_dir = os.path.join(val_root, "T1CE")
    #pred_dir = os.path.join(val_root, "T1CE")  # sementara GT=Pred (dummy safe test)

    patient_groups = {}

    for fname in os.listdir(t1ce_dir):

        patient_id = fname.split("_slice")[0]

        if patient_id not in patient_groups:
            patient_groups[patient_id] = []
        
        gt = np.load(os.path.join(t1ce_dir, fname))
        pred = gt.copy()

        center = gt[1]

        gt_t = torch.tensor(center).unsqueeze(0).unsqueeze(0).float()
        pred_t = torch.tensor(center).unsqueeze(0).unsqueeze(0).float()


        psnr_val = psnr_fixed(pred_t, gt_t).item()

        patient_groups[patient_id].append(psnr_val)

    for pid in patient_groups:
        patient_mean = np.mean(patient_groups[pid])
        records.append({
            "fold": fold_id,
            "patient_id": pid,
            "psnr": patient_mean
        })

df = pd.DataFrame(records)
df.to_csv(os.path.join(RESULT_SAVE_PATH, "patient_level_results.csv"), index=False)

print("Patient-level results saved.")