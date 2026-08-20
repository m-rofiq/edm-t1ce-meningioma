import os
import numpy as np
from tqdm import tqdm

DATASET_ROOT = r"./data/dataset_5fold_final_v2"

MODALITIES = ["T1", "T1CE"]

EXPECTED_SHAPE = (3,512,512)

corr_values = []
shape_errors = []
pairing_errors = []
empty_slices = []

print("\n=== DATASET INTEGRITY AUDIT ===\n")

folders = []

for f in os.listdir(DATASET_ROOT):

    path = os.path.join(DATASET_ROOT,f)

    if os.path.isdir(path):

        folders.append(path)

total_files = 0

for folder in folders:

    print("Scanning:", folder)

    t1_dir = os.path.join(folder,"T1")
    t1ce_dir = os.path.join(folder,"T1CE")

    if not os.path.exists(t1_dir) or not os.path.exists(t1ce_dir):

        continue

    t1_files = sorted([f for f in os.listdir(t1_dir) if f.endswith(".npy")])
    t1ce_files = sorted([f for f in os.listdir(t1ce_dir) if f.endswith(".npy")])

    if len(t1_files) != len(t1ce_files):

        pairing_errors.append(folder)

    for f in tqdm(t1_files):

        total_files += 1

        t1_path = os.path.join(t1_dir,f)
        t1ce_path = os.path.join(t1ce_dir,f)

        if not os.path.exists(t1ce_path):

            pairing_errors.append(f)
            continue

        try:

            t1 = np.load(t1_path)
            t1ce = np.load(t1ce_path)

        except:

            pairing_errors.append(f)
            continue

        if t1.shape != EXPECTED_SHAPE:

            shape_errors.append((f,t1.shape))

        if t1ce.shape != EXPECTED_SHAPE:

            shape_errors.append((f,t1ce.shape))

        center_t1 = t1[1]
        center_t1ce = t1ce[1]

        mask = center_t1ce != 0

        if np.sum(mask) < 100:

            empty_slices.append(f)
            continue

        corr = np.corrcoef(
            center_t1[mask].flatten(),
            center_t1ce[mask].flatten()
        )[0,1]

        corr_values.append(corr)

print("\n=== AUDIT SUMMARY ===\n")

print("Total files scanned:", total_files)

if len(corr_values) > 0:

    print("\nCorrelation statistics")

    print("Mean:", np.mean(corr_values))
    print("Min :", np.min(corr_values))
    print("Max :", np.max(corr_values))

print("\nShape errors:", len(shape_errors))

print("Pairing errors:", len(pairing_errors))

print("Empty slices:", len(empty_slices))

if len(shape_errors) > 0:

    print("\nExample shape error:", shape_errors[0])

if len(pairing_errors) > 0:

    print("\nExample pairing error:", pairing_errors[0])

print("\n=== AUDIT COMPLETE ===")