import os
import numpy as np
import pandas as pd
import SimpleITK as sitk

INPUT_ROOT = r"./work/temp_uniform_normalized_v2"
CV_SPLIT = "5fold_cv_split.csv"
SACRED = "sacred_test_patients.csv"
OUTPUT_ROOT = r"./data/dataset_5fold_final_v2"

modalities = ["T1","T1CE","T2","FLAIR"]

os.makedirs(OUTPUT_ROOT, exist_ok=True)

cv_df = pd.read_csv(CV_SPLIT)
sacred_df = pd.read_csv(SACRED)

metadata_rows = []

# --------- BUILD CV FOLDS ----------
for fold in cv_df["fold"].unique():

    fold_dir = os.path.join(OUTPUT_ROOT, f"fold_{fold}")
    for split in ["train","test"]:
        for mod in modalities:
            os.makedirs(os.path.join(fold_dir, split, mod), exist_ok=True)

    fold_df = cv_df[cv_df["fold"] == fold]

    for _, row in fold_df.iterrows():
        patient = row["patient"]
        split = row["split"]

        patient_dir = os.path.join(INPUT_ROOT, patient)
        volumes = {}

        for mod in modalities:
            img = sitk.ReadImage(os.path.join(patient_dir, f"{mod}.nii.gz"))
            volumes[mod] = sitk.GetArrayFromImage(img)

        depth = volumes["T1"].shape[0]

        for i in range(1, depth-1):

            fname = f"{patient}_slice{i:03d}.npy"

            for mod in modalities:
                vol = volumes[mod]
                triplet = np.stack([
                    vol[i-1,:,:],
                    vol[i,:,:],
                    vol[i+1,:,:]
                ], axis=0).astype(np.float32)
                
                np.save(
                    os.path.join(fold_dir, split, mod, fname),
                    triplet
                )

            metadata_rows.append({
                "patient": patient,
                "fold": fold,
                "split": split,
                "slice_index": i,
                "filename": fname
            })

# --------- BUILD SACRED TEST ----------
sacred_dir = os.path.join(OUTPUT_ROOT, "sacred_test")
for mod in modalities:
    os.makedirs(os.path.join(sacred_dir, mod), exist_ok=True)

for patient in sacred_df["patient"]:
    patient_dir = os.path.join(INPUT_ROOT, patient)
    volumes = {}

    for mod in modalities:
        img = sitk.ReadImage(os.path.join(patient_dir, f"{mod}.nii.gz"))
        volumes[mod] = sitk.GetArrayFromImage(img)

    depth = volumes["T1"].shape[0]

    for i in range(1, depth-1):

        fname = f"{patient}_slice{i:03d}.npy"

        for mod in modalities:
            vol = volumes[mod]
            triplet = np.stack([
                vol[i-1,:,:],
                vol[i,:,:],
                vol[i+1,:,:]
            ], axis=0).astype(np.float32)
            
            np.save(
                os.path.join(sacred_dir, mod, fname),
                triplet
            )

        metadata_rows.append({
            "patient": patient,
            "fold": "sacred",
            "split": "test",
            "slice_index": i,
            "filename": fname
        })

# --------- SAVE METADATA ----------
meta_dir = os.path.join(OUTPUT_ROOT, "metadata")
os.makedirs(meta_dir, exist_ok=True)

pd.DataFrame(metadata_rows).to_csv(
    os.path.join(meta_dir, "slice_metadata.csv"),
    index=False
)

summary = {
    "total_patients": 64,
    "cv_patients": 53,
    "sacred_test_patients": 11,
    "strategy": "5-fold CV + sacred hold-out",
    "2.5D_strategy": "(z-1,z,z+1)",
    "normalization": "ROI Z-score",
    "background": "absolute zero"
}

pd.DataFrame([summary]).to_csv(
    os.path.join(meta_dir, "dataset_summary.csv"),
    index=False
)

print("Final 5-fold v2 dataset built.")