import os
import pandas as pd

ROOT = r"./data/dataset_5fold_final_v2"
META_DIR = os.path.join(ROOT, "metadata")

mapping_df = pd.read_csv(os.path.join(META_DIR, "patient_mapping_private.csv"))
mapping = dict(zip(mapping_df["original_name"], mapping_df["anon_id"]))

# Update metadata
meta_path = os.path.join(META_DIR, "slice_metadata.csv")
meta_df = pd.read_csv(meta_path)

meta_df["anon_patient"] = meta_df["patient"].map(mapping)
meta_df["filename"] = meta_df.apply(
    lambda row: f"{mapping[row['patient']]}_slice{int(row['slice_index']):03d}.npy",
    axis=1
)

meta_df.to_csv(os.path.join(META_DIR, "slice_metadata.csv"), index=False)

# Rename actual files
for root, dirs, files in os.walk(ROOT):
    if "metadata" in root:
        continue
    for file in files:
        if not file.endswith(".npy"):
            continue

        original_patient = file.split("_slice")[0]
        if original_patient not in mapping:
            continue

        slice_part = file.split("_slice")[1]
        new_name = f"{mapping[original_patient]}_slice{slice_part}"

        old_path = os.path.join(root, file)
        new_path = os.path.join(root, new_name)

        os.rename(old_path, new_path)

print("Anonymization applied to dataset.")