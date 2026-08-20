import os
import hashlib
import pandas as pd

ROOT = r"./data/dataset_5fold_final_v2"

# Ambil semua pasien unik dari metadata
meta_path = os.path.join(ROOT, "metadata", "slice_metadata.csv")
df = pd.read_csv(meta_path)

patients = sorted(df["patient"].unique())

def anon_id(name):
    return hashlib.sha1(name.encode()).hexdigest()[:8]

mapping = []

for p in patients:
    mapping.append({
        "original_name": p,
        "anon_id": anon_id(p)
    })

mapping_df = pd.DataFrame(mapping)
mapping_df.to_csv(
    os.path.join(ROOT, "metadata", "patient_mapping_private.csv"),
    index=False
)

print("Mapping generated.")
print(mapping_df.head())