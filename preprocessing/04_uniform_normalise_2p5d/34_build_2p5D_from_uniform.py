import os
import numpy as np
import SimpleITK as sitk

INPUT_ROOT = r"./work/temp_uniform_normalized"
OUTPUT_ROOT = r"./data/dataset_5fold_base"
modalities = ["T1","T1CE","T2","FLAIR"]

os.makedirs(OUTPUT_ROOT, exist_ok=True)

def center_slice_triplet(volume, idx):
    return volume[idx-1:idx+2]

for patient in os.listdir(INPUT_ROOT):
    patient_dir = os.path.join(INPUT_ROOT, patient)

    volumes = {}
    for mod in modalities:
        img = sitk.ReadImage(os.path.join(patient_dir, f"{mod}.nii.gz"))
        volumes[mod] = sitk.GetArrayFromImage(img)

    depth = volumes["T1"].shape[0]

    out_patient = os.path.join(OUTPUT_ROOT, patient)
    os.makedirs(out_patient, exist_ok=True)

    for i in range(1, depth-1):
        for mod in modalities:
            triplet = center_slice_triplet(volumes[mod], i)
            np.save(
                os.path.join(out_patient, f"{mod}_slice{i:03d}.npy"),
                triplet.astype(np.float32)
            )

print("2.5D base dataset built.")