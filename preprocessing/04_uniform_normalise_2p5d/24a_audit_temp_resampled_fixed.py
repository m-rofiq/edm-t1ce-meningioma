import os
import SimpleITK as sitk
import numpy as np

ROOT = r"./work/temp_resampled_fixed"
modalities = ["T1", "T1CE", "T2", "FLAIR"]

patients = os.listdir(ROOT)

print("Total patients:", len(patients))

for patient in patients[:5]:  # cek 5 sample dulu
    patient_path = os.path.join(ROOT, patient)
    print("\nPatient:", patient)

    for mod in modalities:
        path = os.path.join(patient_path, f"{mod}.nii.gz")

        if not os.path.exists(path):
            print("Missing:", mod)
            continue

        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img)

        print(
            mod,
            "| shape:", arr.shape,
            "| spacing:", img.GetSpacing(),
            "| min:", np.min(arr),
            "| max:", np.max(arr)
        )