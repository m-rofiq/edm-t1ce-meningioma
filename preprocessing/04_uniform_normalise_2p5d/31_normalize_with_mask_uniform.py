import os
import SimpleITK as sitk
import numpy as np

INPUT_ROOT = r"./work/temp_uniform_final_v2"
OUTPUT_ROOT = r"./work/temp_uniform_normalized_v2"

modalities = ["T1","T1CE","T2","FLAIR"]

os.makedirs(OUTPUT_ROOT, exist_ok=True)

for patient in os.listdir(INPUT_ROOT):
    in_patient = os.path.join(INPUT_ROOT, patient)
    out_patient = os.path.join(OUTPUT_ROOT, patient)
    os.makedirs(out_patient, exist_ok=True)

    for mod in modalities:
        path = os.path.join(in_patient, f"{mod}.nii.gz")
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img).astype(np.float32)

        mask = arr > 0

        if np.sum(mask) == 0:
            continue

        brain = arr[mask]

        # Percentile clipping
        p_low = np.percentile(brain, 0.5)
        p_high = np.percentile(brain, 99.5)
        arr = np.clip(arr, p_low, p_high)

        brain = arr[mask]

        mean = brain.mean()
        std = brain.std()

        arr_norm = np.zeros_like(arr)
        arr_norm[mask] = (arr[mask] - mean) / (std + 1e-8)

        new_img = sitk.GetImageFromArray(arr_norm)
        new_img.SetSpacing(img.GetSpacing())
        new_img.SetOrigin(img.GetOrigin())
        new_img.SetDirection(img.GetDirection())

        sitk.WriteImage(new_img,
                        os.path.join(out_patient, f"{mod}.nii.gz"))

print("Normalization completed.")