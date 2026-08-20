import os
import SimpleITK as sitk
import numpy as np

ROOT = r"./work/temp_uniform_final"
modalities = ["T1","T1CE","T2","FLAIR"]

zero_volume = 0

for patient in os.listdir(ROOT):
    for mod in modalities:
        path = os.path.join(ROOT, patient, f"{mod}.nii.gz")
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img)

        if np.std(arr) == 0:
            zero_volume += 1

print("Zero-variance volumes:", zero_volume)