import os
import SimpleITK as sitk
import numpy as np

ROOT = r"./work/temp_uniform_normalized"
modalities = ["T1","T1CE","T2","FLAIR"]

for mod in modalities:
    means = []
    stds = []

    for patient in os.listdir(ROOT):
        path = os.path.join(ROOT, patient, f"{mod}.nii.gz")
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img)

        mask = arr != 0
        brain = arr[mask]

        means.append(brain.mean())
        stds.append(brain.std())

    print(f"\n{mod}")
    print("Mean (avg over patients):", np.mean(means))
    print("Std (avg over patients):", np.mean(stds))