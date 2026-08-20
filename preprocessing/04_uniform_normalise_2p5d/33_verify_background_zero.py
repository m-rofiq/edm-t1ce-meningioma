import os
import SimpleITK as sitk
import numpy as np

ROOT = r"./work/temp_uniform_normalized"

corner_values = []

for patient in os.listdir(ROOT):
    path = os.path.join(ROOT, patient, "T1.nii.gz")
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)

    corners = [
        arr[0,0,0],
        arr[0,0,-1],
        arr[0,-1,0],
        arr[-1,0,0]
    ]
    corner_values.extend(corners)

print("Unique corner values:", set(corner_values))