import os
import SimpleITK as sitk

ROOT = r"./work/temp_resampled_uniform"

sizes = set()

for patient in os.listdir(ROOT):
    path = os.path.join(ROOT, patient, "T1.nii.gz")
    img = sitk.ReadImage(path)
    sizes.add(img.GetSize())

print("Unique sizes found:")
for s in sizes:
    print(s)