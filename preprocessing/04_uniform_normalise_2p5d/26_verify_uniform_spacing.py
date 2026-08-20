import os
import SimpleITK as sitk

ROOT = r"./work/temp_resampled_uniform"

spacings = set()

for patient in os.listdir(ROOT):
    path = os.path.join(ROOT, patient, "T1.nii.gz")
    img = sitk.ReadImage(path)
    spacings.add(tuple(round(s,3) for s in img.GetSpacing()))

print("Unique spacings found:")
for s in spacings:
    print(s)