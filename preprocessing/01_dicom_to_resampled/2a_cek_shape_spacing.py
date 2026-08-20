import os
import SimpleITK as sitk

ROOT = r"./work/temp_resampled_fixed"

for patient in os.listdir(ROOT):

    patient_path = os.path.join(ROOT, patient)
    if not os.path.isdir(patient_path):
        continue

    sizes = []
    spacings = []
    directions = []

    for mod in ["T1.nii.gz","T2.nii.gz","FLAIR.nii.gz","T1CE.nii.gz"]:

        img = sitk.ReadImage(os.path.join(patient_path, mod))
        sizes.append(img.GetSize())
        spacings.append(img.GetSpacing())
        directions.append(img.GetDirection())

    print(patient)
    print("  size unique:", len(set(sizes)))
    print("  spacing unique:", len(set(spacings)))
    print("  direction unique:", len(set(directions)))
