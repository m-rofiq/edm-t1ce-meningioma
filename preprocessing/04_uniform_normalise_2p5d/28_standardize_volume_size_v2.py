import os
import SimpleITK as sitk
import numpy as np

INPUT_ROOT = r"./work/temp_resampled_uniform"
OUTPUT_ROOT = r"./work/temp_uniform_final_v2"
TARGET_SIZE = (24, 512, 512) # (z,y,x) BENAR

modalities = ["T1","T1CE","T2","FLAIR"]

os.makedirs(OUTPUT_ROOT, exist_ok=True)

def center_crop_or_pad(arr, target_shape):
    z, y, x = arr.shape
    tz, ty, tx = target_shape

    # Pad if needed
    pad_z = max(tz - z, 0)
    pad_y = max(ty - y, 0)
    pad_x = max(tx - x, 0)

    arr = np.pad(arr,
                 ((pad_z//2, pad_z - pad_z//2),
                  (pad_y//2, pad_y - pad_y//2),
                  (pad_x//2, pad_x - pad_x//2)),
                 mode='constant')

    # Crop if needed
    z, y, x = arr.shape
    start_z = (z - tz)//2
    start_y = (y - ty)//2
    start_x = (x - tx)//2

    return arr[start_z:start_z+tz,
               start_y:start_y+ty,
               start_x:start_x+tx]

for patient in os.listdir(INPUT_ROOT):
    in_patient = os.path.join(INPUT_ROOT, patient)
    out_patient = os.path.join(OUTPUT_ROOT, patient)
    os.makedirs(out_patient, exist_ok=True)

    for mod in modalities:
        path = os.path.join(in_patient, f"{mod}.nii.gz")
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img)  # (z,y,x)

        arr_fixed = center_crop_or_pad(arr, TARGET_SIZE)

        new_img = sitk.GetImageFromArray(arr_fixed)
        new_img.SetSpacing(img.GetSpacing())
        new_img.SetOrigin(img.GetOrigin())
        new_img.SetDirection(img.GetDirection())

        sitk.WriteImage(new_img,
                        os.path.join(out_patient, f"{mod}.nii.gz"))

print("Volume size standardized")