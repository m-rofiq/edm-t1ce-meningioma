import os
import SimpleITK as sitk

INPUT_ROOT = r"./work/temp_resampled_fixed"
OUTPUT_ROOT = r"./work/temp_resampled_uniform"
TARGET_SPACING = (0.43, 0.43, 6.0)

modalities = ["T1","T1CE","T2","FLAIR"]

os.makedirs(OUTPUT_ROOT, exist_ok=True)

for patient in os.listdir(INPUT_ROOT):
    in_patient = os.path.join(INPUT_ROOT, patient)
    out_patient = os.path.join(OUTPUT_ROOT, patient)
    os.makedirs(out_patient, exist_ok=True)

    for mod in modalities:
        path = os.path.join(in_patient, f"{mod}.nii.gz")
        img = sitk.ReadImage(path)

        original_spacing = img.GetSpacing()
        original_size = img.GetSize()

        new_size = [
            int(round(original_size[i] * (original_spacing[i] / TARGET_SPACING[i])))
            for i in range(3)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputSpacing(TARGET_SPACING)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(img.GetDirection())
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetDefaultPixelValue(0)

        resampled = resampler.Execute(img)

        sitk.WriteImage(resampled,
                        os.path.join(out_patient, f"{mod}.nii.gz"))

print("Global resampling completed.")