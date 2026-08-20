import os
import SimpleITK as sitk
from tqdm import tqdm

# ======================
# CONFIG
# ======================
ROOT_DIR = r"./data/raw_dicom"
OUTPUT_DIR = r"./work/temp_resampled"

TARGET_SPACING = (0.46875, 0.46875, 5.0)

MODALITY_MAP = {
    "pre T1": "T1",
    "T2": "T2",
    "T2F": "FLAIR",
    "ce T1": "T1CE"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# FUNCTION
# ======================

def dicom_to_volume(dicom_folder):

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
    reader.SetFileNames(dicom_names)

    image = reader.Execute()

    return image


def reorient_to_RAS(image):
    return sitk.DICOMOrient(image, 'RAS')


def resample_image(image, target_spacing):

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(original_size[i] * (original_spacing[i] / target_spacing[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())

    return resampler.Execute(image)

# ======================
# MAIN
# ======================

patients = os.listdir(ROOT_DIR)

for patient in tqdm(patients, desc="Processing Patients"):

    patient_path = os.path.join(ROOT_DIR, patient)
    if not os.path.isdir(patient_path):
        continue

    patient_out = os.path.join(OUTPUT_DIR, patient)
    os.makedirs(patient_out, exist_ok=True)

    for raw_mod in os.listdir(patient_path):

        if raw_mod not in MODALITY_MAP:
            continue

        modality = MODALITY_MAP[raw_mod]
        mod_path = os.path.join(patient_path, raw_mod)

        try:
            img = dicom_to_volume(mod_path)

            # Reorient
            img = reorient_to_RAS(img)

            # Resample
            img = resample_image(img, TARGET_SPACING)

            sitk.WriteImage(
                img,
                os.path.join(patient_out, f"{modality}.nii.gz")
            )

        except Exception as e:
            print(f"Error {patient} {modality}: {e}")
