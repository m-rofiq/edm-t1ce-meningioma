import os
import SimpleITK as sitk
from tqdm import tqdm

ROOT_DIR = r"./data/raw_dicom"
OUTPUT_DIR = r"./work/temp_resampled_fixed"

MODALITY_MAP = {
    "pre T1": "T1",
    "T2": "T2",
    "T2F": "FLAIR",
    "ce T1": "T1CE"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def dicom_to_volume(folder):
    reader = sitk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(folder)
    reader.SetFileNames(names)
    return reader.Execute()


def reorient(img):
    return sitk.DICOMOrient(img, 'RAS')


def resample_to_reference(moving, reference):

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(sitk.Transform())
    return resampler.Execute(moving)


patients = os.listdir(ROOT_DIR)

for patient in tqdm(patients):

    patient_path = os.path.join(ROOT_DIR, patient)
    if not os.path.isdir(patient_path):
        continue

    out_path = os.path.join(OUTPUT_DIR, patient)
    os.makedirs(out_path, exist_ok=True)

    volumes = {}

    # ---- load & reorient ----
    for raw_mod in os.listdir(patient_path):

        if raw_mod not in MODALITY_MAP:
            continue

        modality = MODALITY_MAP[raw_mod]
        mod_path = os.path.join(patient_path, raw_mod)

        img = dicom_to_volume(mod_path)
        img = reorient(img)

        volumes[modality] = img

    if "T1" not in volumes:
        continue

    # ---- T1 as reference ----
    reference = volumes["T1"]

    for modality, img in volumes.items():

        if modality == "T1":
            sitk.WriteImage(img, os.path.join(out_path, f"{modality}.nii.gz"))
        else:
            img_resampled = resample_to_reference(img, reference)
            sitk.WriteImage(img_resampled,
                            os.path.join(out_path, f"{modality}.nii.gz"))
