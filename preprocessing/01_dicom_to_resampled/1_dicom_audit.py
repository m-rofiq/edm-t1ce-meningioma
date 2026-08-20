import os
import pydicom
import pandas as pd
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================
ROOT_DIR = r"./data/raw_dicom"   # <-- sesuaikan
OUTPUT_CSV = "dicom_audit.csv"

MODALITY_MAP = {
    "pre T1": "T1",
    "T2": "T2",
    "T2F": "FLAIR",
    "ce T1": "T1CE"
}

# ==============================
# HELPER
# ==============================

def safe_get(ds, tag, default="NA"):
    return getattr(ds, tag, default)

def check_missing_slices(instance_numbers):
    instance_numbers = sorted(instance_numbers)
    expected = list(range(min(instance_numbers), max(instance_numbers)+1))
    return len(expected) != len(instance_numbers)

# ==============================
# MAIN AUDIT
# ==============================

rows = []

patients = sorted(os.listdir(ROOT_DIR))

for patient in tqdm(patients, desc="Auditing Patients"):

    patient_path = os.path.join(ROOT_DIR, patient)
    if not os.path.isdir(patient_path):
        continue

    for raw_mod in os.listdir(patient_path):

        if raw_mod not in MODALITY_MAP:
            continue

        modality = MODALITY_MAP[raw_mod]
        mod_path = os.path.join(patient_path, raw_mod)

        dicom_files = [
            os.path.join(mod_path, f)
            for f in os.listdir(mod_path)
            if f.lower().endswith(".dcm") or "." not in f
        ]

        if len(dicom_files) == 0:
            continue

        instance_numbers = []
        orientations = set()
        pixel_spacings = set()
        slice_thicknesses = set()

        for f in dicom_files:
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True)

                instance_numbers.append(int(safe_get(ds, "InstanceNumber", -1)))

                orientation = tuple(
                    round(float(x),5)
                    for x in safe_get(ds, "ImageOrientationPatient", [])
                )
                orientations.add(orientation)

                spacing = tuple(
                    round(float(x),5)
                    for x in safe_get(ds, "PixelSpacing", [])
                )
                pixel_spacings.add(spacing)

                thickness = round(float(safe_get(ds, "SliceThickness", -1)),5)
                slice_thicknesses.add(thickness)

            except Exception as e:
                print(f"Error reading {f}: {e}")

        missing = check_missing_slices(instance_numbers)

        rows.append({
            "patient": patient,
            "modality": modality,
            "orientation": str(sorted(list(orientations))),
            "pixel_spacing": str(sorted(list(pixel_spacings))),
            "slice_thickness": str(sorted(list(slice_thicknesses))),
            "n_slices": len(dicom_files),
            "missing_slice": missing
        })

# ==============================
# SAVE CSV
# ==============================

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print("\nSaved:", OUTPUT_CSV)

# ==============================
# GLOBAL CONSISTENCY CHECK
# ==============================

print("\n===== GLOBAL CONSISTENCY CHECK =====")

print("Unique orientations across dataset:",
      df["orientation"].nunique())

print("Unique pixel spacings:",
      df["pixel_spacing"].nunique())

print("Unique slice thickness:",
      df["slice_thickness"].nunique())

print("Any missing slices:",
      df["missing_slice"].any())
