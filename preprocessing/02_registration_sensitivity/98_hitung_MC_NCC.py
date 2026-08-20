import os
import numpy as np
import pydicom
import torch
from tqdm import tqdm

# =============================
# CONFIG
# =============================
ROOT_DIR = r"./data/raw_dicom"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-8
BINS = 64

print("Using device:", DEVICE)

# =============================
# Helper: Find Modality Folder
# =============================
def find_modality_folder(base_path, keyword):
    for folder in os.listdir(base_path):
        if keyword.lower() in folder.lower():
            return os.path.join(base_path, folder)
    return None

# =============================
# Load DICOM Series (Sorted)
# =============================
def load_dicom_series(folder):
    files = []
    for f in os.listdir(folder):
        full_path = os.path.join(folder, f)
        if os.path.isfile(full_path):
            files.append(full_path)

    slices = []
    for f in files:
        try:
            dcm = pydicom.dcmread(f)
            instance_number = getattr(dcm, "InstanceNumber", 0)
            img = dcm.pixel_array.astype(np.float32)
            slices.append((instance_number, img))
        except:
            continue

    slices.sort(key=lambda x: x[0])
    volume = np.stack([s[1] for s in slices])
    return volume

# =============================
# Normalize
# =============================
def normalize_minmax(x):
    return (x - x.min()) / (x.max() - x.min() + EPS)

# =============================
# NCC
# =============================
def compute_ncc(x, y):
    x = x - torch.mean(x)
    y = y - torch.mean(y)
    return torch.sum(x * y) / (
        torch.sqrt(torch.sum(x**2)) * torch.sqrt(torch.sum(y**2)) + EPS
    )

# =============================
# FAST Mutual Information
# =============================
def compute_mi(x, y, bins=BINS):

    x = torch.clamp((x * (bins - 1)).long(), 0, bins - 1)
    y = torch.clamp((y * (bins - 1)).long(), 0, bins - 1)

    joint_hist = torch.zeros((bins, bins), device=DEVICE)
    joint_hist.index_put_((x.flatten(), y.flatten()), 
                          torch.ones_like(x.flatten(), dtype=torch.float32), 
                          accumulate=True)

    pxy = joint_hist / torch.sum(joint_hist)
    px = torch.sum(pxy, dim=1)
    py = torch.sum(pxy, dim=0)

    px_py = px.unsqueeze(1) * py.unsqueeze(0)

    nz = pxy > 0
    mi = torch.sum(pxy[nz] * torch.log(pxy[nz] / (px_py[nz] + EPS)))
    return mi

# =============================
# MAIN
# =============================

patients = [p for p in os.listdir(ROOT_DIR)
            if os.path.isdir(os.path.join(ROOT_DIR, p))]

print("Total patients found:", len(patients))

mi_before_all = []
ncc_before_all = []

for patient in tqdm(patients, desc="Processing Patients"):

    patient_path = os.path.join(ROOT_DIR, patient)

    t1_path = find_modality_folder(patient_path, "pre")
    t2_path = find_modality_folder(patient_path, "T2")

    if t1_path is None or t2_path is None:
        print("Skipping:", patient)
        continue

    vol_t1 = load_dicom_series(t1_path)
    vol_t2 = load_dicom_series(t2_path)

    mi_patient = []
    ncc_patient = []

    for i in range(min(len(vol_t1), len(vol_t2))):

        img1 = normalize_minmax(vol_t1[i])
        img2 = normalize_minmax(vol_t2[i])

        img1 = torch.tensor(img1, device=DEVICE)
        img2 = torch.tensor(img2, device=DEVICE)

        mi = compute_mi(img1, img2)
        ncc = compute_ncc(img1, img2)

        mi_patient.append(mi.item())
        ncc_patient.append(ncc.item())

    mi_before_all.append(np.mean(mi_patient))
    ncc_before_all.append(np.mean(ncc_patient))

# =============================
# RESULT
# =============================
print("\n==== BEFORE REGISTRATION ====")
print("MI Mean ± Std:", np.mean(mi_before_all), np.std(mi_before_all))
print("NCC Mean ± Std:", np.mean(ncc_before_all), np.std(ncc_before_all))
