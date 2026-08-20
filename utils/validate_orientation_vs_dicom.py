import os
import numpy as np
import pydicom


# =========================
# PATH
# =========================

DICOM_SERIES = r"./data/raw_dicom/<subject>/<series>"   # set to a local DICOM series, not distributed

NPY_ROOT = r"./data/dataset_5fold_final_v2"

FOLD = 0
MODALITY = "T1"


# =========================
# LOAD DICOM VOLUME
# =========================

def load_dicom_volume(folder):

    slices = []

    for f in os.listdir(folder):

        if f.lower().endswith(".dcm"):

            dcm = pydicom.dcmread(os.path.join(folder, f))

            z = float(dcm.ImagePositionPatient[2])

            slices.append((z, dcm.pixel_array))

    slices.sort(key=lambda x: x[0])

    volume = np.stack([s[1] for s in slices])

    return volume.astype(np.float32)


# =========================
# FIND CENTER SLICE
# =========================

def find_center_slice(volume):

    scores = []

    for i in range(volume.shape[0]):

        slice_img = volume[i]

        brain_pixels = np.sum(slice_img > 0)

        scores.append(brain_pixels)

    return volume[np.argmax(scores)]


# =========================
# NORMALIZATION
# =========================

def normalize(img):

    img = img.astype(np.float32)

    img -= img.mean()

    img /= (img.std() + 1e-8)

    return img


# =========================
# NCC
# =========================

def ncc(a, b):

    a = normalize(a)

    b = normalize(b)

    return np.mean(a * b)


# =========================
# ORIENTATION TEST
# =========================

def test_orientations(dicom_slice, npy_slice):

    return {

        "original": ncc(dicom_slice, npy_slice),

        "flip_x": ncc(dicom_slice, np.fliplr(npy_slice)),

        "flip_y": ncc(dicom_slice, np.flipud(npy_slice)),

        "flip_xy": ncc(dicom_slice, np.flip(np.flipud(npy_slice)))
    }


# =========================
# LOAD DICOM
# =========================

dicom_vol = load_dicom_volume(DICOM_SERIES)

dicom_slice = find_center_slice(dicom_vol)


# =========================
# LOAD DATASET SAMPLE
# =========================

train_dir = os.path.join(NPY_ROOT, f"fold_{FOLD}", "train", MODALITY)

files = sorted([f for f in os.listdir(train_dir) if f.endswith(".npy")])


scores_list = []

for f in files[:50]:

    npy = np.load(os.path.join(train_dir, f))

    npy_slice = npy[1]

    scores = test_orientations(dicom_slice, npy_slice)

    scores_list.append(scores)


# =========================
# AGGREGATE
# =========================

summary = {}

for key in scores_list[0].keys():

    summary[key] = np.mean([s[key] for s in scores_list])


print("\n=== Orientation Validation ===\n")

for k,v in summary.items():

    print(f"{k}: {v:.4f}")


best = max(summary, key=summary.get)

print("\nBEST ORIENTATION:", best)
