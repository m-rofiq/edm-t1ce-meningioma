import os
import numpy as np

ROOT = r"./data/dataset_5fold_base"
modalities = ["T1","T1CE","T2","FLAIR"]

shape_set = set()
nan_count = 0
inf_count = 0
flat_count = 0
channel_identical = 0
total = 0

for patient in os.listdir(ROOT):
    patient_dir = os.path.join(ROOT, patient)

    for file in os.listdir(patient_dir):
        if not file.endswith(".npy"):
            continue

        arr = np.load(os.path.join(patient_dir, file))

        total += 1
        shape_set.add(arr.shape)

        if np.isnan(arr).any():
            nan_count += 1
        if np.isinf(arr).any():
            inf_count += 1

        # middle slice std
        if np.std(arr[1][arr[1] != 0]) < 0.01:
            flat_count += 1

        # check redundant channels
        if np.allclose(arr[0], arr[1]) and np.allclose(arr[1], arr[2]):
            channel_identical += 1

print("Unique shapes:", shape_set)
print("NaN:", nan_count)
print("Inf:", inf_count)
print("Flat slices:", flat_count)
print("Redundant triplets:", channel_identical)
print("Total samples:", total)