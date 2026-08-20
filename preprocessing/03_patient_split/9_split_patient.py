import os
import random
import pandas as pd

# ======================
# CONFIG
# ======================
data_root = "temp_resampled_fixed"
random_seed = 42

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# ======================
# LOAD PATIENT LIST
# ======================
patients = sorted([
    p for p in os.listdir(data_root)
    if os.path.isdir(os.path.join(data_root, p))
])

total_patients = len(patients)

# ======================
# SHUFFLE
# ======================
random.seed(random_seed)
random.shuffle(patients)

# ======================
# SPLIT
# ======================
n_train = int(total_patients * train_ratio)
n_val = int(total_patients * val_ratio)

train_patients = patients[:n_train]
val_patients = patients[n_train:n_train+n_val]
test_patients = patients[n_train+n_val:]

# ======================
# LEAKAGE CHECK
# ======================
assert len(set(train_patients) & set(val_patients)) == 0
assert len(set(train_patients) & set(test_patients)) == 0
assert len(set(val_patients) & set(test_patients)) == 0

# ======================
# SAVE SPLIT
# ======================
records = []

for p in train_patients:
    records.append({"patient": p, "split": "train"})

for p in val_patients:
    records.append({"patient": p, "split": "val"})

for p in test_patients:
    records.append({"patient": p, "split": "test"})

df_split = pd.DataFrame(records)
df_split.to_csv("step9_patient_split.csv", index=False)

print("Total patients:", total_patients)
print("Train:", len(train_patients))
print("Val:", len(val_patients))
print("Test:", len(test_patients))
print("\nSaved: step9_patient_split.csv")
