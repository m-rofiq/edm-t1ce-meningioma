import os

DATASET_ROOT = r"./data/dataset_5fold_final_v4"

MOD = "T1"   # cukup cek satu modality karena pairing sama


def analyze_split(path):

    folder = os.path.join(path, MOD)

    files = [f for f in os.listdir(folder) if f.endswith(".npy")]

    patients = set()

    for f in files:
        pid = f.split("_slice")[0]
        patients.add(pid)

    return len(files), len(patients)


print("\n=== DATASET DISTRIBUTION ===\n")

total_slices = 0
total_patients = set()

for fold in sorted(os.listdir(DATASET_ROOT)):

    fold_path = os.path.join(DATASET_ROOT, fold)

    if not os.path.isdir(fold_path):
        continue

    # HOLDOUT
    if fold == "holdout_test":

        slices, patients = analyze_split(fold_path)

        print(f"HOLDOUT TEST")
        print("patients:", patients)
        print("slices :", slices)
        print()

        continue

    # FOLDS
    if fold.startswith("fold"):

        train_path = os.path.join(fold_path, "train")
        val_path = os.path.join(fold_path, "val")

        train_s, train_p = analyze_split(train_path)
        val_s, val_p = analyze_split(val_path)

        print(f"{fold}")
        print(" train patients:", train_p)
        print(" val patients  :", val_p)
        print(" train slices  :", train_s)
        print(" val slices    :", val_s)
        print()

        total_slices += train_s + val_s

print("TOTAL TRAIN+VAL SLICES:", total_slices)