import os

ROOT = r"./data/dataset_5fold_final_v2"

phi_detected = []

for root, dirs, files in os.walk(ROOT):
    for file in files:
        if any(c.isalpha() for c in file.split("_")[0]) and not file.split("_")[0].isalnum():
            phi_detected.append(file)

print("Potential PHI files:", len(phi_detected))
print(phi_detected[:10])