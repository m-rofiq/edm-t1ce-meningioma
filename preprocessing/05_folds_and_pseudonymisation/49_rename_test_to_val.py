import os

ROOT = r"./data/dataset_5fold_final_v2"

for folder in os.listdir(ROOT):
    if folder.startswith("fold_"):
        test_path = os.path.join(ROOT, folder, "test")
        val_path = os.path.join(ROOT, folder, "val")
        if os.path.exists(test_path):
            os.rename(test_path, val_path)

print("Renamed fold test folders to val.")