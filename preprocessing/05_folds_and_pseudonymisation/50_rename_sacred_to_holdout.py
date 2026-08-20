import os

ROOT = r"./data/dataset_5fold_final_v2"

old_path = os.path.join(ROOT, "sacred_test")
new_path = os.path.join(ROOT, "holdout_test")

if os.path.exists(old_path):
    os.rename(old_path, new_path)

print("Sacred test renamed to holdout_test.")