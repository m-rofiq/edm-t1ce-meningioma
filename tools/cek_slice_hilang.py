import os

v2 = r"./data/dataset_5fold_final_v2"
v3 = r"./data/dataset_5fold_final_v3"

for root, dirs, files in os.walk(v2):

    if root.endswith("T1"):

        rel = os.path.relpath(root, v2)
        root_v3 = os.path.join(v3, rel)

        files_v2 = set(files)
        files_v3 = set(os.listdir(root_v3))

        removed = files_v2 - files_v3

        for f in removed:

            print("REMOVED:", os.path.join(rel, f))