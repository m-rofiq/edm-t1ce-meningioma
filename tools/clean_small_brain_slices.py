import os
import numpy as np
import shutil
from tqdm import tqdm

SRC_ROOT = r"./data/dataset_5fold_final_v2"
DST_ROOT = r"./data/dataset_5fold_final_v3"

MODALITIES = ["T1","T2","FLAIR","T1CE"]

MIN_BRAIN_AREA = 5000


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def process_split(src_path, dst_path):

    t1_dir = os.path.join(src_path,"T1")

    if not os.path.exists(t1_dir):
        return 0,0

    files = sorted(os.listdir(t1_dir))

    for m in MODALITIES:
        ensure_dir(os.path.join(dst_path,m))

    total = 0
    removed = 0

    for f in tqdm(files):

        total += 1

        t1ce = np.load(os.path.join(src_path,"T1CE",f))

        center = t1ce[1]
        brain_mask = center != 0

        brain_area = np.sum(brain_mask)

        if brain_area < MIN_BRAIN_AREA:

            removed += 1
            continue

        for m in MODALITIES:

            src_file = os.path.join(src_path,m,f)
            dst_file = os.path.join(dst_path,m,f)

            shutil.copy(src_file,dst_file)

    return total, removed


print("\n=== CLEAN SMALL BRAIN SLICES ===\n")

total_all = 0
removed_all = 0

for folder in os.listdir(SRC_ROOT):

    src_folder = os.path.join(SRC_ROOT,folder)
    dst_folder = os.path.join(DST_ROOT,folder)

    if not os.path.isdir(src_folder):
        continue


    # case 1 : fold structure
    if folder.startswith("fold"):

        for split in os.listdir(src_folder):

            split_src = os.path.join(src_folder,split)
            split_dst = os.path.join(dst_folder,split)

            if not os.path.isdir(split_src):
                continue

            t,r = process_split(split_src,split_dst)

            total_all += t
            removed_all += r


    # case 2 : holdout_test
    elif folder == "holdout_test":

        t,r = process_split(src_folder,dst_folder)

        total_all += t
        removed_all += r


print("\n=== CLEANING COMPLETE ===\n")

print("Total slices:", total_all)
print("Removed slices:", removed_all)
print("Remaining:", total_all-removed_all)