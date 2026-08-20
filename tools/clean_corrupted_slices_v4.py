import os
import numpy as np
import shutil
from tqdm import tqdm

SRC_ROOT = r"./data/dataset_5fold_final_v3"
DST_ROOT = r"./data/dataset_5fold_final_v4"

MODALITIES = ["T1","T2","FLAIR","T1CE"]

VAR_THRESHOLD = 1e-6


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def detect_triangle_artifact(mask):

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return True

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    bbox_area = (x2-x1+1)*(y2-y1+1)
    brain_area = np.sum(mask)

    ratio = brain_area / bbox_area

    if ratio < 0.55:
        return True

    return False


def process_split(src_path, dst_path):

    ensure_dir(dst_path)

    t1ce_dir = os.path.join(src_path,"T1CE")

    if not os.path.exists(t1ce_dir):
        return 0,0

    files = sorted(os.listdir(t1ce_dir))

    for m in MODALITIES:
        ensure_dir(os.path.join(dst_path,m))

    total = 0
    removed = 0

    for f in tqdm(files):

        total += 1

        path = os.path.join(t1ce_dir,f)

        arr = np.load(path)

        center = arr[1]

        if np.std(center) < VAR_THRESHOLD:
            removed += 1
            continue

        mask = center != 0

        if detect_triangle_artifact(mask):
            removed += 1
            continue

        for m in MODALITIES:

            src_file = os.path.join(src_path,m,f)
            dst_file = os.path.join(dst_path,m,f)

            shutil.copy(src_file,dst_file)

    return total, removed


print("\n=== CLEAN CORRUPTED SLICES ===\n")

total_all = 0
removed_all = 0


for folder in os.listdir(SRC_ROOT):

    src_folder = os.path.join(SRC_ROOT,folder)
    dst_folder = os.path.join(DST_ROOT,folder)

    if not os.path.isdir(src_folder):
        continue


    if folder.startswith("fold"):

        for split in os.listdir(src_folder):

            split_src = os.path.join(src_folder,split)
            split_dst = os.path.join(dst_folder,split)

            if not os.path.isdir(split_src):
                continue

            t,r = process_split(split_src,split_dst)

            total_all += t
            removed_all += r


    elif folder == "holdout_test":

        t,r = process_split(src_folder,dst_folder)

        total_all += t
        removed_all += r


print("\n=== CLEANING COMPLETE ===\n")

print("Total slices:", total_all)
print("Removed slices:", removed_all)
print("Remaining:", total_all-removed_all)