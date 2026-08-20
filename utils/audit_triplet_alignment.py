import os
import numpy as np


DATASET_ROOT = r"./data/dataset_5fold_final_v2"

FOLD = 0


def normalize(x):

    x = x.astype(np.float32)

    x -= x.mean()
    x /= (x.std() + 1e-8)

    return x


def corr(a,b):

    a = normalize(a)
    b = normalize(b)

    return np.mean(a*b)


t1_dir = os.path.join(DATASET_ROOT, f"fold_{FOLD}", "train", "T1")
t1ce_dir = os.path.join(DATASET_ROOT, f"fold_{FOLD}", "train", "T1CE")

files = sorted([f for f in os.listdir(t1_dir) if f.endswith(".npy")])

center_scores = []
prev_scores = []
next_scores = []

for f in files[:200]:   # sample cukup

    t1 = np.load(os.path.join(t1_dir,f))
    t1ce = np.load(os.path.join(t1ce_dir,f))

    z_prev = t1[0]
    z_center = t1[1]
    z_next = t1[2]

    target = t1ce[1]

    center_scores.append(corr(z_center,target))
    prev_scores.append(corr(z_prev,target))
    next_scores.append(corr(z_next,target))


print("\n=== Triplet Alignment Audit ===\n")

print("corr(z , target): ", np.mean(center_scores))
print("corr(z-1,target):", np.mean(prev_scores))
print("corr(z+1,target):", np.mean(next_scores))