import os
import numpy as np
from tqdm import tqdm

DATASET_ROOT = r"./data/dataset_5fold_final_v4"

MODALITIES = ["T1","T2","FLAIR","T1CE"]

MIN_BRAIN_AREA = 5000
EDGE_MARGIN = 5


def detect_triangle_artifact(mask):

    h, w = mask.shape

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


def touching_edge(mask):

    h, w = mask.shape

    if np.sum(mask[:EDGE_MARGIN,:]) > 0:
        return True

    if np.sum(mask[-EDGE_MARGIN:,:]) > 0:
        return True

    if np.sum(mask[:,:EDGE_MARGIN]) > 0:
        return True

    if np.sum(mask[:,-EDGE_MARGIN:]) > 0:
        return True

    return False


print("\n=== DATASET CORRUPTION AUDIT ===\n")

total = 0

empty_slices = []
small_brain = []
triangle_artifacts = []
edge_touch = []
constant_slices = []


for root, dirs, files in os.walk(DATASET_ROOT):

    if not any(m in root for m in MODALITIES):
        continue

    modality = os.path.basename(root)

    for f in tqdm(files):

        if not f.endswith(".npy"):
            continue

        total += 1

        path = os.path.join(root,f)

        try:
            arr = np.load(path)
        except:
            print("CORRUPTED FILE:", path)
            continue

        if arr.ndim == 3:
            img = arr[1]
        else:
            img = arr

        if np.std(img) < 1e-6:
            constant_slices.append(path)
            continue

        mask = img != 0

        brain_area = np.sum(mask)

        if brain_area == 0:
            empty_slices.append(path)
            continue

        if brain_area < MIN_BRAIN_AREA:
            small_brain.append(path)

        if detect_triangle_artifact(mask):
            triangle_artifacts.append(path)

        if touching_edge(mask):
            edge_touch.append(path)


print("\n=== AUDIT SUMMARY ===\n")

print("Total slices:", total)

print("\nEmpty slices:", len(empty_slices))
print("Small brain area:", len(small_brain))
print("Triangle artifacts:", len(triangle_artifacts))
print("Touching edge:", len(edge_touch))
print("Constant slices:", len(constant_slices))


def print_examples(title, arr):

    print("\n",title)

    for p in arr[:10]:
        print(p)


print_examples("EMPTY", empty_slices)
print_examples("SMALL BRAIN", small_brain)
print_examples("TRIANGLE ARTIFACT", triangle_artifacts)
print_examples("EDGE TOUCH", edge_touch)
print_examples("CONSTANT", constant_slices)