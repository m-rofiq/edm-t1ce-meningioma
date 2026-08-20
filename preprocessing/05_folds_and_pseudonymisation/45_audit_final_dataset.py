import os

ROOT = r"./data/dataset_5fold_final_v2"
modalities = ["T1","T1CE","T2","FLAIR"]

# ---- 1. Audit fold pairing ----
for fold in sorted([f for f in os.listdir(ROOT) if f.startswith("fold_")]):

    print("\n======", fold, "======")

    for split in ["train","test"]:
        print("\n--", split.upper(), "--")

        counts = []
        for mod in modalities:
            folder = os.path.join(ROOT, fold, split, mod)
            n = len(os.listdir(folder))
            counts.append(n)
            print(mod, ":", n)

        print("Pairing consistent:", len(set(counts)) == 1)


# ---- 2. Audit sacred test isolation ----
print("\n====== SACRED TEST ======")

counts = []
for mod in modalities:
    folder = os.path.join(ROOT, "sacred_test", mod)
    n = len(os.listdir(folder))
    counts.append(n)
    print(mod, ":", n)

print("Sacred pairing consistent:", len(set(counts)) == 1)