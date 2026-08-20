import os
import shutil

SRC = r"./data/dataset_5fold_final_v4/fold_4/train"
DST = r"./data/dataset_sanity"

PATIENT_ID = "0210a536"

for mod in ["T1","T1CE"]:

    os.makedirs(os.path.join(DST,"train",mod), exist_ok=True)
    os.makedirs(os.path.join(DST,"val",mod), exist_ok=True)

    src_mod = os.path.join(SRC,mod)

    for f in os.listdir(src_mod):

        if f.startswith(PATIENT_ID):

            shutil.copy(
                os.path.join(src_mod,f),
                os.path.join(DST,"train",mod,f)
            )

            shutil.copy(
                os.path.join(src_mod,f),
                os.path.join(DST,"val",mod,f)
            )