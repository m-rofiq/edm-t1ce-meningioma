import os
import SimpleITK as sitk
import numpy as np
import torch
from tqdm import tqdm

ROOT = r"./work/temp_registered"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPS = 1e-8
BINS = 64

def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + EPS)

def compute_ncc(x,y):
    x = x - torch.mean(x)
    y = y - torch.mean(y)
    return torch.sum(x*y) / (
        torch.sqrt(torch.sum(x**2)) * torch.sqrt(torch.sum(y**2)) + EPS
    )

def compute_mi(x,y,bins=BINS):
    x = torch.clamp((x*(bins-1)).long(),0,bins-1)
    y = torch.clamp((y*(bins-1)).long(),0,bins-1)

    joint = torch.zeros((bins,bins),device=DEVICE)
    joint.index_put_((x.flatten(),y.flatten()),
                     torch.ones_like(x.flatten(),dtype=torch.float32),
                     accumulate=True)

    pxy = joint/torch.sum(joint)
    px  = torch.sum(pxy,dim=1)
    py  = torch.sum(pxy,dim=0)

    px_py = px.unsqueeze(1)*py.unsqueeze(0)
    nz = pxy>0
    return torch.sum(pxy[nz]*torch.log(pxy[nz]/(px_py[nz]+EPS)))

results = []

for patient in tqdm(os.listdir(ROOT)):

    p_path = os.path.join(ROOT, patient)
    if not os.path.isdir(p_path):
        continue

    t1 = sitk.GetArrayFromImage(
        sitk.ReadImage(os.path.join(p_path,"T1.nii.gz"))
    )
    t1 = normalize(t1)

    for mod in ["T2","FLAIR","T1CE"]:

        img = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(p_path,f"{mod}.nii.gz"))
        )
        img = normalize(img)

        t1_t = torch.tensor(t1,device=DEVICE)
        img_t = torch.tensor(img,device=DEVICE)

        ncc = compute_ncc(t1_t,img_t).item()
        mi  = compute_mi(t1_t,img_t).item()

        results.append((mod,ncc,mi))

import pandas as pd

df = pd.DataFrame(results,columns=["modality","NCC","MI"])

print("\n=== MEAN ALIGNMENT METRICS ===")
print(df.groupby("modality").mean())
