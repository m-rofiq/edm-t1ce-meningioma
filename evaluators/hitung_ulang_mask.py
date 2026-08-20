"""
Hitung ulang statistik mask enhancement HANYA pada bidang tengah.

Latar: berkas T1CE hold-out berbentuk (3, 512, 512) -- pengodean 2.5D
[S_{i-1}, S_i, S_{i+1}]. Skrip cek sebelumnya menghitung ketiga bidang,
sehingga tiap slice terhitung sampai tiga kali dan menghasilkan angka 411.

Slice sebenarnya = bidang tengah (indeks 1) dari tiap berkas = 1 per berkas.
Skrip ini menghitung ulang di atas himpunan itu, sehingga angka yang
dilaporkan di Methods cocok dengan slice yang benar-benar dievaluasi.

    python hitung_ulang_mask.py
"""
import os
import glob
import numpy as np

DIR = r"./data/dataset_5fold_final_v4/holdout_test/T1CE"
K = 1.5
MIN_ENH_VOXELS = 10
CLAMP_CEILING = 1.0

files = sorted(glob.glob(os.path.join(DIR, "*.npy")))
if not files:
    raise SystemExit(f"Tidak ada .npy di:\n  {DIR}")

frac, clipped, thr = [], [], []
skipped = 0

for f in files:
    a = np.load(f)
    if a.ndim == 3 and a.shape[0] == 3:
        sl = a[1]                    # bidang tengah = slice sebenarnya
    elif a.ndim == 3 and a.shape[-1] == 3:
        sl = a[..., 1]
    elif a.ndim == 2:
        sl = a
    else:
        raise SystemExit(f"Bentuk tak dikenali: {a.shape} pada {f}")

    brain = sl[sl != 0]
    if brain.size < 100:
        skipped += 1
        continue
    mu, sd = brain.mean(), brain.std()
    t = mu + K * sd
    enh = brain[brain > t]
    if enh.size <= MIN_ENH_VOXELS:
        skipped += 1
        continue
    thr.append(t)
    frac.append(enh.size / brain.size)
    clipped.append((enh >= CLAMP_CEILING).mean())

frac = np.array(frac); clipped = np.array(clipped); thr = np.array(thr)

print(f"Berkas                       : {len(files)}")
print(f"Slice berkontribusi          : {len(frac)}   (dilewati: {skipped})")
print()
print("Angka untuk Methods:")
print(f"  ukuran mask, rerata        : {100*frac.mean():.1f}%")
print(f"  ukuran mask, median        : {100*np.median(frac):.1f}%")
print(f"  slice ambang > plafon      : {100*(thr>CLAMP_CEILING).mean():.1f}%")
print(f"  voxel enh terpotong ke 1.0 : {100*clipped.mean():.1f}%")
print()
print("Salin ke Methods:")
print(f"  \"it selects the brightest ${100*frac.mean():.1f}\\%$ of brain voxels on average")
print(f"   (median ${100*np.median(frac):.1f}\\%$ over the ${len(frac)}$ hold-out reference slices)\"")
print(f"  \"...exceeds the clipping ceiling on ${100*(thr>CLAMP_CEILING).mean():.1f}\\%$ of hold-out slices,")
print(f"   and ${100*clipped.mean():.1f}\\%$ of the voxels within $M_{{\\mathrm{{Enh}}}}$...\"")
