import os
import numpy as np

# Pastikan path ini menunjuk ke dataset V2 Anda
t1_dir = r"./data/dataset_5fold_final_v2/holdout_test/T1"
t1ce_dir = r"./data/dataset_5fold_final_v2/holdout_test/T1CE"
patient_id = "c1a6046d"

print(f"\n===== AUDIT FORENSIK SPASIAL: PASIEN {patient_id} =====")

files = sorted([f for f in os.listdir(t1_dir) if f.startswith(patient_id)])

for f in files:
    t1 = np.load(os.path.join(t1_dir, f))[1]   # Ambil slice tengah (channel 1)
    t1ce = np.load(os.path.join(t1ce_dir, f))[1]
    
    mask_t1 = t1 != 0
    mask_t1ce = t1ce != 0
    
    area_t1 = mask_t1.sum()
    area_t1ce = mask_t1ce.sum()
    
    # Hitung selisih luasan anatomi
    if area_t1 > 0:
        diff_percent = abs(area_t1 - area_t1ce) / area_t1 * 100
    else:
        diff_percent = 0.0
        
    # Hitung korelasi piksel (jika bentuknya sama, harusnya > 0.8)
    if area_t1 > 0 and area_t1ce > 0:
        flat_t1 = t1.flatten()
        flat_t1ce = t1ce.flatten()
        correlation = np.corrcoef(flat_t1, flat_t1ce)[0, 1]
    else:
        correlation = 0.0

    print(f"Slice: {f.split('.')[0][-2:]} | Area T1: {area_t1:5d} | Area T1CE: {area_t1ce:5d} | Diff: {diff_percent:5.1f}% | Spatial Corr: {correlation:.3f}")