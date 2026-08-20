import os
import sys

# =========================================================
# FIX Q1: NAIK 1 TINGKAT KE ROOT PROYEK (04_training)
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# =========================================================

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import lpips
import torch.nn.functional as F

# PASTIKAN SEMUA METRIK INI TER-IMPORT DENGAN BENAR
from evaluators.bootstrap_ci import bootstrap_ci
from evaluators.psnr_fixed_range import psnr_roi, calibration_bias, variance_ratio, gradient_difference_tumor
from evaluators.ssim_masked import ssim_roi
from evaluators.build_enhancement_mask import build_enhancement_mask

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Inisialisasi LPIPS (Abaikan saja jika ada UserWarning 'pretrained')
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()

    # ==========================================
    # PATH DIREKTORI
    # ==========================================
    # Pastikan Path ini sudah menunjuk ke folder npy_results di pGAN Anda
    PGAN_NPY_DIR = r"./comparators/pGAN/results/exp_pGAN_baseline/test_latest/npy_results" 
    
    # Path ke Ground Truth T1CE (Fold 0 Val) dataset Anda
    GT_DIR = r"./data/dataset_5fold_final_v3/fold_0/val/T1CE"
    # ==========================================

    records = []
    files = [f for f in os.listdir(PGAN_NPY_DIR) if f.endswith('.npy')]
    
    print(f"\nMengevaluasi {len(files)} file presisi tinggi dari baseline pGAN...")
    
    with torch.no_grad():
        for f in tqdm(files, ncols=100):
            # 1. Load PRED (256x256) & GT Asli (3x512x512)
            pred_np = np.load(os.path.join(PGAN_NPY_DIR, f))
            gt_np = np.load(os.path.join(GT_DIR, f))
            
            # Ekstrak slice tengah GT (karena dataset Anda 2.5D)
            if gt_np.ndim == 3:
                gt_slice = gt_np[1]
            else:
                gt_slice = gt_np
                
            # 2. Convert to PyTorch Tensor
            pred_tensor = torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0).float().to(device)
            gt = torch.from_numpy(gt_slice).unsqueeze(0).unsqueeze(0).float().to(device)
            
            # 3. UPSAMPLE PRED kembali ke 512x512 agar adil (Apples-to-Apples)
            pred = F.interpolate(pred_tensor, size=(512, 512), mode='bilinear', align_corners=False)
            
            # Masking hanya pada area otak
            mask = (gt != 0).float()
            if mask.sum() == 0:
                continue
                
            # =========================
            # METRIK DASAR
            # =========================
            psnr = psnr_roi(pred, gt, mask).item()
            ssim = ssim_roi(pred, gt, mask).item()
            mae = torch.mean(torch.abs(pred - gt)[mask.bool()]).item()
            
            # LPIPS (Harus di clamp dan scale ke [-1, 1])
            pred_lp = torch.clamp(pred, 0.0, 1.0)
            gt_lp = torch.clamp(gt, 0.0, 1.0)
            pred_lp = (pred_lp * 2.0) - 1.0
            gt_lp = (gt_lp * 2.0) - 1.0
            if pred_lp.shape[1] == 1:
                pred_lp = pred_lp.repeat(1, 3, 1, 1)
                gt_lp = gt_lp.repeat(1, 3, 1, 1)
            
            lp = lpips_model(pred_lp, gt_lp).mean().item()
            
            # =========================
            # METRIK LANJUTAN (TUMOR)
            # =========================
            bias = calibration_bias(pred, gt, mask).item()
            var_ratio = variance_ratio(pred, gt, mask).item()
            
            enh_mask = build_enhancement_mask(gt, mask)
            if torch.sum(enh_mask) > 10:
                psnr_enh = psnr_roi(pred, gt, enh_mask).item()
                ssim_enh = ssim_roi(pred, gt, enh_mask).item()
                grad_enh = gradient_difference_tumor(pred, gt, enh_mask).item()
            else:
                psnr_enh = np.nan
                ssim_enh = np.nan
                grad_enh = np.nan
                
            records.append({
                "patient_id": f,
                "psnr_roi": psnr,
                "ssim_roi": ssim,
                "mae": mae,
                "calibration_bias": bias,
                "variance_ratio": var_ratio,
                "psnr_enh": psnr_enh,
                "ssim_enh": ssim_enh,
                "grad_enh": grad_enh,
                "lpips": lp
            })
    
    df = pd.DataFrame(records)
    
    # Ambil hanya kolom numerik untuk perhitungan rata-rata
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    print("\n" + "="*50)
    print("HASIL BASELINE pGAN (Tantangan untuk EXP-702)")
    print("="*50)
    for col in numeric_cols:
        mean_val = df[col].mean()
        print(f"{col:<20}: {mean_val:.4f}")
    print("="*50)

    # Simpan ke CSV dengan mengabaikan string
    df_summary = pd.DataFrame([df.mean(numeric_only=True).to_dict()])
    df_summary.to_csv("summary_pGAN_fold0.csv", index=False)
        
if __name__ == "__main__":
    main()