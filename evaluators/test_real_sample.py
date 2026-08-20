import os
import torch
import numpy as np
from psnr_fixed_range import psnr_fixed, psnr_roi
from psnr_fixed_range import calibration_bias
from ssim_masked import ssim_roi
from psnr_fixed_range import variance_ratio
from psnr_fixed_range import gradient_difference_tumor

# === ambil satu sample GT dari holdout_test T1CE ===
gt_path = r"./data/dataset_5fold_final_v2/holdout_test/T1CE"

# ganti dengan salah satu file nyata di folder itu
filename = os.listdir(gt_path)[0]

gt = np.load(os.path.join(gt_path, filename))

# ambil channel tengah saja (index 1)
gt = torch.tensor(gt[1]).unsqueeze(0).unsqueeze(0)

# buat pred dummy (GT + noise kecil)
pred = gt + torch.randn_like(gt) * 0.05

mask = (gt != 0).float()

print("Global PSNR:", psnr_fixed(pred, gt))
print("ROI PSNR:", psnr_roi(pred, gt, mask))
print("Calibration Bias:", calibration_bias(pred, gt, mask))
print("SSIM ROI:", ssim_roi(pred, gt, mask))
print("Variance Ratio:", variance_ratio(pred, gt, mask))
print("Gradient Difference:", gradient_difference_tumor(pred, gt, mask))