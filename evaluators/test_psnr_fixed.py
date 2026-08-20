import torch
from psnr_fixed_range import psnr_fixed, psnr_roi

a = torch.rand(1,1,512,512)
b = torch.rand(1,1,512,512)
mask = torch.ones_like(a)

print("Global:", psnr_fixed(a,b))
print("ROI:", psnr_roi(a,b,mask))