import torch
from psnr_fixed_range import psnr_roi

x = torch.randn(1, 512, 512)
mask = (x != 0).float()

val = psnr_roi(x, x, mask, data_range=5.3)

print("PSNR identical:", val)