import torch
from ssim_masked import ssim_roi

# dummy identical tensor
x = torch.randn(1, 512, 512)
mask = (x != 0).float()

val = ssim_roi(x, x, mask, data_range=5.3)

print("SSIM identical:", val)