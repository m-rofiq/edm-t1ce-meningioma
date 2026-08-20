import torch
from psnr_fixed_range import gradient_difference_tumor

x = torch.randn(1,1,512,512)
mask = torch.zeros_like(x)
mask[:, :, 200:300, 200:300] = 1  # fake tumor region

val = gradient_difference_tumor(x, x, mask)

print("Gradient tumor identical:", val)