import torch
from l1_loss import L1Loss

loss_fn = L1Loss()

pred = torch.randn(1,1,512,512)
gt   = torch.randn(1,1,512,512)

print("Loss:", loss_fn(pred, gt))