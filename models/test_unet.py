import torch
from unet_baseline import UNetBaseline

model = UNetBaseline()

x = torch.randn(1, 3, 512, 512)
y = model(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)