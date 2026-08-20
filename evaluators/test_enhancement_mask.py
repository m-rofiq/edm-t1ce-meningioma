import torch
from build_enhancement_mask import build_enhancement_mask

x = torch.randn(1,1,512,512)
mask = (x != 0).float()

enh = build_enhancement_mask(x, mask)

print("Enhancement mask mean:", torch.mean(enh))