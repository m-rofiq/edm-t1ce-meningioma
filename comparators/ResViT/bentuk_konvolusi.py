import torch
p = r"./comparators/ResViT/experiments/EXP-RESVIT_Dalmaz2022_fold0/best_model.pth"
ck = torch.load(p, map_location="cpu", weights_only=False)
conv4d = [(k, tuple(v.shape)) for k, v in ck.items() if torch.is_tensor(v) and v.dim() == 4]
print("konvolusi pertama :", conv4d[0])
print("konvolusi terakhir:", conv4d[-1])