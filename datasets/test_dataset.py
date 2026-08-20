from dataset_2p5d import MRI2p5DDataset
import torch

DATA_PATH = r"./data/dataset_5fold_final_v2/fold_0/train"

ds = MRI2p5DDataset(DATA_PATH)

print("Total samples:", len(ds))

# ambil sample pertama
x, y, fname = ds[0]

print("Filename:", fname)
print("Input shape:", x.shape)
print("Target shape:", y.shape)
print("Input dtype:", x.dtype)
print("Target dtype:", y.dtype)

# shape validation
assert x.shape == (3, 512, 512), f"Unexpected input shape: {x.shape}"
assert y.shape == (1, 512, 512), f"Unexpected target shape: {y.shape}"

# dtype validation
assert x.dtype == torch.float32
assert y.dtype == torch.float32

# check NaN / Inf
assert not torch.isnan(x).any(), "NaN detected in input"
assert not torch.isnan(y).any(), "NaN detected in target"
assert not torch.isinf(x).any(), "Inf detected in input"
assert not torch.isinf(y).any(), "Inf detected in target"

print("Dataset check PASSED")