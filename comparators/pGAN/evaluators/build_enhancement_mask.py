import torch

def build_enhancement_mask(t1ce, brain_mask, k=1.5):

    brain_pixels = t1ce[brain_mask == 1]

    if brain_pixels.numel() == 0:
        return torch.zeros_like(t1ce)

    mu = torch.mean(brain_pixels)
    sigma = torch.std(brain_pixels)

    threshold = mu + k * sigma

    enh_mask = (t1ce > threshold).float() * brain_mask

    return enh_mask