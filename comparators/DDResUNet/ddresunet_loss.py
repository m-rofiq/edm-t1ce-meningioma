# Loss untuk DD-Res U-Net baseline: L1 + 5*(1-SSIM), sesuai formula custom_loss
# di paper asli (Osman & Tamam, 2023). Ditulis dengan gaya yang sama dengan
# fungsi loss lain di loss_registry Anda (masked_l1, edge_loss, dst).
#
# CATATAN SSIM & masking: SSIM adalah operasi ber-jendela spasial (butuh
# neighborhood piksel), jadi TIDAK bisa dihitung dari daftar piksel ter-mask
# secara flat seperti masked_l1 (pred[mask]). Pendekatan yang dipakai di sini
# mengikuti konvensi yang SUDAH ADA di codebase Anda untuk kasus serupa
# (lihat perceptual_loss di loss_registry_exp602.py: `pred = pred * mask`) --
# yaitu masking lewat perkalian (zero-out background), baru SSIM dihitung
# pada citra penuh. L1 tetap dihitung lewat indexing seperti masked_l1 asli
# (lebih presisi, tidak kena efek "leakage" jendela SSIM di piksel background).
#
# data_range TIDAK dihardcode -- mengikuti convention psnr_fixed_range.py,
# dibaca dari configs/metric_config.json supaya konsisten dengan psnr_roi
# yang dipakai EXP-602/703.

import torch
import torch.nn.functional as F
from psnr_fixed_range import load_data_range


def _gaussian_window(window_size, sigma, device, dtype):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g.unsqueeze(1) @ g.unsqueeze(0)
    return window_2d.unsqueeze(0).unsqueeze(0)  # (1,1,K,K)


def ssim_map(pred, target, data_range, window_size=5, sigma=1.5):
    """SSIM map (bukan cuma skalar), dihitung per-channel lalu dirata-rata,
    memakai gaussian window filter_size=5 (sesuai filter_size=5 yang dipakai
    tf.image.ssim di kode TF asli)."""
    device, dtype = pred.device, pred.dtype
    channels = pred.shape[1]
    window = _gaussian_window(window_size, sigma, device, dtype).repeat(channels, 1, 1, 1)
    pad = window_size // 2

    mu_pred = F.conv2d(pred, window, padding=pad, groups=channels)
    mu_target = F.conv2d(target, window, padding=pad, groups=channels)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, window, padding=pad, groups=channels) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_pred_target

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)

    return numerator / (denominator + 1e-8)


def ddresunet_loss(pred, target, mask, data_range=None):
    """L1 (masked, via indexing) + 5*(1 - SSIM) (masked, via multiplication)."""
    if data_range is None:
        data_range = load_data_range()

    l1 = F.l1_loss(pred[mask], target[mask])

    mask_float = mask.float()
    pred_masked = pred * mask_float
    target_masked = target * mask_float

    smap = ssim_map(pred_masked, target_masked, data_range=data_range)
    # rata-rata SSIM hanya di area mask, supaya piksel background (nilai 0
    # hasil zero-out) tidak ikut mendominasi rata-rata SSIM.
    ssim_val = smap[mask_float.bool()].mean() if mask_float.sum() > 0 else smap.mean()

    return l1 + 5.0 * (1.0 - ssim_val)


# Tambahkan baris ini ke LOSS_REGISTRY di loss_registry.py Anda:
#
#   from losses.ddresunet_loss import ddresunet_loss
#   LOSS_REGISTRY["DDResUNetLoss"] = ddresunet_loss
