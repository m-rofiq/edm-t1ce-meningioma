import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import ssim, ms_ssim


# ==================================================
# HELPERS
# ==================================================
def safe_mean(x, fallback_tensor):
    if x.numel() == 0:
        return fallback_tensor.new_tensor(0.0)
    return x.mean()


def masked_l1(pred, target, mask):
    return safe_mean(torch.abs(pred - target)[mask], pred)


def masked_mse(pred, target, mask):
    return safe_mean((pred - target).pow(2)[mask], pred)


# ==================================================
# EDGE LOSS
# ==================================================
def edge_loss(pred, target, mask):
    sobel_x = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=pred.dtype,
        device=pred.device
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1, -2, -1],
         [0,  0,  0],
         [1,  2,  1]],
        dtype=pred.dtype,
        device=pred.device
    ).view(1, 1, 3, 3)

    px = F.conv2d(pred, sobel_x, padding=1)
    py = F.conv2d(pred, sobel_y, padding=1)

    tx = F.conv2d(target, sobel_x, padding=1)
    ty = F.conv2d(target, sobel_y, padding=1)

    pe = torch.sqrt(px.pow(2) + py.pow(2) + 1e-8)
    te = torch.sqrt(tx.pow(2) + ty.pow(2) + 1e-8)

    return masked_l1(pe, te, mask)


# ==================================================
# LAPLACIAN LOSS
# ==================================================
def laplacian_loss(pred, target, mask):
    kernel = torch.tensor(
        [[0, -1, 0],
         [-1, 4, -1],
         [0, -1, 0]],
        dtype=pred.dtype,
        device=pred.device
    ).view(1, 1, 3, 3)

    p = F.conv2d(pred, kernel, padding=1)
    t = F.conv2d(target, kernel, padding=1)

    return masked_l1(p, t, mask)


# ==================================================
# SSIM LOSSES
# ==================================================
def ssim_loss(pred, target, mask):
    pred = torch.clamp(pred * mask.float(), 0.0, 1.0)
    target = torch.clamp(target * mask.float(), 0.0, 1.0)

    return 1.0 - ssim(
        pred,
        target,
        data_range=1.0,
        size_average=True
    )


def ms_ssim_loss(pred, target, mask):
    pred = torch.clamp(pred * mask.float(), 0.0, 1.0)
    target = torch.clamp(target * mask.float(), 0.0, 1.0)

    return 1.0 - ms_ssim(
        pred,
        target,
        data_range=1.0,
        size_average=True
    )


# ==================================================
# VGG PERCEPTUAL
# ==================================================
class VGGPerceptual(nn.Module):
    def __init__(self):
        super().__init__()

        vgg = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1
        ).features.eval()

        for p in vgg.parameters():
            p.requires_grad = False

        self.s1 = nn.Sequential(*list(vgg)[:4])
        self.s2 = nn.Sequential(*list(vgg)[4:9])
        self.s3 = nn.Sequential(*list(vgg)[9:16])

    def forward(self, pred, target):
        pred = pred.repeat(1, 3, 1, 1)
        target = target.repeat(1, 3, 1, 1)

        p1 = self.s1(pred)
        t1 = self.s1(target)

        p2 = self.s2(p1)
        t2 = self.s2(t1)

        p3 = self.s3(p2)
        t3 = self.s3(t2)

        return (
            F.l1_loss(p1, t1) +
            F.l1_loss(p2, t2) +
            F.l1_loss(p3, t3)
        ) / 3.0


# ==================================================
# GLOBAL OBJECT
# ==================================================
perceptual_model = None


# ==================================================
# MAIN LOSS
# ==================================================
class EDMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_tuple, target, mask, input_tensor):
        global perceptual_model

        pred, base, enh = pred_tuple
        mask_f = mask.float()

        t1 = input_tensor[:, 1:2, :, :]
        diff = target - t1

        healthy = (diff <= 0.025) & mask
        tumor = (diff > 0.025) & mask

        # -------------------------
        # Pixel losses
        # -------------------------
        loss_global = masked_l1(pred, target, mask)
        loss_tumor = masked_l1(pred, target, tumor)
        loss_base = masked_l1(base, target, healthy)

        enh_gt = torch.clamp(target - base.detach(), min=0.0)

        if tumor.any():
            loss_enh = masked_l1(enh, enh_gt, tumor)
        else:
            loss_enh = masked_l1(enh, enh_gt, mask)

        # -------------------------
        # Structural losses
        # -------------------------
        loss_edge = edge_loss(pred, target, mask)
        loss_lap = laplacian_loss(pred, target, mask)
        loss_msssim = ms_ssim_loss(pred, target, mask)

        # -------------------------
        # Perceptual
        # -------------------------
        if perceptual_model is None:
            perceptual_model = VGGPerceptual().to(pred.device).eval()

        pred_clip = torch.clamp(pred, 0.0, 1.0)
        target_clip = torch.clamp(target, 0.0, 1.0)

        loss_vgg = perceptual_model(
            (pred_clip * mask_f).float(),
            (target_clip * mask_f).float()
        )

        # -------------------------
        # Final total
        # -------------------------
        total = (
            0.35 * loss_global +
            1.50 * loss_tumor +
            1.00 * loss_base +
            1.20 * loss_enh +
            0.08 * loss_edge +
            0.10 * loss_lap +
            0.08 * loss_vgg +
            0.05 * loss_msssim
        )

        return total


LOSS_REGISTRY = {
    "EDMLoss": EDMLoss()
}