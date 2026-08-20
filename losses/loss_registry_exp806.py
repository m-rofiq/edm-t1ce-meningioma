import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import ms_ssim


def masked_l1(pred, target, mask):
    return F.l1_loss(pred[mask], target[mask])


def masked_mse(pred, target, mask):
    return F.mse_loss(pred[mask], target[mask])


def edge_loss(pred, target, mask):
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=pred.dtype, device=pred.device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=pred.dtype, device=pred.device
    ).view(1, 1, 3, 3)
    px = F.conv2d(pred,   sobel_x, padding=1)
    py = F.conv2d(pred,   sobel_y, padding=1)
    tx = F.conv2d(target, sobel_x, padding=1)
    ty = F.conv2d(target, sobel_y, padding=1)
    pe = torch.sqrt(px.pow(2) + py.pow(2) + 1e-8)
    te = torch.sqrt(tx.pow(2) + ty.pow(2) + 1e-8)
    return F.l1_loss(pe[mask], te[mask])


def laplacian_loss(pred, target, mask):
    kernel = torch.tensor(
        [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
        dtype=pred.dtype, device=pred.device
    ).view(1, 1, 3, 3)
    p = F.conv2d(pred,   kernel, padding=1)
    t = F.conv2d(target, kernel, padding=1)
    return F.l1_loss(p[mask], t[mask])


def ms_ssim_loss(pred, target, mask):
    pred_m   = torch.clamp(pred   * mask.float(), 0.0, 1.0)
    target_m = torch.clamp(target * mask.float(), 0.0, 1.0)
    return 1.0 - ms_ssim(pred_m, target_m, data_range=1.0, size_average=True)


class VGGPerceptual(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.s1 = nn.Sequential(*list(vgg)[:4])
        self.s2 = nn.Sequential(*list(vgg)[4:9])
        self.s3 = nn.Sequential(*list(vgg)[9:16])

    def forward(self, pred, target):
        pred   = pred.repeat(1, 3, 1, 1)
        target = target.repeat(1, 3, 1, 1)
        p1 = self.s1(pred);  t1 = self.s1(target)
        p2 = self.s2(p1);    t2 = self.s2(t1)
        p3 = self.s3(p2);    t3 = self.s3(t2)
        return (F.l1_loss(p1,t1) + F.l1_loss(p2,t2) + F.l1_loss(p3,t3)) / 3.0


class ROIContrastEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(net.children())[:6])
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.features(x.repeat(1, 3, 1, 1))


def roi_contrastive_loss(pred, target, tumor_mask):
    global roi_encoder
    if not tumor_mask.any():
        return pred.new_tensor(0.0)
    if roi_encoder is None:
        roi_encoder = ROIContrastEncoder().to(pred.device).eval()
    pred   = torch.clamp(pred,   0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)
    fp = roi_encoder(pred   * tumor_mask.float())
    ft = roi_encoder(target * tumor_mask.float())
    return F.l1_loss(fp, ft)


# ======================================================
# FOCAL FREQUENCY LOSS
# [CHANGED vs 805] alpha 2.0 → 1.5
# Mengurangi spike train loss yang terjadi di epoch
# 31 dan 48 pada EXP-805 akibat penalti kuadratik
# terlalu agresif pada batch dengan banyak fine detail.
# alpha=1.5 tetap lebih kuat dari 803/804 (alpha=1.0)
# tapi lebih stabil.
# ======================================================
class FocalFrequencyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, alpha=1.5, patch_factor=1,
                 ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super().__init__()
        self.loss_weight  = loss_weight
        self.alpha        = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix   = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        pf = self.patch_factor
        _, _, H, W = x.shape
        pH, pW = H // pf, W // pf
        patches = []
        for i in range(pf):
            for j in range(pf):
                p    = x[..., i*pH:(i+1)*pH, j*pW:(j+1)*pW]
                freq = torch.fft.fft2(p, norm='ortho')
                patches.append(torch.stack([freq.real, freq.imag], dim=-1))
        return torch.stack(patches, dim=1)

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        tmp           = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]
        w = (matrix.detach() if matrix is not None
             else freq_distance.clone().detach())
        if self.log_matrix:
            w = torch.log(w + 1.0)
        w = w ** self.alpha
        if self.batch_matrix:
            w = w / (w.sum() + 1e-8)
        else:
            B = w.shape[0]
            w = w / (w.reshape(B, -1).sum(1, keepdim=True)
                      .reshape(B, *([1]*(w.dim()-1))) + 1e-8)
        with torch.amp.autocast('cuda', enabled=False):
            loss = torch.mean(w.float() * freq_distance.float())
        return loss

    def forward(self, pred, target, matrix=None):
        pf = self.tensor2freq(pred)
        tf = self.tensor2freq(target)
        if self.ave_spectrum:
            pf = torch.mean(pf, dim=0, keepdim=True)
            tf = torch.mean(tf, dim=0, keepdim=True)
        return self.loss_weight * self.loss_formulation(pf, tf, matrix)


def gradient_magnitude_consistency_loss(pred, target, mask):
    def grad_mag(x):
        dx = F.pad(x[:,:,:,1:] - x[:,:,:,:-1], (0,1,0,0))
        dy = F.pad(x[:,:,1:,:] - x[:,:,:-1,:], (0,0,0,1))
        return torch.sqrt(dx**2 + dy**2 + 1e-8)

    gp = grad_mag(pred)
    gt = grad_mag(target)

    mask_bool = mask.bool()
    if mask_bool.sum() == 0:
        return pred.new_tensor(0.0)

    mean_loss = F.l1_loss(gp[mask_bool], gt[mask_bool])
    var_loss  = torch.abs(gp[mask_bool].var() - gt[mask_bool].var())
    return mean_loss + 0.5 * var_loss


def lsgan_generator_loss(fake_pred):
    return torch.mean((fake_pred - 1.0) ** 2)

def lsgan_discriminator_loss(real_pred, fake_pred):
    return 0.5 * (torch.mean((real_pred - 1.0) ** 2) + torch.mean(fake_pred ** 2))


perceptual_model = None
roi_encoder      = None
focal_freq_loss  = None


# ======================================================
# EDM LOSS — EXP-806
# ------------------------------------------------------
# vs EXP-805:
#   FFL alpha    : 2.0 → 1.5   [stabilitas ↑]
#   FFL weight   : 0.25         [tidak berubah]
#   edge         : 0.20         [tidak berubah]
#   laplacian    : 0.20         [tidak berubah]
#   grad_mag     : 0.15         [tidak berubah]
#   VGG          : 0.25         [tidak berubah]
#   pixel losses : tidak berubah
#   GAN          : off          [tidak berubah]
# Perubahan utama ada di model (GroupNorm) dan
# trainer (CosineAnnealingLR + grad clipping).
# ======================================================
class EDMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_tuple, target, mask, input_tensor):
        global perceptual_model, focal_freq_loss

        pred, base, enh = pred_tuple
        mask_f = mask.float()

        t1   = input_tensor[:, 1:2, :, :]
        diff = target - t1

        healthy = (diff <= 0.025) & mask
        tumor   = (diff >  0.025) & mask

        loss_global = torch.mean(torch.abs(pred - target)[mask])
        loss_tumor  = (torch.mean(torch.abs(pred - target)[tumor])
                       if tumor.any() else pred.new_tensor(0.0))
        loss_base   = (torch.mean(torch.abs(base - target)[healthy])
                       if healthy.any() else pred.new_tensor(0.0))
        enh_gt      = torch.clamp(target - base.detach(), min=0.0)
        loss_enh    = (torch.mean(torch.abs(enh - enh_gt)[tumor])
                       if tumor.any()
                       else torch.mean(torch.abs(enh - enh_gt)[mask]))

        loss_edge   = edge_loss(pred, target, mask)
        loss_lap    = laplacian_loss(pred, target, mask)
        loss_msssim = ms_ssim_loss(pred, target, mask)

        if perceptual_model is None:
            perceptual_model = VGGPerceptual().to(pred.device).eval()
        pred_clip   = torch.clamp(pred,   0.0, 1.0)
        target_clip = torch.clamp(target, 0.0, 1.0)
        loss_vgg = perceptual_model(
            (pred_clip   * mask_f).float(),
            (target_clip * mask_f).float()
        )

        loss_roi = roi_contrastive_loss(pred, target, tumor)

        if focal_freq_loss is None:
            focal_freq_loss = FocalFrequencyLoss(
                loss_weight=1.0, alpha=1.5,
                patch_factor=1, ave_spectrum=False,
                log_matrix=False, batch_matrix=False,
            ).to(pred.device)
        pred_masked   = torch.clamp(pred_clip   * mask_f, 0.0, 1.0)
        target_masked = torch.clamp(target_clip * mask_f, 0.0, 1.0)
        loss_freq = focal_freq_loss(pred_masked, target_masked)

        loss_grad_mag = gradient_magnitude_consistency_loss(
            pred_clip, target_clip, mask
        )

        total = (
            0.21 * loss_global  +
            0.90 * loss_tumor   +
            0.60 * loss_base    +
            0.72 * loss_enh     +
            0.20 * loss_edge    +
            0.20 * loss_lap     +
            0.25 * loss_vgg     +
            0.15 * loss_msssim  +
            0.12 * loss_roi     +
            0.25 * loss_freq    +
            0.15 * loss_grad_mag
        )

        return total


LOSS_REGISTRY = {
    "EDMLoss": EDMLoss()
}
