import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import ms_ssim


# ======================================================
# PRIMITIVE LOSSES
# ======================================================

def masked_l1(pred, target, mask):
    return F.l1_loss(pred[mask], target[mask])


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
    with torch.amp.autocast('cuda', enabled=False):
        val = ms_ssim(
            pred_m.float(), target_m.float(),
            data_range=1.0, size_average=True
        )
    if not torch.isfinite(val):
        return pred.new_tensor(0.0)
    return 1.0 - val


# ======================================================
# [NEW EXP-813] STRATIFIED ENHANCEMENT LOSS
# ------------------------------------------------------
# Masalah EXP-812:
#   - Threshold flat diff > 0.025 → semua pixel enhancement
#     diperlakukan sama, tidak peduli seberapa kuat
#   - Model gagal mensintesis focal bright enhancement
#     (PSNR Enh hanya 17.1, SSIM Enh 0.581)
#
# Fix: tiga tier berdasarkan kekuatan enhancement
#   weak       (0.025 < diff ≤ 0.10)  → bobot 1.0
#   strong     (0.10  < diff ≤ 0.25)  → bobot 3.0
#   very_strong (diff > 0.25)          → bobot 5.0
#
# Pixel dengan enhancement kuat mendapat gradient signal
# 3–5x lebih besar, memaksa model fokus ke area tersebut.
# ======================================================
def stratified_enhancement_loss(pred, target, diff, mask, enh_cfg):
    weak_thr        = enh_cfg["weak_thr"]
    strong_thr      = enh_cfg["strong_thr"]
    very_strong_thr = enh_cfg["very_strong_thr"]
    w_weak          = enh_cfg["w_weak"]
    w_strong        = enh_cfg["w_strong"]
    w_very_strong   = enh_cfg["w_very_strong"]

    tier_weak        = (diff >  weak_thr)        & (diff <= strong_thr)      & mask
    tier_strong      = (diff >  strong_thr)      & (diff <= very_strong_thr) & mask
    tier_very_strong = (diff >  very_strong_thr) & mask

    err = torch.abs(pred - target)

    total_loss   = pred.new_tensor(0.0)
    total_weight = 0.0

    if tier_weak.any():
        total_loss   = total_loss + w_weak * err[tier_weak].mean()
        total_weight += w_weak

    if tier_strong.any():
        total_loss   = total_loss + w_strong * err[tier_strong].mean()
        total_weight += w_strong

    if tier_very_strong.any():
        total_loss   = total_loss + w_very_strong * err[tier_very_strong].mean()
        total_weight += w_very_strong

    if total_weight == 0.0:
        return pred.new_tensor(0.0)

    loss = total_loss / total_weight

    if not torch.isfinite(loss):
        return pred.new_tensor(0.0)

    return loss


# ======================================================
# [NEW EXP-813] LOCAL SHARPNESS LOSS
# ------------------------------------------------------
# Masalah EXP-812:
#   - Sharpness ratio = 0.595 → pred hanya 59% setajam GT
#   - L1/L2 loss mendorong output ke rata-rata distribusi
#     → gambar yang smooth tapi blur
#
# Fix: Local Contrast Sharpness Loss
#   Mengukur perbedaan local contrast (std per patch)
#   antara pred dan GT di ROI.
#   - Lebih stabil dari TV loss (tidak menghukum tekstur)
#   - Mendorong model mempertahankan local variance
#     yang mencerminkan detail anatomi dan enhancement
#   - Pure tensor ops, tidak ada model tambahan
# ======================================================
def local_sharpness_loss(pred, target, mask, patch_size=8):
    B, C, H, W = pred.shape

    # Pastikan dimensi bisa dibagi patch_size
    H_crop = (H // patch_size) * patch_size
    W_crop = (W // patch_size) * patch_size
    pred_c   = pred[..., :H_crop, :W_crop]
    target_c = target[..., :H_crop, :W_crop]
    mask_c   = mask[..., :H_crop, :W_crop]

    # Unfold jadi patches: [B, C, nH, nW, p, p]
    def to_patches(x):
        # [B*C, 1, H, W] → unfold → [B*C, p*p, nH*nW]
        x_bc = x.reshape(B * C, 1, H_crop, W_crop)
        patches = F.unfold(x_bc, kernel_size=patch_size, stride=patch_size)
        # patches: [B*C, p*p, nH*nW]
        return patches.reshape(B, C, patch_size * patch_size, -1)

    # Mask: ambil patch yang mayoritas di dalam ROI (>50%)
    mask_f   = mask_c.float()
    mask_bc  = mask_f.reshape(B * C, 1, H_crop, W_crop)
    m_patches = F.unfold(mask_bc, kernel_size=patch_size, stride=patch_size)
    m_patches = m_patches.reshape(B, C, patch_size * patch_size, -1)
    valid_patches = (m_patches.mean(dim=2) > 0.5)  # [B, C, nPatches]

    if not valid_patches.any():
        return pred.new_tensor(0.0)

    pred_patches   = to_patches(pred_c)    # [B, C, p*p, nPatches]
    target_patches = to_patches(target_c)

    # Local std per patch sebagai proxy sharpness
    pred_std   = pred_patches.std(dim=2)    # [B, C, nPatches]
    target_std = target_patches.std(dim=2)

    # Hanya hitung di patch yang valid (dalam ROI)
    loss = torch.abs(pred_std[valid_patches] - target_std[valid_patches]).mean()

    if not torch.isfinite(loss):
        return pred.new_tensor(0.0)

    return loss


# ======================================================
# TUMOR INTENSITY CONSISTENCY (dari EXP-812)
# ======================================================
def tumor_intensity_consistency_loss(pred, target, tumor_mask):
    if not tumor_mask.any() or tumor_mask.sum() < 10:
        return pred.new_tensor(0.0)

    mask_f = tumor_mask.float()
    pred_roi   = pred   * mask_f
    target_roi = target * mask_f
    n = mask_f.sum().clamp(min=1.0)

    pred_mean   = pred_roi.sum()   / n
    target_mean = target_roi.sum() / n
    loss_mean   = torch.abs(pred_mean - target_mean)

    pred_std   = torch.sqrt(((pred_roi   - pred_mean)   ** 2 * mask_f).sum() / n + 1e-8)
    target_std = torch.sqrt(((target_roi - target_mean) ** 2 * mask_f).sum() / n + 1e-8)
    loss_std   = torch.abs(pred_std - target_std)

    loss = loss_mean + 0.5 * loss_std

    if not torch.isfinite(loss):
        return pred.new_tensor(0.0)

    return loss


# ======================================================
# VGG PERCEPTUAL (identik EXP-812)
# ======================================================
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


# ======================================================
# FOCAL FREQUENCY LOSS (identik EXP-812)
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
        w = torch.clamp(w, min=0.0, max=1e4)
        if self.log_matrix:
            w = torch.log(w + 1.0)
        w = w ** self.alpha
        w = torch.clamp(w, min=0.0, max=1e6)
        if self.batch_matrix:
            w = w / (w.sum() + 1e-6)
        else:
            B     = w.shape[0]
            w_sum = (w.reshape(B, -1)
                      .sum(1, keepdim=True)
                      .reshape(B, *([1] * (w.dim() - 1))))
            w = w / (w_sum + 1e-6)
        with torch.amp.autocast('cuda', enabled=False):
            freq_dist_f = torch.clamp(freq_distance.float(), max=1e6)
            loss = torch.mean(w.float() * freq_dist_f)
        if not torch.isfinite(loss):
            return recon_freq.new_tensor(0.0)
        return loss

    def forward(self, pred, target, matrix=None):
        pf = self.tensor2freq(pred)
        tf = self.tensor2freq(target)
        if self.ave_spectrum:
            pf = torch.mean(pf, dim=0, keepdim=True)
            tf = torch.mean(tf, dim=0, keepdim=True)
        result = self.loss_weight * self.loss_formulation(pf, tf, matrix)
        if not torch.isfinite(result):
            return pred.new_tensor(0.0)
        return result


# ======================================================
# GRADIENT MAGNITUDE CONSISTENCY (identik EXP-812)
# ======================================================
def gradient_magnitude_consistency_loss(pred, target, mask):
    def grad_mag(x):
        dx = F.pad(x[:,:,:,1:] - x[:,:,:,:-1], (0,1,0,0))
        dy = F.pad(x[:,:,1:,:] - x[:,:,:-1,:], (0,0,0,1))
        return torch.sqrt(dx**2 + dy**2 + 1e-8)
    gp        = grad_mag(pred)
    gt        = grad_mag(target)
    mask_bool = mask.bool()
    if mask_bool.sum() == 0:
        return pred.new_tensor(0.0)
    mean_loss = F.l1_loss(gp[mask_bool], gt[mask_bool])
    var_loss  = torch.abs(gp[mask_bool].var() - gt[mask_bool].var())
    return mean_loss + 0.5 * var_loss


# ======================================================
# GAN LOSSES
# ======================================================
def lsgan_generator_loss(fake_pred):
    return torch.mean((fake_pred - 1.0) ** 2)

def lsgan_discriminator_loss(real_pred, fake_pred):
    return 0.5 * (torch.mean((real_pred - 1.0) ** 2) + torch.mean(fake_pred ** 2))


# ======================================================
# GLOBAL SINGLETONS
# ======================================================
perceptual_model = None
focal_freq_loss  = None


# ======================================================
# EDM LOSS — EXP-813
# [CHANGED vs EXP-812]:
#
#   1. loss_tumor (flat diff>0.025) → DIGANTI
#      stratified_enhancement_loss (3 tier: weak/strong/very_strong)
#      Target: angkat PSNR Enh dari 17.1 → ≥20 dB
#
#   2. loss_enh (aux head enh_gt) → TETAP
#      Sinyal tambahan dari auxiliary enhancement head
#
#   3. local_sharpness_loss DITAMBAHKAN (bobot 0.30)
#      Target: angkat sharpness ratio dari 0.595 → ≥0.75
#
#   4. loss_global, loss_base, loss_tic, loss_freq,
#      loss_edge, loss_lap, loss_msssim, loss_vgg,
#      loss_grad_mag → identik dengan EXP-812
#
# Semua bobot di-rebalance untuk menjaga total loss
# dalam range yang serupa dengan EXP-812 (~0.7–1.2).
# ======================================================
class EDMLoss(nn.Module):
    def __init__(self, enh_cfg, sharpness_cfg):
        super().__init__()
        self.enh_cfg       = enh_cfg
        self.sharpness_cfg = sharpness_cfg

    def _safe(self, val, name=""):
        if not torch.isfinite(val):
            return val.new_tensor(0.0)
        return val

    def forward(self, pred_tuple, target, mask, input_tensor):
        global perceptual_model, focal_freq_loss

        pred, base, enh = pred_tuple
        mask_f = mask.float()

        t1   = input_tensor[:, 1:2, :, :]
        diff = target - t1

        healthy = (diff <= self.enh_cfg["weak_thr"]) & mask

        # --- global pixel loss ---
        loss_global = self._safe(
            torch.mean(torch.abs(pred - target)[mask]), "global"
        )

        # --- [NEW] stratified enhancement loss ---
        # Menggantikan loss_tumor yang flat
        loss_strat_enh = self._safe(
            stratified_enhancement_loss(pred, target, diff, mask, self.enh_cfg),
            "strat_enh"
        )

        # --- base (healthy region) ---
        loss_base = (
            self._safe(torch.mean(torch.abs(base - target)[healthy]), "base")
            if healthy.any() else pred.new_tensor(0.0)
        )

        # --- auxiliary enhancement head ---
        any_tumor = (diff > self.enh_cfg["weak_thr"]) & mask
        enh_gt    = torch.clamp(target - base.detach(), min=0.0)
        loss_enh  = (
            self._safe(torch.mean(torch.abs(enh - enh_gt)[any_tumor]), "enh")
            if any_tumor.any()
            else self._safe(torch.mean(torch.abs(enh - enh_gt)[mask]), "enh")
        )

        # --- structural ---
        loss_edge   = self._safe(edge_loss(pred, target, mask),      "edge")
        loss_lap    = self._safe(laplacian_loss(pred, target, mask), "lap")
        loss_msssim = self._safe(ms_ssim_loss(pred, target, mask),   "msssim")

        # --- perceptual ---
        if perceptual_model is None:
            perceptual_model = VGGPerceptual().to(pred.device).eval()
        pred_clip   = torch.clamp(pred,   0.0, 1.0)
        target_clip = torch.clamp(target, 0.0, 1.0)
        loss_vgg = self._safe(
            perceptual_model(
                (pred_clip   * mask_f).float(),
                (target_clip * mask_f).float()
            ), "vgg"
        )

        # --- tumor intensity consistency ---
        tumor_mask = (diff > self.enh_cfg["weak_thr"]) & mask
        loss_tic = self._safe(
            tumor_intensity_consistency_loss(pred_clip, target_clip, tumor_mask),
            "tic"
        )

        # --- [NEW] local sharpness ---
        if self.sharpness_cfg["use_sharpness"]:
            loss_sharp = self._safe(
                local_sharpness_loss(
                    pred_clip, target_clip, mask,
                    patch_size=self.sharpness_cfg["patch_size"]
                ), "sharp"
            )
        else:
            loss_sharp = pred.new_tensor(0.0)

        # --- frequency ---
        if focal_freq_loss is None:
            focal_freq_loss = FocalFrequencyLoss(
                loss_weight=1.0, alpha=1.5, patch_factor=1,
                ave_spectrum=False, log_matrix=False, batch_matrix=False,
            ).to(pred.device)
        pred_masked   = torch.clamp(pred_clip * mask_f, 0.0, 1.0)
        target_masked = torch.clamp(target_clip * mask_f, 0.0, 1.0)
        loss_freq = self._safe(
            focal_freq_loss(pred_masked, target_masked), "freq"
        )

        # --- gradient magnitude ---
        loss_grad_mag = self._safe(
            gradient_magnitude_consistency_loss(pred_clip, target_clip, mask),
            "grad_mag"
        )

        # ======================================================
        # TOTAL LOSS — EXP-813
        # Perubahan bobot vs EXP-812:
        #   loss_tumor (0.90) → loss_strat_enh (1.20)
        #     dinaikkan karena sekarang sudah ternormalisasi
        #     per tier (bukan raw sum), jadi skala lebih stabil
        #   local_sharpness: ditambahkan (0.30)
        #   semua bobot lain tetap
        # ======================================================
        total = (
            0.21 * loss_global    +
            1.20 * loss_strat_enh +   # [NEW] ganti loss_tumor 0.90
            0.60 * loss_base      +
            0.72 * loss_enh       +
            0.20 * loss_edge      +
            0.20 * loss_lap       +
            0.25 * loss_vgg       +
            0.15 * loss_msssim    +
            0.18 * loss_tic       +
            0.25 * loss_freq      +
            0.15 * loss_grad_mag  +
            self.sharpness_cfg["weight"] * loss_sharp   # [NEW] 0.30
        )

        if not torch.isfinite(total):
            return loss_global if torch.isfinite(loss_global) else pred.new_tensor(0.0)

        return total


# ======================================================
# FACTORY — diinstansiasi dengan config dari CONFIG
# ======================================================
def build_loss(config):
    return EDMLoss(
        enh_cfg       = config["enhancement"],
        sharpness_cfg = config["sharpness"],
    )


LOSS_REGISTRY = {
    "EDMLoss": None   # diisi oleh trainer setelah CONFIG diload
}
