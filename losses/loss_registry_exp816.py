import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import ms_ssim


# ======================================================
# PRIMITIVE LOSSES — identik EXP-815
# ======================================================

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
    total_loss, total_weight = pred.new_tensor(0.0), 0.0

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
    return loss if torch.isfinite(loss) else pred.new_tensor(0.0)


def enhancement_seg_loss(enh_logits, target, input_tensor, enh_threshold, roi_mask):
    t1      = input_tensor[:, 1:2, :, :]
    diff    = target - t1
    gt_mask = (diff > enh_threshold).float()

    H_l, W_l = enh_logits.shape[-2:]
    if gt_mask.shape[-2:] != (H_l, W_l):
        gt_mask  = F.interpolate(gt_mask,         size=(H_l, W_l), mode="nearest")
        roi_down = F.interpolate(roi_mask.float(), size=(H_l, W_l), mode="nearest")
    else:
        roi_down = roi_mask.float()

    roi_bool = roi_down.bool()
    if not roi_bool.any():
        return enh_logits.new_tensor(0.0)

    logits_roi = enh_logits[roi_bool]
    gt_roi     = gt_mask[roi_bool]
    n_pos      = gt_roi.sum().clamp(min=1.0)
    n_neg      = (1.0 - gt_roi).sum().clamp(min=1.0)
    pos_weight = torch.clamp(n_neg / n_pos, max=20.0)

    loss = F.binary_cross_entropy_with_logits(
        logits_roi, gt_roi, pos_weight=pos_weight.detach()
    )
    return loss if torch.isfinite(loss) else enh_logits.new_tensor(0.0)


def local_sharpness_loss(pred, target, mask, patch_size=8):
    B, C, H, W = pred.shape
    H_crop = (H // patch_size) * patch_size
    W_crop = (W // patch_size) * patch_size
    pred_c, target_c, mask_c = (pred[..., :H_crop, :W_crop],
                                 target[..., :H_crop, :W_crop],
                                 mask[..., :H_crop, :W_crop])

    def to_patches(x):
        return F.unfold(
            x.reshape(B * C, 1, H_crop, W_crop),
            kernel_size=patch_size, stride=patch_size
        ).reshape(B, C, patch_size * patch_size, -1)

    mask_f    = mask_c.float()
    m_patches = to_patches(mask_f)
    valid     = (m_patches.mean(dim=2) > 0.5)
    if not valid.any():
        return pred.new_tensor(0.0)

    loss = torch.abs(
        to_patches(pred_c).std(dim=2)[valid] -
        to_patches(target_c).std(dim=2)[valid]
    ).mean()
    return loss if torch.isfinite(loss) else pred.new_tensor(0.0)


def tumor_intensity_consistency_loss(pred, target, tumor_mask):
    if not tumor_mask.any() or tumor_mask.sum() < 10:
        return pred.new_tensor(0.0)
    mask_f    = tumor_mask.float()
    n         = mask_f.sum().clamp(min=1.0)
    pm        = (pred   * mask_f).sum() / n
    tm        = (target * mask_f).sum() / n
    loss_mean = torch.abs(pm - tm)
    ps        = torch.sqrt(((pred   - pm) ** 2 * mask_f).sum() / n + 1e-8)
    ts        = torch.sqrt(((target - tm) ** 2 * mask_f).sum() / n + 1e-8)
    loss      = loss_mean + 0.5 * torch.abs(ps - ts)
    return loss if torch.isfinite(loss) else pred.new_tensor(0.0)


def gradient_magnitude_consistency_loss(pred, target, mask):
    def grad_mag(x):
        dx = F.pad(x[:,:,:,1:] - x[:,:,:,:-1], (0,1,0,0))
        dy = F.pad(x[:,:,1:,:] - x[:,:,:-1,:], (0,0,0,1))
        return torch.sqrt(dx**2 + dy**2 + 1e-8)
    gp, gt = grad_mag(pred), grad_mag(target)
    mb     = mask.bool()
    if mb.sum() == 0:
        return pred.new_tensor(0.0)
    loss = F.l1_loss(gp[mb], gt[mb]) + 0.5 * torch.abs(gp[mb].var() - gt[mb].var())
    return loss if torch.isfinite(loss) else pred.new_tensor(0.0)


# ======================================================
# [NEW EXP-816] INTENSITY CALIBRATION LOSS
# ------------------------------------------------------
# Masalah EXP-815: calibration_bias = -0.048
#   Model secara sistematis under-predicts intensitas ~5%
#   di seluruh ROI. Penyebabnya EnhancementGate menekan
#   skala feature map di region non-enhancement secara
#   tidak langsung.
#
# Fix: Intensity Calibration Loss
#   Mengukur perbedaan mean intensitas antara pred dan
#   target di dalam ROI secara global (bukan hanya tumor).
#   Jika pred terlalu gelap → loss besar → model koreksi.
#   Jika pred sudah terkalibrasi → loss ≈ 0.
#
#   Tambahan: variance calibration — pred harus memiliki
#   distribusi intensitas yang serupa dengan GT (variance
#   ratio EXP-815 turun ke 0.871, target ≥ 0.92).
#
#   Pure tensor ops, tidak ada overhead komputasi.
# ======================================================
def intensity_calibration_loss(pred, target, mask):
    """
    Mengukur perbedaan mean dan std intensitas global
    antara pred dan target di seluruh ROI.
    Mencegah systematic under/over-prediction.
    """
    mask_f = mask.float()
    n      = mask_f.sum().clamp(min=1.0)

    pred_roi   = pred   * mask_f
    target_roi = target * mask_f

    # Mean calibration
    pred_mean   = pred_roi.sum()   / n
    target_mean = target_roi.sum() / n
    loss_mean   = torch.abs(pred_mean - target_mean)

    # Variance calibration — pastikan distribusi intensitas serupa
    pred_var   = ((pred_roi   - pred_mean)   ** 2 * mask_f).sum() / n
    target_var = ((target_roi - target_mean) ** 2 * mask_f).sum() / n
    pred_std   = torch.sqrt(pred_var   + 1e-8)
    target_std = torch.sqrt(target_var + 1e-8)
    loss_var   = torch.abs(pred_std - target_std)

    loss = loss_mean + 0.5 * loss_var

    if not torch.isfinite(loss):
        return pred.new_tensor(0.0)
    return loss


# ======================================================
# VGG PERCEPTUAL — identik EXP-815
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
        pred   = pred.float().repeat(1, 3, 1, 1)
        target = target.float().repeat(1, 3, 1, 1)
        with torch.amp.autocast('cuda', enabled=False):
            p1 = self.s1(pred);   t1 = self.s1(target)
            p2 = self.s2(p1);     t2 = self.s2(t1)
            p3 = self.s3(p2);     t3 = self.s3(t2)
        loss = (F.l1_loss(p1, t1) + F.l1_loss(p2, t2) + F.l1_loss(p3, t3)) / 3.0
        return loss if torch.isfinite(loss) else pred.new_tensor(0.0)


# ======================================================
# FOCAL FREQUENCY LOSS — identik EXP-815
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
        w = torch.clamp(w, 0.0, 1e4) ** self.alpha
        w = torch.clamp(w, 0.0, 1e6)
        if self.batch_matrix:
            w = w / (w.sum() + 1e-6)
        else:
            B     = w.shape[0]
            w_sum = (w.reshape(B, -1).sum(1, keepdim=True)
                      .reshape(B, *([1] * (w.dim() - 1))))
            w = w / (w_sum + 1e-6)
        with torch.amp.autocast('cuda', enabled=False):
            loss = torch.mean(w.float() *
                              torch.clamp(freq_distance.float(), max=1e6))
        return loss if torch.isfinite(loss) else recon_freq.new_tensor(0.0)

    def forward(self, pred, target, matrix=None):
        pf = self.tensor2freq(pred)
        tf = self.tensor2freq(target)
        if self.ave_spectrum:
            pf = torch.mean(pf, dim=0, keepdim=True)
            tf = torch.mean(tf, dim=0, keepdim=True)
        r = self.loss_weight * self.loss_formulation(pf, tf, matrix)
        return r if torch.isfinite(r) else pred.new_tensor(0.0)


def lsgan_generator_loss(fake_pred):
    return torch.mean((fake_pred - 1.0) ** 2)

def lsgan_discriminator_loss(real_pred, fake_pred):
    return 0.5 * (torch.mean((real_pred - 1.0) ** 2) + torch.mean(fake_pred ** 2))


# ======================================================
# EDM LOSS — EXP-816
# [CHANGED vs EXP-815]:
#
#   Fix 1: seg_loss_weight 1.50 → 0.80
#     BCE terlalu dominan di EXP-814/815 → rekonstruksi
#     global tertahan. Dikurangi 47% untuk rebalancing.
#
#   Fix 2: gate_weight 0.70 → 0.35 (di model/config)
#     EnhancementGate terlalu agresif → calibration bias
#     -0.048. Gate lebih lemah = spatial guidance tanpa
#     distorsi skala intensitas.
#
#   Fix 3: intensity_calibration_loss DITAMBAHKAN (0.40)
#     Secara eksplisit menghukum perbedaan mean dan std
#     intensitas global di ROI. Target: bias mendekati 0
#     dan variance ratio ≥ 0.92.
#
#   Semua loss lain identik EXP-815.
# ======================================================
class EDMLoss(nn.Module):
    def __init__(self, enh_cfg, sharpness_cfg, seg_cfg, cal_cfg):
        super().__init__()
        self.enh_cfg       = enh_cfg
        self.sharpness_cfg = sharpness_cfg
        self.seg_cfg       = seg_cfg
        self.cal_cfg       = cal_cfg
        self.perceptual    = None
        self.freq_loss     = None

    def inject_models(self, perceptual: VGGPerceptual,
                      freq_loss: FocalFrequencyLoss):
        self.perceptual = perceptual
        self.freq_loss  = freq_loss

    def _safe(self, val, name=""):
        return val.new_tensor(0.0) if not torch.isfinite(val) else val

    def forward(self, pred_tuple, target, mask, input_tensor):
        pred, enh_logits, base, _ = pred_tuple
        mask_f = mask.float()

        t1   = input_tensor[:, 1:2, :, :]
        diff = target - t1

        healthy   = (diff <= self.enh_cfg["weak_thr"]) & mask
        any_tumor = (diff >  self.enh_cfg["weak_thr"]) & mask

        # --- global pixel ---
        loss_global = self._safe(
            torch.mean(torch.abs(pred - target)[mask]), "global"
        )

        # --- stratified enhancement ---
        loss_strat_enh = self._safe(
            stratified_enhancement_loss(pred, target, diff, mask, self.enh_cfg),
            "strat_enh"
        )

        # --- base healthy region ---
        loss_base = (
            self._safe(torch.mean(torch.abs(base - target)[healthy]), "base")
            if healthy.any() else pred.new_tensor(0.0)
        )

        # --- aux enh head ---
        enh_gt   = torch.clamp(target - base.detach(), min=0.0)
        loss_enh = (
            self._safe(torch.mean(torch.abs(pred - enh_gt)[any_tumor]), "enh")
            if any_tumor.any()
            else self._safe(torch.mean(torch.abs(pred - enh_gt)[mask]), "enh")
        )

        pred_clip   = torch.clamp(pred,   0.0, 1.0)
        target_clip = torch.clamp(target, 0.0, 1.0)

        # --- structural ---
        loss_edge   = self._safe(edge_loss(pred, target, mask),      "edge")
        loss_lap    = self._safe(laplacian_loss(pred, target, mask), "lap")
        loss_msssim = self._safe(ms_ssim_loss(pred, target, mask),   "msssim")

        # --- perceptual ---
        loss_vgg = self._safe(
            self.perceptual(pred_clip * mask_f, target_clip * mask_f), "vgg"
        )

        # --- tumor intensity consistency ---
        loss_tic = self._safe(
            tumor_intensity_consistency_loss(pred_clip, target_clip, any_tumor),
            "tic"
        )

        # --- sharpness ---
        loss_sharp = (
            self._safe(
                local_sharpness_loss(pred_clip, target_clip, mask,
                                     self.sharpness_cfg["patch_size"]),
                "sharp"
            ) if self.sharpness_cfg["use_sharpness"]
            else pred.new_tensor(0.0)
        )

        # --- frequency ---
        pred_m   = torch.clamp(pred_clip * mask_f, 0.0, 1.0)
        target_m = torch.clamp(target_clip * mask_f, 0.0, 1.0)
        loss_freq = self._safe(self.freq_loss(pred_m, target_m), "freq")

        # --- gradient magnitude ---
        loss_grad_mag = self._safe(
            gradient_magnitude_consistency_loss(pred_clip, target_clip, mask),
            "grad_mag"
        )

        # --- segmentation BCE ---
        roi_mask_1ch = mask_f[:, :1, :, :] if mask_f.dim() == 4 else mask_f.unsqueeze(1)
        loss_seg = (
            self._safe(
                enhancement_seg_loss(
                    enh_logits, target, input_tensor,
                    self.seg_cfg["enh_threshold"], roi_mask_1ch
                ), "seg"
            ) if self.seg_cfg["use_enh_seg"]
            else pred.new_tensor(0.0)
        )

        # --- [NEW] intensity calibration ---
        loss_cal = (
            self._safe(
                intensity_calibration_loss(pred_clip, target_clip, mask),
                "cal"
            ) if self.cal_cfg["use_calibration"]
            else pred.new_tensor(0.0)
        )

        # ======================================================
        # TOTAL LOSS — EXP-816
        # Perubahan vs EXP-815:
        #   seg weight  : 1.50 → 0.80  [Fix 1]
        #   cal weight  : +0.40 baru    [Fix 3]
        #   (gate_weight diatur di config/model, bukan di sini)
        # ======================================================
        total = (
            0.21 * loss_global    +
            1.20 * loss_strat_enh +
            0.60 * loss_base      +
            0.72 * loss_enh       +
            0.20 * loss_edge      +
            0.20 * loss_lap       +
            0.25 * loss_vgg       +
            0.15 * loss_msssim    +
            0.18 * loss_tic       +
            self.sharpness_cfg["weight"] * loss_sharp   +   # 0.30
            0.25 * loss_freq      +
            0.15 * loss_grad_mag  +
            self.seg_cfg["seg_loss_weight"] * loss_seg  +   # [Fix 1] 0.80
            self.cal_cfg["weight"] * loss_cal               # [Fix 3] 0.40
        )

        if not torch.isfinite(total):
            return loss_global if torch.isfinite(loss_global) else pred.new_tensor(0.0)

        return total


def build_loss(config):
    return EDMLoss(
        enh_cfg       = config["enhancement"],
        sharpness_cfg = config["sharpness"],
        seg_cfg       = config["enh_seg"],
        cal_cfg       = config["calibration"],
    )


LOSS_REGISTRY = {"EDMLoss": None}
