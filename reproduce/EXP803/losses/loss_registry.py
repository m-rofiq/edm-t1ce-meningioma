import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import ssim, ms_ssim


# ==================================================
# BASIC
# ==================================================
def masked_l1(pred, target, mask):
    return F.l1_loss(pred[mask], target[mask])


def masked_mse(pred, target, mask):
    return F.mse_loss(pred[mask], target[mask])


# ==================================================
# EDGE
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

    return F.l1_loss(pe[mask], te[mask])


# ==================================================
# LAPLACIAN
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

    return F.l1_loss(p[mask], t[mask])


# ==================================================
# SSIM FAMILY
# ==================================================
def ssim_loss(pred, target, mask):
    pred_m = torch.clamp(pred * mask.float(), 0.0, 1.0)
    target_m = torch.clamp(target * mask.float(), 0.0, 1.0)

    val = ssim(
        pred_m,
        target_m,
        data_range=1.0,
        size_average=True
    )
    return 1.0 - val


def ms_ssim_loss(pred, target, mask):
    pred_m = torch.clamp(pred * mask.float(), 0.0, 1.0)
    target_m = torch.clamp(target * mask.float(), 0.0, 1.0)

    val = ms_ssim(
        pred_m,
        target_m,
        data_range=1.0,
        size_average=True
    )
    return 1.0 - val


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
# ROI CONTRASTIVE
# ==================================================
class ROIContrastEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        net = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        self.features = nn.Sequential(*list(net.children())[:6])

        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        return self.features(x)


def roi_contrastive_loss(pred, target, tumor_mask):
    global roi_encoder

    if not tumor_mask.any():
        return pred.new_tensor(0.0)

    if roi_encoder is None:
        roi_encoder = ROIContrastEncoder().to(pred.device).eval()

    pred = torch.clamp(pred, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)

    pred_roi = pred * tumor_mask.float()
    target_roi = target * tumor_mask.float()

    fp = roi_encoder(pred_roi)
    ft = roi_encoder(target_roi)

    return F.l1_loss(fp, ft)


# ==================================================
# [NEW] FOCAL FREQUENCY LOSS
# --------------------------------------------------
# Jiang et al., "Focal Frequency Loss for Image
# Reconstruction and Synthesis", ICCV 2021.
#
# Ide utama: berikan bobot lebih pada frekuensi
# tinggi yang sulit dipelajari oleh model karena
# pixel loss mendorong mean prediction (= blur).
# Weight matrix W_ij dihitung dari selisih spektral
# iterasi sebelumnya sehingga loss secara adaptif
# memfokuskan diri pada komponen yang paling jauh
# dari target — itulah sumber blur.
# ==================================================
class FocalFrequencyLoss(nn.Module):
    """
    Focal Frequency Loss.

    Args:
        loss_weight (float): bobot loss ini dalam total loss.
        alpha (float): eksponen focal weighting. Default=1.0.
            Nilai lebih tinggi → penekanan lebih kuat pada
            frekuensi yang sulit.
        patch_factor (int): membagi gambar jadi patch untuk
            estimasi frekuensi lokal. Default=1 (global).
        ave_spectrum (bool): rata-rata spektrum across batch.
        log_matrix (bool): log-transform weight matrix untuk
            stabilitas numerik pada rentang dinamis lebar.
        batch_matrix (bool): hitung weight per-batch (True)
            atau per-sample (False).
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        alpha: float = 1.0,
        patch_factor: int = 1,
        ave_spectrum: bool = False,
        log_matrix: bool = False,
        batch_matrix: bool = False,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x: torch.Tensor) -> torch.Tensor:
        """FFT2D → stack [real, imag] di dim channel."""
        # patch decomposition
        patch_factor = self.patch_factor
        _, _, H, W = x.shape
        assert H % patch_factor == 0 and W % patch_factor == 0, (
            f"Spatial dims ({H}×{W}) harus habis dibagi patch_factor={patch_factor}"
        )
        patch_list = []
        patch_H = H // patch_factor
        patch_W = W // patch_factor

        for i in range(patch_factor):
            for j in range(patch_factor):
                patch = x[..., i * patch_H:(i + 1) * patch_H,
                                 j * patch_W:(j + 1) * patch_W]
                freq = torch.fft.fft2(patch, norm='ortho')
                # [B, C, pH, pW] → [B, C, pH, pW, 2]
                freq = torch.stack([freq.real, freq.imag], dim=-1)
                patch_list.append(freq)

        # gabungkan semua patch: [B, C, H, W, 2]
        return torch.stack(patch_list, dim=1)  # [B, P*P, C, pH, pW, 2]

    def loss_formulation(
        self,
        recon_freq: torch.Tensor,
        real_freq: torch.Tensor,
        matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # selisih real dan imag
        tmp = (recon_freq - real_freq) ** 2       # [..., 2]
        freq_distance = tmp[..., 0] + tmp[..., 1] # [...] tanpa dim terakhir

        # weight matrix
        if matrix is not None:
            weight_matrix = matrix.detach()
        else:
            weight_matrix = freq_distance.clone().detach()

        if self.log_matrix:
            weight_matrix = torch.log(weight_matrix + 1.0)

        weight_matrix = weight_matrix ** self.alpha

        # normalisasi per-sample atau per-batch
        if self.batch_matrix:
            weight_matrix = weight_matrix / (weight_matrix.sum() + 1e-8)
        else:
            # normalisasi per-elemen batch
            B = weight_matrix.shape[0]
            weight_matrix = weight_matrix / (
                weight_matrix.reshape(B, -1).sum(dim=1, keepdim=True).reshape(
                    B, *([1] * (weight_matrix.dim() - 1))
                ) + 1e-8
            )

        with torch.amp.autocast('cuda', enabled=False):
            loss = torch.mean(
                weight_matrix.float() * freq_distance.float()
            )

        return loss

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            pred:   prediksi model  [B, C, H, W], range [0,1]
            target: ground truth    [B, C, H, W], range [0,1]
            matrix: optional pre-computed weight matrix (jarang dipakai)
        Returns:
            scalar loss × loss_weight
        """
        pred_freq   = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        if self.ave_spectrum:
            pred_freq   = torch.mean(pred_freq,   dim=0, keepdim=True)
            target_freq = torch.mean(target_freq, dim=0, keepdim=True)

        return self.loss_weight * self.loss_formulation(
            pred_freq, target_freq, matrix
        )


# ==================================================
# [NEW] LSGAN GENERATOR LOSS
# --------------------------------------------------
# Pindah dari hinge loss → LSGAN (Mao et al., 2017).
# LSGAN meminimasi least-squares distance antara
# output discriminator dan label target.
#
# Untuk generator: E[(D(G(x)) - 1)^2]
# → gradient tidak saturate meski D sangat kuat,
#   sehingga signal ke G tetap informatif.
#
# Dipanggil dari trainer, bukan dari EDMLoss,
# tapi didefinisikan di sini agar loss registry
# menjadi satu-satunya tempat semua loss.
# ==================================================
def lsgan_generator_loss(fake_pred: torch.Tensor) -> torch.Tensor:
    """
    Generator loss untuk LSGAN.
    fake_pred: output D(G(x)), shape bebas.
    Target label = 1 (generator ingin D mengira output nyata).
    """
    return torch.mean((fake_pred - 1.0) ** 2)


def lsgan_discriminator_loss(
    real_pred: torch.Tensor,
    fake_pred: torch.Tensor,
) -> torch.Tensor:
    """
    Discriminator loss untuk LSGAN.
    real_pred: D(x_real)
    fake_pred: D(G(x)).detach()
    """
    loss_real = torch.mean((real_pred - 1.0) ** 2)
    loss_fake = torch.mean(fake_pred ** 2)
    return 0.5 * (loss_real + loss_fake)


# ==================================================
# GLOBAL OBJECTS
# ==================================================
perceptual_model = None
roi_encoder      = None
focal_freq_loss  = None   # instance FocalFrequencyLoss


# ==================================================
# MAIN EDM LOSS  —  EXP-803 (Iterasi 1)
# --------------------------------------------------
# Perubahan dari EXP-715A:
#
#  [1] Pixel losses diturunkan ~40%:
#        loss_global  : 0.35 → 0.21
#        loss_tumor   : 1.50 → 0.90
#        loss_base    : 1.00 → 0.60
#        loss_enh     : 1.20 → 0.72
#
#  [2] VGG perceptual dinaikkan 3×:
#        loss_vgg     : 0.08 → 0.25
#        (VGG adalah anti-blur loss paling proven;
#         bobot 0.08 terlalu kecil untuk melawan
#         dominasi pixel loss)
#
#  [3] Focal Frequency Loss ditambahkan (BARU):
#        loss_freq    : 0.00 → 0.20
#        (menghukum hilangnya komponen frekuensi
#         tinggi secara adaptif → langsung menyerang
#         akar penyebab blur)
#
#  [4] Structural losses (edge, lap, ms-ssim, roi)
#      dipertahankan nilainya agar stabilitas
#      training tetap terjaga.
#
#  [5] GAN: lambda_gan 1.0 → 2.5, hinge → LSGAN
#      (diatur di config.py / trainer, bukan di sini)
#
#  Perbandingan total effective pixel weight:
#        EXP-715A : ~4.05
#        EXP-803  : ~2.43  (↓ ~40%)
#  Perbandingan total perceptual/freq weight:
#        EXP-715A : ~0.43
#        EXP-803  : ~0.80  (↑ ~86%)
# ==================================================
class EDMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_tuple, target, mask, input_tensor):
        global perceptual_model, focal_freq_loss

        pred, base, enh = pred_tuple
        mask_f = mask.float()

        t1 = input_tensor[:, 1:2, :, :]
        diff = target - t1

        healthy = (diff <= 0.025) & mask
        tumor   = (diff >  0.025) & mask

        # -----------------------------------------------
        # Pixel losses  [↓ ~40% dari EXP-715A]
        # -----------------------------------------------
        loss_global = torch.mean(torch.abs(pred - target)[mask])

        loss_tumor = (
            torch.mean(torch.abs(pred - target)[tumor])
            if tumor.any()
            else pred.new_tensor(0.0)
        )

        loss_base = (
            torch.mean(torch.abs(base - target)[healthy])
            if healthy.any()
            else pred.new_tensor(0.0)
        )

        enh_gt = torch.clamp(target - base.detach(), min=0.0)

        loss_enh = (
            torch.mean(torch.abs(enh - enh_gt)[tumor])
            if tumor.any()
            else torch.mean(torch.abs(enh - enh_gt)[mask])
        )

        # -----------------------------------------------
        # Structural losses  [bobot tidak berubah]
        # -----------------------------------------------
        loss_edge    = edge_loss(pred, target, mask)
        loss_lap     = laplacian_loss(pred, target, mask)
        loss_msssim  = ms_ssim_loss(pred, target, mask)

        # -----------------------------------------------
        # VGG Perceptual  [↑ 0.08 → 0.25]
        # -----------------------------------------------
        if perceptual_model is None:
            perceptual_model = VGGPerceptual().to(pred.device).eval()

        pred_clip   = torch.clamp(pred,   0.0, 1.0)
        target_clip = torch.clamp(target, 0.0, 1.0)

        loss_vgg = perceptual_model(
            (pred_clip   * mask_f).float(),
            (target_clip * mask_f).float()
        )

        # -----------------------------------------------
        # ROI feature consistency  [bobot tidak berubah]
        # -----------------------------------------------
        loss_roi = roi_contrastive_loss(pred, target, tumor)

        # -----------------------------------------------
        # [NEW] Focal Frequency Loss  [0.00 → 0.20]
        # Hanya dihitung pada region brain (mask) untuk
        # menghindari kontaminasi dari background hitam
        # yang bisa mendominasi spektrum.
        # -----------------------------------------------
        if focal_freq_loss is None:
            focal_freq_loss = FocalFrequencyLoss(
                loss_weight=1.0,
                alpha=1.0,
                patch_factor=1,
                ave_spectrum=False,
                log_matrix=False,
                batch_matrix=False,
            ).to(pred.device)

        pred_masked   = torch.clamp(pred_clip   * mask_f, 0.0, 1.0)
        target_masked = torch.clamp(target_clip * mask_f, 0.0, 1.0)
        loss_freq = focal_freq_loss(pred_masked, target_masked)

        # -----------------------------------------------
        # Final weighted sum
        # -----------------------------------------------
        #
        #  Komponen            EXP-715A   EXP-803   Δ
        #  ─────────────────── ──────── ──────── ────
        #  loss_global          0.35     0.21    -40%
        #  loss_tumor           1.50     0.90    -40%
        #  loss_base            1.00     0.60    -40%
        #  loss_enh             1.20     0.72    -40%
        #  loss_edge            0.08     0.08     =
        #  loss_lap             0.10     0.10     =
        #  loss_vgg             0.08     0.25    +213%
        #  loss_msssim          0.15     0.15     =
        #  loss_roi             0.12     0.12     =
        #  loss_freq [NEW]      0.00     0.20    NEW
        #
        total = (
            0.21 * loss_global   +   # pixel ↓
            0.90 * loss_tumor    +   # pixel ↓
            0.60 * loss_base     +   # pixel ↓
            0.72 * loss_enh      +   # pixel ↓
            0.08 * loss_edge     +   # structural =
            0.10 * loss_lap      +   # structural =
            0.25 * loss_vgg      +   # perceptual ↑↑
            0.15 * loss_msssim   +   # structural =
            0.12 * loss_roi      +   # feature    =
            0.20 * loss_freq         # freq [NEW]
        )

        return total


LOSS_REGISTRY = {
    "EDMLoss": EDMLoss()
}
