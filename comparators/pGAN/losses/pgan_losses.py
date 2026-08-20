# DESTINATION: losses/pgan_losses.py (letakkan di package `losses/` proyek Anda)
#
# Loss resmi pGAN (Dar et al.): L1 pixel + VGG perceptual + adversarial LSGAN.
# ADAPTASI vs kode resmi (lihat strategi_adaptasi_pGAN_cGAN.md bagian 2d):
#   - L1 dan VGG loss di-MASK ke ROI otak (mask = y!=0), supaya konsisten dgn
#     EDMLoss/psnr_roi Anda. Kode resmi pGAN menghitung loss di seluruh gambar
#     termasuk background.
#   - VGGPerceptual DI-REUSE dari losses/loss_registry.py (definisi EXP-703:
#     multi-scale relu1_2 + relu2_2 + relu3_3, rata-rata 3 stage) -- BUKAN
#     ditulis ulang -- supaya perceptual loss dihitung dgn cara yang SAMA
#     persis dgn baseline EDMSynth Anda (apple-to-apple).
#   - Adversarial loss: LSGAN (default resmi pGAN), BUKAN hinge (yg dipakai
#     EXP-703 Anda) -- supaya baseline tetap merepresentasikan pGAN sebagaimana
#     didesain penulisnya, bukan pGAN yang diubah sesuai training recipe Anda.

import torch
import torch.nn as nn
import torch.nn.functional as F


class PGANReconLoss(nn.Module):
    """
    Recon loss resmi pGAN: lambda_A * L1 + lambda_vgg * VGGPerceptual, di-mask ke ROI.

    vgg_module: instance dari losses.loss_registry.VGGPerceptual, dibuat SEKALI
    di train_pGAN.py dan di-pass ke sini (hindari load VGG16 pretrained dua kali
    kalau ada beberapa loss yang butuh VGG).
    """

    def __init__(self, lambda_A=100.0, lambda_vgg=100.0, vgg_module=None):
        super().__init__()
        if vgg_module is None:
            raise ValueError(
                "vgg_module wajib diisi -- reuse VGGPerceptual dari losses.loss_registry "
                "(instance yang sama dgn yang dipakai EDMSynth) supaya perceptual loss "
                "dihitung dgn cara yang identik."
            )
        self.lambda_A = lambda_A
        self.lambda_vgg = lambda_vgg
        self.vgg = vgg_module

    def forward(self, pred, target, mask):
        mask_f = mask.float()

        l1 = F.l1_loss(pred[mask], target[mask]) * self.lambda_A

        pred_m = pred * mask_f
        target_m = target * mask_f
        vgg = self.vgg(pred_m, target_m) * self.lambda_vgg

        total = l1 + vgg
        return total, {"L1": l1.detach(), "VGG": vgg.detach()}


class LSGANLoss(nn.Module):
    """
    LSGAN (least-squares GAN) -- default resmi pGAN/cGAN (Dar et al.).
    Interface (d_loss/g_loss) disamakan dgn losses/gan_loss.py (HingeGANLoss)
    Anda supaya training loop bisa reuse pola pemanggilan yang sama.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def d_loss(self, real_pred, fake_pred):
        loss_real = self.mse(real_pred, torch.ones_like(real_pred))
        loss_fake = self.mse(fake_pred, torch.zeros_like(fake_pred))
        return 0.5 * (loss_real + loss_fake)

    def g_loss(self, fake_pred):
        return self.mse(fake_pred, torch.ones_like(fake_pred))
