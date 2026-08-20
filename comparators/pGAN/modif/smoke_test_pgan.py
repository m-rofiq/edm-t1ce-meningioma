# DESTINATION: jalankan sekali di root proyek Anda (sejajar train_pGAN.py) untuk
# validasi cepat sebelum full training run. TIDAK perlu dataset asli -- script ini
# membuat dummy .npy (shape identik dgn data asli Anda) di folder temporary.
#
# Kenapa script ini ada: saya tidak punya akses ke dataset/GPU/torch di sandbox
# analisis ini (tidak bisa download torch ~750MB di sini), jadi saya tidak bisa
# menjalankan train_pGAN.py end-to-end dari sisi saya. Kode sudah lolos
# py_compile + pyflakes (tidak ada syntax/undefined-name error) dan sudah saya
# telusuri manual utk konsistensi shape (lihat ringkasan di chat), tapi jalankan
# script ini di mesin Anda (yang sudah py torch+lpips+project modules-nya)
# sebagai verifikasi runtime terakhir sebelum training penuh 200 epoch.
#
# Cara pakai:
#   python smoke_test_pgan.py
#
# Script ini HANYA menguji wiring (shape, forward, backward, loss finite) --
# BUKAN kualitas hasil sintesis (wajar rugi tinggi/random krn bobot random & data dummy).

import os
import shutil
import tempfile

import numpy as np
import torch

MODALITIES = ["T1", "T2", "FLAIR"]
N_SAMPLES = 6
SHAPE = (3, 512, 512)


def build_dummy_dataset(root):
    for m in MODALITIES + ["T1CE"]:
        os.makedirs(os.path.join(root, m), exist_ok=True)

    rng = np.random.default_rng(0)

    for i in range(N_SAMPLES):
        fname = f"sample_{i:03d}.npy"
        for m in MODALITIES:
            vol = rng.normal(0, 1, SHAPE).astype(np.float32)
            # sisakan sedikit background persis 0 spy mask=(y!=0) tidak semuanya True
            vol[:, :16, :16] = 0.0
            np.save(os.path.join(root, m, fname), vol)

        t1ce = rng.normal(0, 1, SHAPE).astype(np.float32)
        t1ce[:, :16, :16] = 0.0
        np.save(os.path.join(root, "T1CE", fname), t1ce)


def main():
    tmp_root = tempfile.mkdtemp(prefix="pgan_smoketest_")
    print(f"Dummy dataset: {tmp_root}")

    try:
        build_dummy_dataset(tmp_root)

        # =========================
        # PENTING: dataset_2p5d.py melakukan `from configs.config import CONFIG`
        # SAAT DI-IMPORT (bukan lazy), jadi configs/config.py HARUS SUDAH berisi
        # config_pGAN.py Anda (modalities=["T1","T2","FLAIR"]) SEBELUM baris impor
        # di bawah ini dieksekusi -- monkeypatch setelah import tidak akan
        # terbaca oleh dataset_2p5d.py (Python `from x import y` mengikat nilai
        # saat itu juga, bukan referensi hidup ke modul).
        # =========================
        from configs.config import CONFIG as _ACTIVE_CONFIG
        if list(_ACTIVE_CONFIG.get("modalities", [])) != MODALITIES:
            raise RuntimeError(
                "configs/config.py belum berisi config_pGAN.py Anda "
                f"(modalities saat ini = {_ACTIVE_CONFIG.get('modalities')}, "
                f"seharusnya {MODALITIES}). Jalankan dulu:\n"
                "  copy configs\\config_pGAN.py configs\\config.py\n"
                "baru jalankan smoke_test_pgan.py lagi."
            )

        from datasets.dataset_2p5d import MRI2p5DDataset  # noqa
        from models.pgan_networks import PGANGenerator, PGANDiscriminator, init_weights
        from losses.pgan_losses import PGANReconLoss, LSGANLoss

        ds = MRI2p5DDataset(tmp_root)
        assert len(ds) == N_SAMPLES, f"Jumlah sample tidak sesuai: {len(ds)}"

        x, y, fname = ds[0]
        print(f"x.shape={tuple(x.shape)} y.shape={tuple(y.shape)} fname={fname}")
        assert x.shape == (9, 512, 512), f"x shape salah: {x.shape} (harus 9,512,512)"
        assert y.shape == (1, 512, 512), f"y shape salah: {y.shape} (harus 1,512,512)"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        G = PGANGenerator(input_nc=9, output_nc=1, ngf=32, n_blocks=2).to(device)
        init_weights(G, init_type="normal")

        D = PGANDiscriminator(in_channels=10, base_channels=16, n_layers=2).to(device)
        init_weights(D, init_type="normal")

        # VGG perceptual: coba reuse implementasi asli Anda; kalau gagal
        # (mis. tidak ada koneksi internet utk download bobot VGG16 pretrained),
        # fallback ke stub ringan HANYA utk smoke test wiring, BUKAN utk training asli.
        try:
            from losses.loss_registry import VGGPerceptual
            vgg_module = VGGPerceptual().to(device)
            print("VGGPerceptual asli berhasil dimuat.")
        except Exception as e:
            print(f"[WARNING] Gagal load VGGPerceptual asli ({e}); pakai stub utk smoke test.")

            class _StubVGG(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)

                def forward(self, pred, target):
                    return torch.nn.functional.l1_loss(
                        self.conv(pred.repeat(1, 3, 1, 1)),
                        self.conv(target.repeat(1, 3, 1, 1)),
                    )

            vgg_module = _StubVGG().to(device)

        recon_loss_fn = PGANReconLoss(lambda_A=100.0, lambda_vgg=100.0, vgg_module=vgg_module)
        gan_loss = LSGANLoss()

        opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
        opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

        loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)

        print("\n--- Menjalankan 2 iterasi training dummy ---")
        for step, (x, y, _) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            mask = (y != 0)
            mask_f = mask.float()

            pred = G(x)
            assert pred.shape == y.shape, f"Shape mismatch pred={pred.shape} y={y.shape}"

            recon, parts = recon_loss_fn(pred, y, mask)
            assert torch.isfinite(recon), "Recon loss tidak finite (NaN/Inf)"

            pred_gan = pred * mask_f
            y_gan = y * mask_f

            opt_D.zero_grad(set_to_none=True)
            real_pred = D(x, y_gan)
            fake_pred = D(x, pred_gan.detach())
            d_loss = gan_loss.d_loss(real_pred, fake_pred)
            d_loss.backward()
            opt_D.step()

            opt_G.zero_grad(set_to_none=True)
            fake_pred = D(x, pred_gan)
            g_adv = gan_loss.g_loss(fake_pred)
            total_loss = recon + 1.0 * g_adv
            total_loss.backward()
            opt_G.step()

            print(f"step {step}: recon={recon.item():.4f} "
                  f"L1={parts['L1'].item():.4f} VGG={parts['VGG'].item():.4f} "
                  f"D={d_loss.item():.4f} G_adv={g_adv.item():.4f}")

            if step >= 1:
                break

        print("\nSMOKE TEST PASSED -- wiring dataset -> generator -> discriminator -> "
              "loss -> backward semuanya berjalan tanpa error.")

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
