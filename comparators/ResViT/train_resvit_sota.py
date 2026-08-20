# Training script untuk ResViT SOTA baseline.
#
# LOSS & TRAINING RECIPE FAITHFUL ke ResViT/pix2pix asli:
#   - LSGAN (least-squares GAN, default resmi ResViT: no_lsgan=False) + L1
#     (lambda_A=10.0, lambda_adv=1.0) -- TIDAK diganti hinge loss atau
#     ditambah VGG/LPIPS seperti EXP-703 Anda (itu desain EDMSynth, bukan
#     bagian dari metode ResViT yang sedang dibandingkan).
#   - ImagePool (buffer 50 gambar) untuk update discriminator -- trik stabilisasi
#     asli dari kode ResViT/pix2pix, disalin dari resvit_image_pool.py.
#   - Linear LR decay: lr konstan untuk `niter` epoch, lalu decay linear ke 0
#     selama `niter_decay` epoch -- sesuai train_options.py ResViT asli
#     (bukan cosine/step schedule).
#
# INTEGRASI ke infrastruktur eksperimen Anda (mengikuti prinsip yang sama
# seperti DD-Res U-Net): checkpoint terbaik dipilih berdasarkan psnr_roi
# (evaluators/psnr_fixed_range.py Anda -- data_range TETAP dari
# metric_config.json, KONSISTEN dengan EXP-602/703 dan DD-Res U-Net),
# history.csv, prepare_experiment(), best_model.pth -- semua path/format sama.
#
# CATATAN INPUT/OUTPUT: model bekerja di ruang [-1,1] (Tanh), sesuai
# ResViTDataset yang sudah me-rescale data z-score Anda. val_psnr dihitung
# SETELAH inverse-transform kembali ke skala z-score asli, supaya angka
# psnr_roi ini benar-benar sebanding dengan DD-Res U-Net/SynDiff (bukan psnr
# di ruang [-1,1] yang skala/artinya beda).

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from resvit_dataset import ResViTDataset, from_tanh_range, _load_global_range
from resvit_model_wrapper import build_resvit_generator, build_resvit_discriminator
from resvit_image_pool import ImagePool
import resvit_networks_orig as networks

from experiment_utils import prepare_experiment
from psnr_fixed_range import psnr_roi
from config_resvit_sota import CONFIG
from early_stopping import EarlyStopping


def seed_worker(worker_id):
    worker_seed = CONFIG["seed"] + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def check_experiment_folder(CONFIG):
    exp_name = f"{CONFIG['experiment_id']}_{CONFIG['experiment_name']}_fold{CONFIG['fold']}"
    exp_dir = os.path.join(CONFIG["experiment_root"], exp_name)
    if os.path.exists(exp_dir):
        raise RuntimeError(
            f"\nExperiment folder already exists:\n{exp_dir}\n"
            "Change experiment_id / experiment_name / fold\n"
        )


def linear_decay_lr(optimizer, epoch, niter, niter_decay, base_lr):
    """Linear decay sesuai lr_policy='lambda' default ResViT: lr konstan
    selama `niter` epoch pertama, lalu turun linear ke 0 selama `niter_decay`
    epoch berikutnya."""
    if epoch <= niter:
        lr = base_lr
    else:
        decay_frac = (epoch - niter) / float(niter_decay + 1)
        lr = base_lr * max(0.0, 1.0 - decay_frac)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def main():

    check_experiment_folder(CONFIG)
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    random.seed(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EXP_DIR = prepare_experiment(CONFIG)
    print(f"\nRunning experiment folder:\n{EXP_DIR}\n")

    global_min, global_max = _load_global_range()

    # =========================
    # Dataset
    # =========================
    DATASET_ROOT = CONFIG["dataset_root"]
    FOLD = CONFIG["fold"]
    TRAIN_PATH = os.path.join(DATASET_ROOT, f"fold_{FOLD}", "train")
    VAL_PATH = os.path.join(DATASET_ROOT, f"fold_{FOLD}", "val")

    train_ds = ResViTDataset(TRAIN_PATH)
    val_ds = ResViTDataset(VAL_PATH)

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=CONFIG["num_workers"], pin_memory=CONFIG["pin_memory"],
        worker_init_fn=seed_worker, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=CONFIG["num_workers"], pin_memory=CONFIG["pin_memory"],
        worker_init_fn=seed_worker, persistent_workers=True
    )

    in_ch = len(CONFIG["modalities"])
    out_ch = 1

    # =========================
    # Model: Generator (ResViT) + Discriminator (PatchGAN, faithful)
    # =========================
    netG = build_resvit_generator(
        in_channels=in_ch, out_channels=out_ch,
        base_channels=CONFIG["base_channels"],
        vit_name=CONFIG["vit_name"], img_size=CONFIG["img_size"],
        pretrained_vit_path=CONFIG["pretrained_vit_path"],
        pretrained_resnet=CONFIG["pretrained_resnet"],
    ).to(device)

    netD = build_resvit_discriminator(in_ch + out_ch, ndf=64, n_layers=3).to(device)

    optimizer_G = torch.optim.Adam(netG.parameters(), lr=CONFIG["lr"], betas=(CONFIG["beta1"], 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=CONFIG["lr"], betas=(CONFIG["beta1"], 0.999))

    tensor_type = torch.cuda.FloatTensor if device.type == "cuda" else torch.FloatTensor
    criterion_gan = networks.GANLoss(use_lsgan=CONFIG["use_lsgan"], tensor=tensor_type)
    criterion_l1 = nn.L1Loss()

    fake_pool = ImagePool(CONFIG["pool_size"])

    total_epochs = CONFIG["niter"] + CONFIG["niter_decay"]
    early_stop = EarlyStopping(CONFIG.get("early_stop_patience", total_epochs))

    best_val_psnr = -float("inf")
    history = []

    for epoch in range(1, total_epochs + 1):

        lr_now = linear_decay_lr(optimizer_G, epoch, CONFIG["niter"], CONFIG["niter_decay"], CONFIG["lr"])
        linear_decay_lr(optimizer_D, epoch, CONFIG["niter"], CONFIG["niter_decay"], CONFIG["lr"])

        netG.train()
        netD.train()

        g_loss_sum, d_loss_sum, valid_batches = 0.0, 0.0, 0

        for x, y, _ in tqdm(train_loader, ncols=80, desc=f"Epoch {epoch}/{total_epochs} (lr={lr_now:.2e})"):
            x = x.to(device)
            y = y.to(device)

            fake_y = netG(x)

            # ---- Update D ----
            optimizer_D.zero_grad()
            fake_AB = fake_pool.query(torch.cat([x, fake_y], dim=1).data)
            pred_fake = netD(fake_AB.detach())
            loss_d_fake = criterion_gan(pred_fake, False)

            real_AB = torch.cat([x, y], dim=1)
            pred_real = netD(real_AB)
            loss_d_real = criterion_gan(pred_real, True)

            loss_d = (loss_d_fake + loss_d_real) * 0.5 * CONFIG["lambda_adv"]
            loss_d.backward()
            optimizer_D.step()

            # ---- Update G ----
            optimizer_G.zero_grad()
            fake_AB_for_g = torch.cat([x, fake_y], dim=1)
            pred_fake_for_g = netD(fake_AB_for_g)
            loss_g_gan = criterion_gan(pred_fake_for_g, True) * CONFIG["lambda_adv"]
            loss_g_l1 = criterion_l1(fake_y, y) * CONFIG["lambda_A"]
            loss_g = loss_g_gan + loss_g_l1
            loss_g.backward()
            optimizer_G.step()

            g_loss_sum += loss_g.item()
            d_loss_sum += loss_d.item()
            valid_batches += 1

        g_loss_avg = g_loss_sum / (valid_batches + 1e-8)
        d_loss_avg = d_loss_sum / (valid_batches + 1e-8)

        # =========================
        # Validation
        # =========================
        netG.eval()
        val_psnr = 0.0
        valid_batches = 0

        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device)
                y = y.to(device)

                fake_y = netG(x)

                # inverse-transform ke skala z-score asli sebelum psnr_roi,
                # supaya sebanding dengan DD-Res U-Net/SynDiff
                # CATATAN: val_psnr di sini dihitung di resolusi 256x256 (baik
                # fake_y maupun y sama-sama sudah di-resize dataset) -- ini
                # CUKUP untuk memilih checkpoint terbaik selama training, TAPI
                # BELUM sebanding langsung dengan DD-Res U-Net (native 512x512).
                # Untuk tabel hasil final di paper, buat skrip evaluasi terpisah
                # yang meng-upsample fake_y kembali ke 512x512 (bilinear) lalu
                # bandingkan dengan T1CE native 512x512 asli sebelum psnr_roi.
                y_orig = from_tanh_range(y, global_min, global_max)
                fake_orig = from_tanh_range(fake_y, global_min, global_max)
                mask = (y_orig != 0).float()

                if mask.sum() == 0:
                    continue

                val_psnr += psnr_roi(fake_orig, y_orig, mask).item()
                valid_batches += 1

        val_psnr /= (valid_batches + 1e-8)

        print(f"Epoch {epoch} | G_loss {g_loss_avg:.4f} | D_loss {d_loss_avg:.4f} | Val PSNR {val_psnr:.4f}")

        early_stop.step(val_psnr, epoch)
        if early_stop.stop:
            print(f"\nEarly stopping triggered at epoch {early_stop.stop_epoch}\n")
            break

        history.append({
            "epoch": epoch, "g_loss": g_loss_avg, "d_loss": d_loss_avg,
            "val_psnr": val_psnr, "lr": lr_now,
        })

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(netG.state_dict(), os.path.join(EXP_DIR, "best_model.pth"))

    pd.DataFrame(history).to_csv(os.path.join(EXP_DIR, "history.csv"), index=False)
    print("\nTraining completed\n")


if __name__ == "__main__":
    import sys, json

    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            new_cfg = json.load(f)
        CONFIG.clear()
        CONFIG.update(new_cfg)

    main()
