# DESTINATION: train_pGAN.py (root proyek, sejajar dgn train_edm_exp602.py /
# train_exp703rerun3.py)
#
# Training loop pGAN (Dar et al., 2019) diadaptasi ke dataset 2.5D multi-modal
# Anda, MENGIKUTI STRUKTUR train_exp703rerun3.py Anda seteliti mungkin
# (seeding/determinism, EarlyStopping, psnr_roi, LPIPS, prepare_experiment,
# history.csv, best_model.pth) supaya hasilnya bisa dibandingkan langsung
# di tabel yang sama dgn EXP-602/703.
#
# PRASYARAT sebelum menjalankan:
#   1) Salin configs/config_pGAN.py -> configs/config.py (lihat catatan di file itu)
#   2) Tempatkan models/pgan_networks.py dan losses/pgan_losses.py di proyek Anda
#   3) Pastikan configs/metric_config.json (dipakai psnr_fixed_range.py) sudah ada
#      -- file ini sudah dipakai EXP-602/703 sehingga seharusnya sudah tersedia.
#
# Semua adaptasi dari kode resmi pGAN dijelaskan di strategi_adaptasi_pGAN_cGAN.md.

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random
import numpy as np
import pandas as pd
import torch
import lpips
from tqdm import tqdm

from utils.experiment_utils import prepare_experiment
from utils.early_stopping import EarlyStopping
from evaluators.psnr_fixed_range import psnr_roi

from torch.utils.data import DataLoader
from datasets.dataset_2p5d import MRI2p5DDataset

from configs.config import CONFIG  # <- WAJIB configs/config.py berisi config_pGAN.py Anda

from losses.loss_registry import VGGPerceptual  # reuse definisi VGG persis milik EXP-703
from losses.pgan_losses import PGANReconLoss, LSGANLoss
from models.pgan_networks import PGANGenerator, PGANDiscriminator, init_weights


def seed_worker(worker_id):
    worker_seed = CONFIG["seed"] + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def check_experiment_folder(CONFIG):
    exp_name = f"{CONFIG['experiment_id']}_{CONFIG['experiment_name']}_fold{CONFIG['fold']}"
    exp_dir = os.path.join(CONFIG["experiment_root"], exp_name)
    if os.path.exists(exp_dir):
        raise RuntimeError(
            f"\nExperiment folder already exists:\n{exp_dir}\n"
            "Change experiment_id / experiment_name / fold\n"
        )


def main():

    check_experiment_folder(CONFIG)

    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])
    torch.use_deterministic_algorithms(True)
    np.random.seed(CONFIG["seed"])
    random.seed(CONFIG["seed"])

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================
    # Prepare Experiment Folder
    # =========================
    EXP_DIR = prepare_experiment(CONFIG)
    print(f"\nRunning experiment folder:\n{EXP_DIR}\n")

    # =========================
    # Dataset (dipakai apa adanya -- sudah kompatibel: x=9ch input, y=1ch T1CE)
    # =========================
    DATASET_ROOT = CONFIG["dataset_root"]
    FOLD = CONFIG["fold"]

    TRAIN_PATH = os.path.join(DATASET_ROOT, f"fold_{FOLD}", "train")
    VAL_PATH = os.path.join(DATASET_ROOT, f"fold_{FOLD}", "val")

    train_ds = MRI2p5DDataset(TRAIN_PATH)
    val_ds = MRI2p5DDataset(VAL_PATH)

    g = torch.Generator()
    g.manual_seed(CONFIG["seed"])

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
        worker_init_fn=seed_worker,
        persistent_workers=False,
    )

    # =========================
    # Model: Generator (ResNet-9block) + Discriminator (PatchGAN)
    # =========================
    n_modalities = len(CONFIG["modalities"])
    input_nc = 3 * n_modalities   # 2.5D (3 slice tetangga) x jumlah modalitas
    output_nc = 1                  # T1CE center slice

    G = PGANGenerator(
        input_nc=input_nc,
        output_nc=output_nc,
        ngf=CONFIG["ngf"],
        norm=CONFIG["norm"],
        n_blocks=CONFIG["n_blocks"],
        final_activation=CONFIG["final_activation"],
    ).to(device)
    init_weights(G, init_type=CONFIG["init_type"])

    use_gan = CONFIG["gan"]["use_gan"]

    if use_gan:
        d_cfg = CONFIG["discriminator"]
        D = PGANDiscriminator(
            in_channels=input_nc + output_nc,
            base_channels=d_cfg["base_channels"],
            n_layers=CONFIG["n_layers_D"],
            norm=CONFIG["norm"],
            use_sigmoid=d_cfg["use_sigmoid"],
        ).to(device)
        init_weights(D, init_type=CONFIG["init_type"])

        optimizer_D = torch.optim.Adam(
            D.parameters(), lr=CONFIG["lr_D"], betas=(CONFIG["beta1"], 0.999)
        )
        gan_loss = LSGANLoss()
        lambda_adv = CONFIG["gan"]["lambda_adv"]
        warmup_epochs = CONFIG["gan"]["warmup_epochs"]

    optimizer_G = torch.optim.Adam(
        G.parameters(), lr=CONFIG["lr"], betas=(CONFIG["beta1"], 0.999),
        weight_decay=CONFIG["weight_decay"],
    )

    # =========================
    # Recon loss (L1 + VGG perceptual, reuse VGGPerceptual milik EXP-703)
    # =========================
    vgg_module = VGGPerceptual().to(device)
    recon_loss_fn = PGANReconLoss(
        lambda_A=CONFIG["loss"]["lambda_A"],
        lambda_vgg=CONFIG["loss"]["lambda_vgg"],
        vgg_module=vgg_module,
    )

    # =========================
    # LPIPS (evaluasi, identik dgn EXP-703)
    # =========================
    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()

    scaler = torch.cuda.amp.GradScaler(enabled=CONFIG["amp"])
    early_stop = EarlyStopping(CONFIG["early_stop_patience"])

    best_score = -float("inf")
    history = []

    for epoch in range(CONFIG["epochs"]):

        G.train()
        if use_gan:
            D.train()

        train_loss = 0.0
        valid_batches = 0
        total_d_loss, total_g_loss, gan_batches = 0.0, 0.0, 0

        d_loss = torch.tensor(0.0, device=device)
        gan_g_loss = torch.tensor(0.0, device=device)

        use_gan_now = use_gan and (epoch >= warmup_epochs)

        pbar = tqdm(train_loader, ncols=100, desc=f"Epoch {epoch+1}")
        for x, y, _ in pbar:

            x = x.to(device)
            y = y.to(device)
            mask = (y != 0)
            mask_float = mask.float()

            if mask.sum() == 0:
                continue

            # ======================
            # GENERATOR FORWARD + RECON LOSS
            # ======================
            with torch.amp.autocast(device_type=device.type, enabled=CONFIG["amp"]):
                pred = G(x)
                recon, recon_parts = recon_loss_fn(pred, y, mask)

            if use_gan and use_gan_now:
                pred_gan = pred * mask_float
                y_gan = y * mask_float

            # ======================
            # UPDATE D
            # ======================
            if use_gan and use_gan_now:

                optimizer_D.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=device.type, enabled=CONFIG["amp"]):
                    real_pred = D(x, y_gan)
                    fake_pred = D(x, pred_gan.detach())
                    d_loss = gan_loss.d_loss(real_pred, fake_pred)

                scaler.scale(d_loss).backward()
                scaler.step(optimizer_D)

            # ======================
            # UPDATE G
            # ======================
            optimizer_G.zero_grad(set_to_none=True)

            if use_gan and use_gan_now:
                with torch.amp.autocast(device_type=device.type, enabled=CONFIG["amp"]):
                    fake_pred = D(x, pred_gan)
                    gan_g_loss = gan_loss.g_loss(fake_pred)
                    total_loss = recon + lambda_adv * gan_g_loss
            else:
                total_loss = recon

            scaler.scale(total_loss).backward()
            scaler.step(optimizer_G)
            scaler.update()

            train_loss += total_loss.item()
            if use_gan and use_gan_now:
                total_d_loss += d_loss.item()
                total_g_loss += gan_g_loss.item()
                gan_batches += 1

            valid_batches += 1

            if use_gan and use_gan_now:
                pbar.set_postfix(
                    D=f"{d_loss.item():.4f}",
                    G_adv=f"{gan_g_loss.item():.4f}",
                    Recon=f"{recon.item():.4f}",
                )
            else:
                pbar.set_postfix(Recon=f"{recon.item():.4f}")

        train_loss /= (valid_batches + 1e-8)
        epoch_d_loss = (total_d_loss / gan_batches) if gan_batches > 0 else 0.0
        epoch_g_loss = (total_g_loss / gan_batches) if gan_batches > 0 else 0.0

        if use_gan_now and valid_batches > 0:
            print(f"D: {epoch_d_loss:.4f} | G_gan: {epoch_g_loss:.4f}")

        # =========================
        # Validation
        # =========================
        G.eval()

        val_psnr = 0.0
        val_lpips = 0.0
        valid_batches = 0

        with torch.no_grad():

            for x, y, _ in val_loader:

                x = x.to(device)
                y = y.to(device)

                mask = (y != 0).float()
                if mask.sum() == 0:
                    continue

                with torch.amp.autocast(device_type=device.type, enabled=CONFIG["amp"]):
                    pred = G(x)

                pred = pred.float()

                val_psnr += psnr_roi(pred, y, mask).item()

                pred_lp = torch.clamp(pred, 0.0, 1.0)
                y_lp = torch.clamp(y, 0.0, 1.0)
                pred_lp = (pred_lp * 2.0) - 1.0
                y_lp = (y_lp * 2.0) - 1.0

                if pred_lp.shape[1] == 1:
                    pred_lp = pred_lp.repeat(1, 3, 1, 1)
                    y_lp = y_lp.repeat(1, 3, 1, 1)

                val_lpips += lpips_model(pred_lp, y_lp).mean().item()

                valid_batches += 1

        val_psnr /= (valid_batches + 1e-8)
        val_lpips /= (valid_batches + 1e-8)

        if CONFIG["early_stop_metric"] == "psnr_lpips":
            score = val_psnr - 0.5 * val_lpips
        else:
            score = val_psnr

        print(f"Epoch {epoch+1} | Train Loss {train_loss:.4f} | Val PSNR {val_psnr:.4f} | LPIPS {val_lpips:.4f} | Score {score:.4f}")

        early_stop.step(score, epoch + 1)

        if early_stop.stop:
            print(f"\nEarly stopping triggered at epoch {early_stop.stop_epoch}\n")
            break

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_psnr": val_psnr,
            "val_lpips": val_lpips,
            "d_loss": epoch_d_loss,
            "g_loss": epoch_g_loss,
            "score": score,
        })

        if score > best_score:

            best_score = score

            torch.save(G.state_dict(), os.path.join(EXP_DIR, "best_model.pth"))

            if use_gan and use_gan_now:
                torch.save(D.state_dict(), os.path.join(EXP_DIR, "best_discriminator.pth"))

    pd.DataFrame(history).to_csv(
        os.path.join(EXP_DIR, "history.csv"),
        index=False,
    )

    print("\nTraining completed\n")


if __name__ == "__main__":
    # PATCH: dukung override CONFIG lewat file JSON di argv[1], mengikuti
    # pola yang SUDAH dipakai evaluate_experiment_pGAN.py / select_visualization_cases.py /
    # inference_visualize_pGAN.py -- WAJIB supaya run_experiments_all_fold.py
    # (yang memanggil `python train_pGAN.py configs/temp_config.json` per fold)
    # benar-benar mengganti fold/experiment_id, bukan diam-diam mengabaikannya
    # dan selalu memakai isi configs/config.py yang statis (ini penyebab error
    # "Fold 1" tapi folder yang dicek tetap "...fold0").
    import sys
    import json
    from configs.config import CONFIG

    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            new_cfg = json.load(f)
        CONFIG.clear()
        CONFIG.update(new_cfg)

    main()
