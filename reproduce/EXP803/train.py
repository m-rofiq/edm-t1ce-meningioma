import torch
import random
import numpy as np
import os
import pandas as pd
import torch.nn.functional as F
import lpips

from utils.experiment_utils import prepare_experiment

from torch.utils.data import DataLoader
from datasets.dataset_2p5d import MRI2p5DDataset
from evaluators.psnr_fixed_range import psnr_roi
from configs.config import CONFIG
from tqdm import tqdm
from models.model_registry import MODEL_REGISTRY
from losses.loss_registry import LOSS_REGISTRY
from utils.early_stopping import EarlyStopping

# [CHANGED] HingeGANLoss dihapus → pakai LSGAN dari loss_registry
from losses.loss_registry import lsgan_generator_loss, lsgan_discriminator_loss


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


def main():

    check_experiment_folder(CONFIG)
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])
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
    # Dataset
    # =========================
    DATASET_ROOT = CONFIG["dataset_root"]
    FOLD         = CONFIG["fold"]

    TRAIN_PATH = os.path.join(DATASET_ROOT, f"fold_{FOLD}", "train")
    VAL_PATH   = os.path.join(DATASET_ROOT, f"fold_{FOLD}", "val")

    train_ds = MRI2p5DDataset(TRAIN_PATH)
    val_ds   = MRI2p5DDataset(VAL_PATH)

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
        worker_init_fn=seed_worker,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
        worker_init_fn=seed_worker,
        persistent_workers=True
    )

    # =========================
    # Model
    # =========================
    model_name = CONFIG["model"]

    if model_name in ["DMECNetStep1", "DMECNetStep2", "DMECNetStep3"]:
        model = MODEL_REGISTRY[model_name](
            use_t2    = "T2"    in CONFIG["modalities"],
            use_flair = "FLAIR" in CONFIG["modalities"],
            base_ch   = CONFIG["base_channels"]
        ).to(device)
    else:
        model = MODEL_REGISTRY[model_name](
            base_channels=CONFIG["base_channels"],
            in_channels=3 * len(CONFIG["modalities"])
        ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        betas=(0.5, 0.999),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )

    # =========================
    # LPIPS INIT
    # =========================
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()

    # =========================
    # GAN INIT
    # =========================
    use_gan = CONFIG["gan"]["use_gan"]

    if use_gan:
        d_cfg = CONFIG["discriminator"]

        D = MODEL_REGISTRY[d_cfg["name"]](
            in_channels      = d_cfg["in_channels"],
            base_channels    = d_cfg["base_channels"],
            use_spectral_norm= d_cfg["use_spectral_norm"]
        ).to(device)

        optimizer_D = torch.optim.Adam(
            D.parameters(),
            lr=CONFIG["lr"],
            betas=(0.5, 0.999)
        )

        # [CHANGED] lambda_gan 1.0 → 2.5  (diambil dari config)
        # [CHANGED] gan_loss object dihapus → pakai lsgan_*_loss langsung
        lambda_gan     = CONFIG["gan"]["lambda_gan"]    # 2.5
        warmup_epochs  = CONFIG["gan"]["warmup_epochs"] # 20
        gan_loss_type  = CONFIG["gan"]["gan_loss"]      # "lsgan"

        # Validasi: pastikan config konsisten
        assert gan_loss_type == "lsgan", (
            f"Trainer ini hanya mendukung gan_loss='lsgan'. "
            f"Dapat: '{gan_loss_type}'. Periksa config.py."
        )

    loss_fn = LOSS_REGISTRY[CONFIG["loss"]]

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda" and CONFIG["amp"]))

    early_stop = EarlyStopping(CONFIG["early_stop_patience"])

    best_val_psnr = -float("inf")
    history       = []

    for epoch in range(CONFIG["epochs"]):

        model.train()

        train_loss    = 0.0
        valid_batches = 0
        total_d_loss  = 0.0
        total_g_loss  = 0.0
        gan_batches   = 0

        # Inisialisasi agar tidak error saat logging epoch pertama (warmup)
        d_loss     = torch.tensor(0.0, device=device)
        gan_g_loss = torch.tensor(0.0, device=device)

        use_gan_now = use_gan and (epoch >= warmup_epochs)

        pbar = tqdm(train_loader, ncols=110, desc=f"Epoch {epoch+1}")

        for x, y, _ in pbar:

            x    = x.to(device)
            y    = y.to(device)
            mask = (y != 0)
            mask_float = mask.float()

            if mask.sum() == 0:
                continue

            # ==========================================
            # GENERATOR FORWARD
            # ==========================================
            with torch.amp.autocast(device_type=device.type, enabled=CONFIG["amp"]):
                output = model(x)

                if isinstance(output, tuple):
                    pred, base, enh = output
                    edm_loss = loss_fn(output, y, mask, x)
                else:
                    pred     = output
                    edm_loss = loss_fn(pred, y, mask)

            # ==========================================
            # PREP GAN INPUT
            # Masking dilakukan sekali di sini, dipakai
            # oleh update D maupun G.
            # ==========================================
            if use_gan and use_gan_now:
                pred_gan = pred * mask_float        # detach di step D
                y_gan    = y    * mask_float

            # ==========================================
            # UPDATE D  (LSGAN)
            # ------------------------------------------
            # [CHANGED] Ganti HingeGANLoss.d_loss()
            #           → lsgan_discriminator_loss()
            #
            # LSGAN D loss:
            #   0.5 * [E(D(real) - 1)^2 + E(D(fake))^2]
            # ==========================================
            if use_gan and use_gan_now:

                optimizer_D.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=device.type, enabled=CONFIG["amp"]):
                    real_pred = D(x, y_gan)
                    fake_pred = D(x, pred_gan.detach())  # .detach() → tidak update G
                    d_loss    = lsgan_discriminator_loss(real_pred, fake_pred)

                scaler.scale(d_loss).backward()
                scaler.step(optimizer_D)
                # [NOTE] scaler.update() HANYA dipanggil sekali di akhir step,
                # setelah semua optimizer di-step. Dipindah ke bawah.

            # ==========================================
            # UPDATE G  (EDMLoss + lambda_gan * LSGAN_G)
            # ------------------------------------------
            # [CHANGED] Ganti HingeGANLoss.g_loss()
            #           → lsgan_generator_loss()
            #
            # LSGAN G loss:
            #   E(D(G(x)) - 1)^2
            # Gradient tidak saturate → signal ke G tetap
            # informatif bahkan saat D sangat kuat.
            # ==========================================
            optimizer.zero_grad(set_to_none=True)

            if use_gan and use_gan_now:
                with torch.amp.autocast(device_type=device.type, enabled=CONFIG["amp"]):
                    fake_pred_g = D(x, pred_gan)        # tanpa .detach()
                    gan_g_loss  = lsgan_generator_loss(fake_pred_g)
                    total_loss  = edm_loss + lambda_gan * gan_g_loss
            else:
                total_loss = edm_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)

            # [FIX] scaler.update() dipanggil sekali per iterasi,
            # setelah SEMUA optimizer di-step.
            # Di EXP-715A, scaler.update() hanya ada setelah step(optimizer),
            # tapi tidak setelah step(optimizer_D) → potensi scale drift
            # pada iterasi GAN aktif. Ini sudah benar dengan posisi di sini.
            scaler.update()

            # ==========================================
            # LOGGING PER BATCH
            # ==========================================
            train_loss += total_loss.item()
            valid_batches += 1

            if use_gan and use_gan_now:
                total_d_loss += d_loss.item()
                total_g_loss += gan_g_loss.item()
                gan_batches  += 1
                pbar.set_postfix(
                    D      = f"{d_loss.item():.4f}",
                    G_adv  = f"{gan_g_loss.item():.4f}",
                    EDM    = f"{edm_loss.item():.4f}"
                )
            else:
                pbar.set_postfix(EDM=f"{edm_loss.item():.4f}")

        # ==========================================
        # EPOCH SUMMARY
        # ==========================================
        train_loss   /= (valid_batches + 1e-8)
        epoch_d_loss  = (total_d_loss / gan_batches) if gan_batches > 0 else 0.0
        epoch_g_loss  = (total_g_loss / gan_batches) if gan_batches > 0 else 0.0

        if use_gan_now:
            print(f"[GAN] D: {epoch_d_loss:.4f} | G_adv: {epoch_g_loss:.4f} | λ={lambda_gan}")

        # ==========================================
        # VALIDATION
        # ==========================================
        model.eval()

        val_psnr      = 0.0
        val_lpips     = 0.0
        valid_batches = 0

        with torch.no_grad():
            for x, y, _ in val_loader:

                x    = x.to(device)
                y    = y.to(device)
                mask = (y != 0).float()

                if mask.sum() == 0:
                    continue

                with torch.amp.autocast(device_type=device.type, enabled=CONFIG["amp"]):
                    output = model(x)

                pred = output[0].float() if isinstance(output, tuple) else output.float()

                val_psnr += psnr_roi(pred, y, mask).item()

                pred_lp = torch.clamp(pred, 0.0, 1.0)
                y_lp    = torch.clamp(y,    0.0, 1.0)

                pred_lp = (pred_lp * 2.0) - 1.0
                y_lp    = (y_lp    * 2.0) - 1.0

                if pred_lp.shape[1] == 1:
                    pred_lp = pred_lp.repeat(1, 3, 1, 1)
                    y_lp    = y_lp.repeat(1, 3, 1, 1)

                val_lpips    += lpips_model(pred_lp, y_lp).mean().item()
                valid_batches += 1

        val_psnr  /= (valid_batches + 1e-8)
        val_lpips /= (valid_batches + 1e-8)

        print(
            f"Epoch {epoch+1:>3} | "
            f"Train Loss {train_loss:.4f} | "
            f"Val PSNR {val_psnr:.4f} | "
            f"LPIPS {val_lpips:.4f}"
        )

        # ==========================================
        # EARLY STOPPING & CHECKPOINT
        # ==========================================
        early_stop.step(val_psnr, epoch + 1)

        if early_stop.stop:
            print(f"\nEarly stopping triggered at epoch {early_stop.stop_epoch}\n")
            break

        history.append({
            "epoch"      : epoch + 1,
            "train_loss" : train_loss,
            "val_psnr"   : val_psnr,
            "val_lpips"  : val_lpips,
            "d_loss"     : epoch_d_loss,
            "g_loss"     : epoch_g_loss
        })

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(model.state_dict(), os.path.join(EXP_DIR, "best_model.pth"))
            if use_gan and use_gan_now:
                torch.save(D.state_dict(), os.path.join(EXP_DIR, "best_discriminator.pth"))

    pd.DataFrame(history).to_csv(
        os.path.join(EXP_DIR, "history.csv"),
        index=False
    )
    print("\nTraining completed\n")


if __name__ == "__main__":
    main()
