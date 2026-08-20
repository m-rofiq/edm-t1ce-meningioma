import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import random
import numpy as np
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
from losses.gan_loss import HingeGANLoss


def seed_worker(worker_id):

    worker_seed = CONFIG["seed"] + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    
    

def check_experiment_folder(CONFIG):

    exp_name = f"{CONFIG['experiment_id']}_{CONFIG['experiment_name']}_fold{CONFIG['fold']}"

    exp_dir = os.path.join(
        CONFIG["experiment_root"],
        exp_name
    )

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
    # Dataset
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
        persistent_workers=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
        worker_init_fn=seed_worker,
        persistent_workers=False
    )

    # =========================
    # Model
    # =========================

    model_name = CONFIG["model"]

    if model_name in ["DMECNetStep1", "DMECNetStep2", "DMECNetStep3"]:

        model = MODEL_REGISTRY[model_name](
            use_t2 = "T2" in CONFIG["modalities"],
            use_flair = "FLAIR" in CONFIG["modalities"],
            base_ch = CONFIG["base_channels"]
        ).to(device)

    else:
        model = MODEL_REGISTRY[model_name](
            base_channels=CONFIG["base_channels"],
            in_channels = 3 * len(CONFIG["modalities"])
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
            in_channels=d_cfg["in_channels"],
            base_channels=d_cfg["base_channels"],
            use_spectral_norm=d_cfg["use_spectral_norm"]
        ).to(device)

        optimizer_D = torch.optim.Adam(D.parameters(), lr=CONFIG["lr_D"], betas=(0.5, 0.999))

        gan_loss = HingeGANLoss()
        lambda_gan = CONFIG["gan"]["lambda_gan"]
        warmup_epochs = CONFIG["gan"]["warmup_epochs"]

    loss_fn = LOSS_REGISTRY[CONFIG["loss"]]

    #scaler = torch.amp.GradScaler(enabled=(device.type == "cuda" and CONFIG["amp"]))
    scaler = torch.cuda.amp.GradScaler(enabled=CONFIG["amp"])

    early_stop = EarlyStopping(CONFIG["early_stop_patience"])

    #best_val_psnr = -float("inf")
    best_score = -float("inf")

    history = []

    #print("USE GAN:", use_gan)
    #print("WARMUP:", warmup_epochs)

    for epoch in range(CONFIG["epochs"]):

        model.train()
        train_loss = 0
        valid_batches = 0
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        total_d_loss = 0.0 
        total_g_loss = 0.0 
        gan_batches = 0

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
            # GENERATOR FORWARD
            # ======================
            with torch.amp.autocast(device_type=device.type, enabled=CONFIG["amp"]):
                output = model(x)

                if isinstance(output, tuple):
                    pred, base, enh = output
                    edm_loss = loss_fn(output, y, mask, x)
                else:
                    pred = output
                    edm_loss = loss_fn(pred, y, mask)
            
            # ======================
            # PREP GAN INPUT (DI SINI)
            # ======================
            if use_gan and use_gan_now:
                pred_gan = pred * mask_float
                y_gan = y * mask_float

            # ======================
            # UPDATE D
            # ======================
            if use_gan and use_gan_now:

                optimizer_D.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=device.type, enabled=CONFIG["amp"]):
                    # VARIABEL GAN
                    real_pred = D(x, y_gan)
                    fake_pred = D(x, pred_gan.detach()) 
                    d_loss = gan_loss.d_loss(real_pred, fake_pred)

                scaler.scale(d_loss).backward()
                scaler.step(optimizer_D)

            # ======================
            # UPDATE G
            # ======================
            optimizer.zero_grad(set_to_none=True)

            if use_gan and use_gan_now:
                
                with torch.amp.autocast(device_type=device.type, enabled=CONFIG["amp"]):
                    # VARIABEL GAN (Tanpa detach untuk update G)
                    fake_pred = D(x, pred_gan) 
                    gan_g_loss = gan_loss.g_loss(fake_pred)
                    total_loss = edm_loss + lambda_gan * gan_g_loss
            else:
                total_loss = edm_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += total_loss.item()
            if use_gan and use_gan_now:
                total_d_loss += d_loss.item()
                total_g_loss += gan_g_loss.item()
                gan_batches += 1

            valid_batches += 1

            #if use_gan_now:
            if use_gan and use_gan_now:
                pbar.set_postfix(
                    D=f"{d_loss.item():.4f}", 
                    G_adv=f"{gan_g_loss.item():.4f}",
                    EDM=f"{edm_loss.item():.4f}"
                )
            else:
                pbar.set_postfix(EDM=f"{edm_loss.item():.4f}")

        
        # Hitung rata-rata loss di akhir epoch
        train_loss /= (valid_batches + 1e-8)
        
        #epoch_d_loss = (total_d_loss / valid_batches) if (use_gan_now and valid_batches > 0) else 0.0
        #epoch_g_loss = (total_g_loss / valid_batches) if (use_gan_now and valid_batches > 0) else 0.0

        epoch_d_loss = (total_d_loss / gan_batches) if gan_batches > 0 else 0.0
        epoch_g_loss = (total_g_loss / gan_batches) if gan_batches > 0 else 0.0

        if use_gan_now and valid_batches > 0:
            print(f"D: {epoch_d_loss:.4f} | G_gan: {epoch_g_loss:.4f}")
        

        # =========================
        # Validation
        # =========================
        model.eval()

        val_psnr = 0
        valid_batches = 0
        val_lpips = 0.0

        with torch.no_grad():

            for x, y, _ in val_loader:

                x = x.to(device)
                y = y.to(device)

                mask = (y != 0).float()

                if mask.sum() == 0:
                    continue

                with torch.amp.autocast(device_type=device.type, enabled=CONFIG["amp"]):
                    output = model(x)

                if isinstance(output, tuple):
                    pred = output[0]
                else:
                    pred = output

                pred = pred.float()
                    
                val_psnr += psnr_roi(pred, y, mask).item()

                # 1. Pastikan rentang mentah benar-benar [0, 1] sebelum di-scale
                #pred_lp = torch.clamp(pred.clone(), min=0.0, max=1.0)
                #y_lp = torch.clamp(y.clone(), min=0.0, max=1.0)
                pred_lp = torch.clamp(pred, 0.0, 1.0)
                y_lp = torch.clamp(y, 0.0, 1.0)

                # 2. Scale ke [-1, 1] untuk LPIPS
                pred_lp = (pred_lp * 2.0) - 1.0
                y_lp = (y_lp * 2.0) - 1.0

                # 3. Hitung LPIPS (LPIPS akan mereplikasi channel 1->3 secara internal jika Anda memodifikasi lpips_model, tapi lebih aman pastikan input 3 channel)
                # Model LPIPS mengharapkan input 3 channel (RGB). Jika MRI Anda 1 channel, repeat di sini.
                if pred_lp.shape[1] == 1:
                    pred_lp = pred_lp.repeat(1, 3, 1, 1)
                    y_lp = y_lp.repeat(1, 3, 1, 1)

                val_lpips += lpips_model(pred_lp, y_lp).mean().item()
                
                valid_batches += 1

        val_psnr /= (valid_batches + 1e-8)
        val_lpips /= (valid_batches + 1e-8)

        #print(f"Epoch {epoch+1} | Train Loss {train_loss:.4f} | Val PSNR {val_psnr:.4f}")
        print(f"Epoch {epoch+1} | Train Loss {train_loss:.4f} | Val PSNR {val_psnr:.4f} | LPIPS {val_lpips:.4f}")

        score = val_psnr - 0.5 * val_lpips
        early_stop.step(score, epoch + 1)
        #early_stop.step(val_psnr, epoch + 1)

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
            "score": score
        })

        #if val_psnr > best_val_psnr:

        #    best_val_psnr = val_psnr
        if score > best_score:

            best_score = score

            torch.save(
                model.state_dict(),
                os.path.join(EXP_DIR, "best_model.pth")
            )

            if use_gan and use_gan_now:
                torch.save(D.state_dict(), os.path.join(EXP_DIR, "best_discriminator.pth"))

    pd.DataFrame(history).to_csv(
        os.path.join(EXP_DIR, "history.csv"),
        index=False
    )

    print("\nTraining completed\n")


if __name__ == "__main__":
    main()