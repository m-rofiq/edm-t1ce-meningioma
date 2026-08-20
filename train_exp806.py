import torch
import random
import numpy as np
import os
import pandas as pd
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


def seed_worker(worker_id):
    worker_seed = CONFIG["seed"] + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def check_experiment_folder(CONFIG):
    exp_name = f"{CONFIG['experiment_id']}_{CONFIG['experiment_name']}_fold{CONFIG['fold']}"
    exp_dir  = os.path.join(CONFIG["experiment_root"], exp_name)
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
    torch.backends.cudnn.benchmark     = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EXP_DIR = prepare_experiment(CONFIG)
    print(f"\nRunning experiment folder:\n{EXP_DIR}\n")

    # =========================
    # Dataset
    # =========================
    DATASET_ROOT = CONFIG["dataset_root"]
    FOLD         = CONFIG["fold"]

    train_loader = DataLoader(
        MRI2p5DDataset(os.path.join(DATASET_ROOT, f"fold_{FOLD}", "train")),
        batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=CONFIG["num_workers"], pin_memory=CONFIG["pin_memory"],
        worker_init_fn=seed_worker, persistent_workers=True
    )
    val_loader = DataLoader(
        MRI2p5DDataset(os.path.join(DATASET_ROOT, f"fold_{FOLD}", "val")),
        batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=CONFIG["num_workers"], pin_memory=CONFIG["pin_memory"],
        worker_init_fn=seed_worker, persistent_workers=True
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
            base_channels = CONFIG["base_channels"],
            in_channels   = 3 * len(CONFIG["modalities"])
        ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), betas=(0.5, 0.999),
        lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
    )

    # =========================
    # [NEW] Cosine LR Scheduler
    # Menurunkan lr dari 1e-4 → 2e-5 secara cosine
    # selama T_max epoch. Mengatasi masalah training
    # stagnan setelah epoch 25 pada EXP-805.
    # =========================
    sched_cfg      = CONFIG["scheduler"]
    use_scheduler  = sched_cfg["use_scheduler"]

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max   = CONFIG["epochs"],
            eta_min = sched_cfg["eta_min"]
        )

    # =========================
    # [NEW] Gradient Clipping config
    # Mencegah spike loss dari FFL pada batch "sulit".
    # =========================
    clip_cfg  = CONFIG["grad_clip"]
    use_clip  = clip_cfg["use_clip"]
    max_norm  = clip_cfg["max_norm"]

    # =========================
    # LPIPS
    # =========================
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()

    use_gan    = CONFIG["gan"]["use_gan"]   # False
    loss_fn    = LOSS_REGISTRY[CONFIG["loss"]]
    scaler     = torch.amp.GradScaler(enabled=(device.type == "cuda" and CONFIG["amp"]))
    early_stop = EarlyStopping(CONFIG["early_stop_patience"])

    best_val_psnr = -float("inf")
    history       = []

    for epoch in range(CONFIG["epochs"]):

        model.train()
        train_loss    = 0.0
        valid_batches = 0

        current_lr = optimizer.param_groups[0]['lr']
        pbar = tqdm(train_loader, ncols=110, desc=f"Epoch {epoch+1} | lr={current_lr:.2e}")

        for x, y, _ in pbar:

            x    = x.to(device)
            y    = y.to(device)
            mask = (y != 0)

            if mask.sum() == 0:
                continue

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=CONFIG["amp"]):
                output = model(x)
                if isinstance(output, tuple):
                    pred, base, enh = output
                    loss = loss_fn(output, y, mask, x)
                else:
                    pred = output
                    loss = loss_fn(pred, y, mask)

            scaler.scale(loss).backward()

            # [NEW] gradient clipping — unscale dulu sebelum clip
            if use_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

            scaler.step(optimizer)
            scaler.update()

            train_loss    += loss.item()
            valid_batches += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")

        train_loss /= (valid_batches + 1e-8)

        # =========================
        # [NEW] Step scheduler setelah setiap epoch
        # =========================
        if use_scheduler:
            scheduler.step()

        # =========================
        # Validation
        # =========================
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

                pred_lp = (torch.clamp(pred, 0.0, 1.0) * 2.0) - 1.0
                y_lp    = (torch.clamp(y,    0.0, 1.0) * 2.0) - 1.0

                if pred_lp.shape[1] == 1:
                    pred_lp = pred_lp.repeat(1, 3, 1, 1)
                    y_lp    = y_lp.repeat(1,   3, 1, 1)

                val_lpips     += lpips_model(pred_lp, y_lp).mean().item()
                valid_batches += 1

        val_psnr  /= (valid_batches + 1e-8)
        val_lpips /= (valid_batches + 1e-8)

        print(
            f"Epoch {epoch+1:>3} | "
            f"lr {current_lr:.2e} | "
            f"Train Loss {train_loss:.4f} | "
            f"Val PSNR {val_psnr:.4f} | "
            f"LPIPS {val_lpips:.4f}"
        )

        early_stop.step(val_psnr, epoch + 1)
        if early_stop.stop:
            print(f"\nEarly stopping triggered at epoch {early_stop.stop_epoch}\n")
            break

        history.append({
            "epoch"      : epoch + 1,
            "lr"         : current_lr,
            "train_loss" : train_loss,
            "val_psnr"   : val_psnr,
            "val_lpips"  : val_lpips,
            "d_loss"     : 0.0,
            "g_loss"     : 0.0,
        })

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(model.state_dict(), os.path.join(EXP_DIR, "best_model.pth"))

    pd.DataFrame(history).to_csv(
        os.path.join(EXP_DIR, "history.csv"), index=False
    )
    print("\nTraining completed\n")


if __name__ == "__main__":
    main()
