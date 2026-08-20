# Training script untuk DD-Res U-Net SOTA baseline.
#
# INI PRAKTIS SALINAN train_edm_exp602.py (infrastruktur eksperimen Anda),
# BUKAN pipeline baru -- sesuai kesepakatan: yang diadaptasi hanya model,
# loss, dan config; data loader (MRI2p5DDataset), evaluator (psnr_roi),
# prepare_experiment, EarlyStopping SEMUA dipakai identik dari codebase Anda.
#
# PERBEDAAN vs train_edm_exp602.py:
#   1. Tidak ada percabangan DMECNetStep1/2/3 (bukan model itu) -- langsung
#      masuk ke branch else generik: MODEL_REGISTRY[model_name](base_channels=...,
#      in_channels=...), persis pola yang sudah ada di kode Anda.
#   2. loss_fn dipanggil sebagai loss_fn(pred, y, mask) -- TIDAK ada cabang
#      "isinstance(output, tuple)" karena DDResUNet menghasilkan satu tensor
#      output tunggal (bukan tuple base/enh seperti EDMSynth).
#
# CARA PAKAI:
#   1. Salin file ini ke root project Anda (folder yang sama dengan
#      train_edm_exp602.py), sebagai train_ddresunet_sota.py
#   2. Salin ddresunet_pytorch.py ke models/ddresunet.py
#   3. Salin ddresunet_loss.py ke losses/ddresunet_loss.py
#   4. Salin config_ddresunet_sota.py ke configs/ (atau timpa configs/config.py
#      sesuai fold yang mau dijalankan -- sama seperti pola exp602/703 Anda)
#   5. Tambahkan 2 baris registrasi (lihat instruksi di akhir masing-masing file)

import torch
import random
import numpy as np
import os
import pandas as pd

from experiment_utils import prepare_experiment

from torch.utils.data import DataLoader
from dataset_2p5d import MRI2p5DDataset
from psnr_fixed_range import psnr_roi
from config_ddresunet_sota import CONFIG
from tqdm import tqdm
from model_registry import MODEL_REGISTRY
from loss_registry import LOSS_REGISTRY
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


def main():

    check_experiment_folder(CONFIG)
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    random.seed(CONFIG["seed"])

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    # Model (DDResUNet -- selalu branch generik, bukan DMECNetStep*)
    # =========================
    model_name = CONFIG["model"]
    model = MODEL_REGISTRY[model_name](
        base_channels=CONFIG["base_channels"],
        in_channels=3 * len(CONFIG["modalities"])
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )

    loss_fn = LOSS_REGISTRY[CONFIG["loss"]]

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda" and CONFIG["amp"]))

    early_stop = EarlyStopping(CONFIG["early_stop_patience"])

    best_val_psnr = -float("inf")
    history = []

    for epoch in range(CONFIG["epochs"]):

        model.train()
        train_loss = 0
        valid_batches = 0

        for x, y, _ in tqdm(train_loader, ncols=80, desc=f"Epoch {epoch+1}"):

            x = x.to(device)
            y = y.to(device)

            mask = (y != 0)

            if mask.sum() == 0:
                continue

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=CONFIG["amp"]):
                pred = model(x)
                loss = loss_fn(pred, y, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            valid_batches += 1

        train_loss /= (valid_batches + 1e-8)

        # =========================
        # Validation
        # =========================
        model.eval()

        val_psnr = 0
        valid_batches = 0

        with torch.no_grad():
            for x, y, _ in val_loader:

                x = x.to(device)
                y = y.to(device)

                mask = (y != 0).float()

                if mask.sum() == 0:
                    continue

                with torch.amp.autocast(device_type=device.type, enabled=CONFIG["amp"]):
                    pred = model(x)

                pred = pred.float()
                val_psnr += psnr_roi(pred, y, mask).item()
                valid_batches += 1

        val_psnr /= (valid_batches + 1e-8)

        print(f"Epoch {epoch+1} | Train Loss {train_loss:.4f} | Val PSNR {val_psnr:.4f}")

        early_stop.step(val_psnr, epoch + 1)

        if early_stop.stop:
            print(f"\nEarly stopping triggered at epoch {early_stop.stop_epoch}\n")
            break

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_psnr": val_psnr
        })

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(
                model.state_dict(),
                os.path.join(EXP_DIR, "best_model.pth")
            )

    pd.DataFrame(history).to_csv(
        os.path.join(EXP_DIR, "history.csv"),
        index=False
    )

    print("\nTraining completed\n")


if __name__ == "__main__":
    import sys, json

    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            new_cfg = json.load(f)
        CONFIG.clear()
        CONFIG.update(new_cfg)

    main()