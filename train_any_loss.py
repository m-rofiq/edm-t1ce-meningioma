import torch
import random
import numpy as np
import os
import pandas as pd
import torch.nn.functional as F

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
    # Model
    # =========================
    model = MODEL_REGISTRY[CONFIG["model"]](
        base_channels=CONFIG["base_channels"]
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

                pred = pred.float() # Cast kembali ke FP32 untuk presisi MSE
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
    main()