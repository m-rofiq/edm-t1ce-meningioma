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
from losses.loss_registry import build_loss   # [NEW] factory, bukan singleton
from utils.early_stopping import EarlyStopping

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


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


def build_optimizer(model, config):
    og_cfg  = config.get("optimizer_groups", {})
    use_grp = og_cfg.get("use_groups", False)

    if not use_grp:
        return torch.optim.Adam(
            model.parameters(), betas=(0.5, 0.999),
            lr=config["lr"], weight_decay=config["weight_decay"]
        )

    proj_lr   = og_cfg["projection_lr"]
    proj_keys = og_cfg["projection_keys"]
    proj_params, other_params = [], []

    for name, param in model.named_parameters():
        if any(k in name for k in proj_keys):
            proj_params.append(param)
        else:
            other_params.append(param)

    print(f"Optimizer groups:")
    print(f"  projection layers : {len(proj_params)} params | lr={proj_lr}")
    print(f"  other layers      : {len(other_params)} params | lr={config['lr']}\n")

    return torch.optim.Adam(
        [
            {"params": other_params, "lr": config["lr"]},
            {"params": proj_params,  "lr": proj_lr},
        ],
        betas=(0.5, 0.999),
        weight_decay=config["weight_decay"]
    )


def safe_backward(loss, scaler, optimizer, use_clip, current_clip, model,
                  epoch, nan_guard_cfg):
    loss_val  = loss.item()
    upper_bnd = nan_guard_cfg.get("loss_upper_bound", 50.0)

    if not (torch.isfinite(loss) and loss_val < upper_bnd):
        reason = "NaN/Inf" if not torch.isfinite(loss) else f"loss>{upper_bnd:.0f}"
        print(f"\n  [SKIP] {reason} loss={loss_val:.4f} at epoch {epoch+1}")
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return False, loss_val

    try:
        scaler.scale(loss).backward()
    except torch.OutOfMemoryError:
        print(f"\n  [OOM] during backward epoch {epoch+1} — clearing cache")
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return False, loss_val
    except RuntimeError as e:
        msg = str(e).lower()
        if "nan" in msg or "inf" in msg or "out of memory" in msg:
            print(f"\n  [RuntimeError] {str(e)[:80]} — skipping batch")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return False, loss_val
        raise

    if use_clip:
        scaler.unscale_(optimizer)
        has_nan_grad = any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        if has_nan_grad:
            print(f"\n  [NaN Grad] epoch {epoch+1}, skipping batch")
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            torch.cuda.empty_cache()
            return False, loss_val
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=current_clip)

    scaler.step(optimizer)
    scaler.update()
    return True, loss_val


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

    enh_cfg = CONFIG["enhancement"]
    shr_cfg = CONFIG["sharpness"]
    gc      = CONFIG["grad_clip"]
    ng      = CONFIG["nan_guard"]

    print(f"\nRunning : {EXP_DIR}")
    print(f"Model   : {CONFIG['model']} | enc_ch=48 | dec_ch=32 | CBAM=ON | bridge=2-stage")
    print(f"Sched   : cosine {CONFIG['lr']} → {CONFIG['scheduler']['eta_min']}")
    print(f"Clip    : warmup {gc['warmup_norm']} (ep1-{gc['warmup_epochs']}) → {gc['max_norm']}")
    print(f"Patience: {CONFIG['early_stop_patience']} epochs")
    print(f"NaNGuard: loss_upper={ng['loss_upper_bound']} | max_nan/epoch={ng['max_nan_per_epoch']}")
    print(f"Enh tier: weak>{enh_cfg['weak_thr']} w={enh_cfg['w_weak']} | "
          f"strong>{enh_cfg['strong_thr']} w={enh_cfg['w_strong']} | "
          f"vstrong>{enh_cfg['very_strong_thr']} w={enh_cfg['w_very_strong']}")
    print(f"Sharpness: use={shr_cfg['use_sharpness']} | "
          f"weight={shr_cfg['weight']} | patch={shr_cfg['patch_size']}\n")

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

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}\n")

    optimizer = build_optimizer(model, CONFIG)

    sched_cfg     = CONFIG["scheduler"]
    use_scheduler = sched_cfg["use_scheduler"]
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONFIG["epochs"], eta_min=sched_cfg["eta_min"]
        )

    gc_cfg        = CONFIG["grad_clip"]
    use_clip      = gc_cfg["use_clip"]
    max_norm      = gc_cfg["max_norm"]
    warmup_norm   = gc_cfg["warmup_norm"]
    warmup_epochs = gc_cfg["warmup_epochs"]

    nan_guard_cfg = CONFIG["nan_guard"]
    max_nan_epoch = nan_guard_cfg["max_nan_per_epoch"]

    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()

    # [NEW EXP-813] Loss diinstansiasi dari CONFIG — bukan singleton global
    loss_fn = build_loss(CONFIG)

    scaler     = torch.amp.GradScaler(enabled=(device.type == "cuda" and CONFIG["amp"]))
    early_stop = EarlyStopping(CONFIG["early_stop_patience"])

    best_val_psnr = -float("inf")
    history       = []

    for epoch in range(CONFIG["epochs"]):

        model.train()
        train_loss    = 0.0
        valid_batches = 0
        nan_count     = 0
        current_lr    = optimizer.param_groups[0]['lr']
        current_clip  = warmup_norm if epoch < warmup_epochs else max_norm

        pbar = tqdm(train_loader, ncols=125,
                    desc=f"Epoch {epoch+1} | lr={current_lr:.2e} | clip={current_clip}")

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

            ok, loss_val = safe_backward(
                loss, scaler, optimizer, use_clip, current_clip, model,
                epoch, nan_guard_cfg
            )

            if not ok:
                nan_count += 1
                if nan_count >= max_nan_epoch:
                    print(f"\n  [ABORT EPOCH] {nan_count} NaN at epoch {epoch+1}\n")
                    break
                continue

            train_loss    += loss_val
            valid_batches += 1
            pbar.set_postfix(loss=f"{loss_val:.4f}", nan=nan_count)

        if valid_batches == 0:
            print(f"Epoch {epoch+1}: no valid batches (nan={nan_count}), skipping")
            if use_scheduler:
                scheduler.step()
            continue

        train_loss /= valid_batches

        if use_scheduler:
            scheduler.step()

        if nan_count > 0:
            print(f"  [INFO] Epoch {epoch+1}: {nan_count} NaN skipped "
                  f"(of {valid_batches + nan_count} total)")

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

                if not torch.isfinite(pred).all():
                    continue

                val_psnr  += psnr_roi(pred, y, mask).item()

                pred_lp = (torch.clamp(pred, 0.0, 1.0) * 2.0) - 1.0
                y_lp    = (torch.clamp(y,    0.0, 1.0) * 2.0) - 1.0
                if pred_lp.shape[1] == 1:
                    pred_lp = pred_lp.repeat(1, 3, 1, 1)
                    y_lp    = y_lp.repeat(1,   3, 1, 1)

                val_lpips     += lpips_model(pred_lp, y_lp).mean().item()
                valid_batches += 1

        if valid_batches == 0:
            print(f"Epoch {epoch+1}: validation no valid outputs")
            continue

        val_psnr  /= valid_batches
        val_lpips /= valid_batches

        print(
            f"Epoch {epoch+1:>3} | "
            f"lr {current_lr:.2e} | "
            f"Loss {train_loss:.4f} | "
            f"PSNR {val_psnr:.4f} | "
            f"LPIPS {val_lpips:.4f} | "
            f"NaN {nan_count}"
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
            "nan_batches": nan_count,
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
