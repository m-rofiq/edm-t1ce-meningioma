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
from losses.loss_registry import build_loss, VGGPerceptual, FocalFrequencyLoss
from utils.early_stopping import EarlyStopping

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def seed_worker(worker_id):
    worker_seed = CONFIG["seed"] + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def check_experiment_folder(CONFIG):

    exp_name = (
        f"{CONFIG['experiment_id']}_"
        f"{CONFIG['experiment_name']}_"
        f"fold{CONFIG['fold']}"
    )

    exp_dir = os.path.join(CONFIG["experiment_root"], exp_name)

    resume_ckpt = os.path.join(exp_dir, "resume_checkpoint.pth")

    # folder belum ada → training baru
    if not os.path.exists(exp_dir):
        return False, exp_dir, resume_ckpt

    # folder ada + checkpoint ada → resume
    if os.path.exists(resume_ckpt):

        print(f"\n[RESUME MODE]")
        print(f"Experiment folder found:")
        print(exp_dir)

        return True, exp_dir, resume_ckpt

    # folder ada tapi tidak ada checkpoint
    raise RuntimeError(
        f"\nExperiment folder already exists "
        f"without valid checkpoint:\n{exp_dir}\n"
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
    print(f"  projection+enh layers : {len(proj_params)} params | lr={proj_lr}")
    print(f"  other layers          : {len(other_params)} params | lr={config['lr']}\n")

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
    
    if (
        not isinstance(loss, torch.Tensor)
        or not loss.requires_grad
        or loss.grad_fn is None
    ):
        print(f"\n  [Detached Loss] epoch {epoch+1}")

        optimizer.zero_grad(set_to_none=True)

        torch.cuda.empty_cache()

        return False, float(loss) if torch.is_tensor(loss) else loss

    try:
        scaler.scale(loss).backward()
    except torch.OutOfMemoryError:
        print(f"\n  [OOM] during backward epoch {epoch+1} — clearing cache")
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return False, loss_val
    except Exception as e:
    #except RuntimeError as e:
        msg = str(e).lower()
        #if "nan" in msg or "inf" in msg or "out of memory" in msg:
        if (
            "nan" in msg
            or "inf" in msg
            or "out of memory" in msg
            or "does not require grad" in msg
            or "grad_fn" in msg
        ):
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

def save_resume_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    best_val_psnr,
    history,
):

    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "scaler_state": scaler.state_dict(),
        "best_val_psnr": best_val_psnr,
        "history": history,
    }

    torch.save(checkpoint, path)

def main():

    resume_mode, EXP_DIR_EXISTING, RESUME_CKPT = (
        check_experiment_folder(CONFIG)
    )
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    random.seed(CONFIG["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if resume_mode:
        EXP_DIR = EXP_DIR_EXISTING
    else:
        EXP_DIR = prepare_experiment(CONFIG)

    gc      = CONFIG["grad_clip"]
    ng      = CONFIG["nan_guard"]
    seg_cfg = CONFIG["enh_seg"]

    print(f"\nRunning : {EXP_DIR}")
    print(f"Model   : {CONFIG['model']} | EnhSegHead=ON | EnhGate=ON | EagerInit=ON")
    print(f"Sched   : cosine {CONFIG['lr']} → {CONFIG['scheduler']['eta_min']}")
    print(f"Clip    : warmup {gc['warmup_norm']} (ep1-{gc['warmup_epochs']}) → {gc['max_norm']}")
    print(f"Patience: {CONFIG['early_stop_patience']} epochs")
    print(f"NaNGuard: loss_upper={ng['loss_upper_bound']} | max_nan/ep={ng['max_nan_per_epoch']}")
    print(f"EnhSeg  : seg_w={seg_cfg['seg_loss_weight']} | gate_w={seg_cfg['gate_weight']}\n")

    DATASET_ROOT = CONFIG["dataset_root"]
    FOLD         = CONFIG["fold"]

    train_loader = DataLoader(
        MRI2p5DDataset(os.path.join(DATASET_ROOT, f"fold_{FOLD}", "train")),
        batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=CONFIG["num_workers"], pin_memory=CONFIG["pin_memory"],
        worker_init_fn=seed_worker, persistent_workers=False
    )
    val_loader = DataLoader(
        MRI2p5DDataset(os.path.join(DATASET_ROOT, f"fold_{FOLD}", "val")),
        batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=CONFIG["num_workers"], pin_memory=CONFIG["pin_memory"],
        worker_init_fn=seed_worker, persistent_workers=True
    )

    model = MODEL_REGISTRY[CONFIG["model"]](
        base_channels = CONFIG["base_channels"],
        in_channels   = 3 * len(CONFIG["modalities"]),
        gate_weight   = seg_cfg["gate_weight"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # ===========================================================
    # [FIX EXP-816] EAGER PRE-INITIALIZATION
    # ===========================================================
    # Semua model berat diinisialisasi DI SINI, sebelum loop
    # epoch dimulai. Ini menghilangkan lazy init yang menjadi
    # penyebab slowdown 10–20x di EXP-814 (epoch 19+).
    #
    # Urutan init penting: VRAM harus cukup sebelum training
    # model utama dimulai. Karena itu init dilakukan setelah
    # model utama di-load ke GPU agar kita tahu sisa VRAM.
    # ===========================================================
    print("\n[EagerInit] Initializing auxiliary models before training loop...")

    print("  → VGGPerceptual (VGG16 features)...", end=" ", flush=True)
    perceptual_model = VGGPerceptual().to(device).eval()
    # Warmup forward pass — supaya CUDA kernel sudah terkompilasi
    # sebelum epoch 1 dimulai. Ini menghilangkan spike di batch pertama.
    with torch.no_grad():
        _dummy = torch.zeros(1, 1, 64, 64, device=device)
        perceptual_model(_dummy, _dummy)
    print("OK")

    print("  → FocalFrequencyLoss...", end=" ", flush=True)
    freq_loss_model = FocalFrequencyLoss(
        loss_weight=1.0, alpha=1.5, patch_factor=1,
        ave_spectrum=False, log_matrix=False, batch_matrix=False
    ).to(device)
    # Warmup
    with torch.no_grad():
        _dummy2 = torch.zeros(1, 1, 64, 64, device=device)
        freq_loss_model(_dummy2, _dummy2)
    print("OK")

    print("  → LPIPS (AlexNet)...", end=" ", flush=True)
    lpips_model = lpips.LPIPS(net='alex').to(device).eval()
    with torch.no_grad():
        _dummy3 = torch.zeros(1, 3, 64, 64, device=device)
        lpips_model(_dummy3, _dummy3)
    print("OK")

    print("[EagerInit] Done — semua model siap, tidak ada lazy init di training loop\n")

    # Build loss dan inject model yang sudah diinit
    loss_fn = build_loss(CONFIG)
    loss_fn.inject_models(
        perceptual = perceptual_model,
        freq_loss  = freq_loss_model,
    )

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

    scaler     = torch.amp.GradScaler(enabled=(device.type == "cuda" and CONFIG["amp"]))
    early_stop = EarlyStopping(CONFIG["early_stop_patience"])

    best_val_psnr = -float("inf")
    history       = []
    start_epoch = 0

    if resume_mode:

        print("\nLoading resume checkpoint...")

        ckpt = torch.load(RESUME_CKPT, map_location=device)

        model.load_state_dict(ckpt["model_state"])

        optimizer.load_state_dict(ckpt["optimizer_state"])

        if use_scheduler and ckpt["scheduler_state"] is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])

        scaler.load_state_dict(ckpt["scaler_state"])

        best_val_psnr = ckpt["best_val_psnr"]

        history = ckpt["history"]

        start_epoch = ckpt["epoch"] + 1

        print(f"Resuming from epoch {start_epoch}")

    

    for epoch in range(start_epoch, CONFIG["epochs"]):

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
                loss   = loss_fn(output, y, mask, x)

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
            print(f"Epoch {epoch+1}: no valid batches, skipping")
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

                #pred = output[0].float()
                if isinstance(output, tuple):
                    pred = output[0].float()
                else:
                    pred = output.float()

                if not torch.isfinite(pred).all():
                    continue

                val_psnr += psnr_roi(pred, y, mask).item()

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

        # [FIX] Simpan CSV setiap epoch — aman di-interrupt kapan saja
        pd.DataFrame(history).to_csv(
            os.path.join(EXP_DIR, "history.csv"), index=False
        )

        save_resume_checkpoint(
            RESUME_CKPT,
            epoch,
            model,
            optimizer,
            scheduler if use_scheduler else None,
            scaler,
            best_val_psnr,
            history,
        )

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(model.state_dict(), os.path.join(EXP_DIR, "best_model.pth"))

    print("\nTraining completed\n")


if __name__ == "__main__":
    main()
