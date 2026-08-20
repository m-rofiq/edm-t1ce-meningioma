import copy
import subprocess
import sys
import os
import json

from configs.config import CONFIG


# =========================
# PATH SETUP (RELATIVE)
# =========================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

TRAIN_SCRIPT  = os.path.join(PROJECT_ROOT, "train_pGAN.py")
EVAL_SCRIPT   = os.path.join(PROJECT_ROOT, "evaluators", "evaluate_experiment_paper.py")
SELECT_SCRIPT = os.path.join(PROJECT_ROOT, "evaluators", "select_visualization_cases.py")
VIS_SCRIPT    = os.path.join(PROJECT_ROOT, "inference_visualize_pGAN.py")

TEMP_CONFIG = os.path.abspath(
    os.path.join(PROJECT_ROOT, "configs", "temp_config.json")
)


# =========================
# DEFINE EXPERIMENT
# =========================

EXPERIMENTS = [
    {"experiment_id": "EXP-PGAN01", "experiment_name": "pGAN_baseline", "fold": i,
     "model": "pGAN", "modalities": ["T1", "T2", "FLAIR"]}
    for i in range(1, 5)
]
 

# =========================
# RUN COMMAND WITH LIVE OUTPUT
# =========================

def run_command(cmd):

    process = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        text=True
    )

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(
            f"\nCommand failed:\n{' '.join(cmd)}\n"
        )

# =========================
# RUN SINGLE EXPERIMENT
# =========================

def run_experiment(exp_cfg):

    cfg = copy.deepcopy(CONFIG)

    cfg["experiment_id"]   = exp_cfg["experiment_id"]
    cfg["experiment_name"] = exp_cfg["experiment_name"]
    cfg["fold"]            = exp_cfg["fold"]
    cfg["model"]           = exp_cfg["model"]
    cfg["modalities"]      = exp_cfg["modalities"]

    with open(TEMP_CONFIG, "w") as f:
        json.dump(cfg, f, indent=4)

    print(f"\n🚀 RUN: {cfg['experiment_id']} | Fold {cfg['fold']} | {cfg['model']} | {cfg['modalities']}\n")

    # =========================
    # 1. TRAIN (LIVE OUTPUT)
    # =========================
    run_command([
        sys.executable,
        TRAIN_SCRIPT,
        TEMP_CONFIG
    ])

    # =========================
    # CHECK MODEL EXISTS
    # =========================
    exp_dir = os.path.join(
        CONFIG["experiment_root"],
        f"{cfg['experiment_id']}_{cfg['experiment_name']}_fold{cfg['fold']}"
    )

    model_path = os.path.join(exp_dir, "best_model.pth")

    if not os.path.exists(model_path):
        print("\n⚠️ WARNING: best_model.pth tidak ditemukan → SKIP evaluation\n")
        return

    # =========================
    # 2. EVALUATE
    # =========================
    run_command([
        sys.executable,
        EVAL_SCRIPT,
        TEMP_CONFIG
    ])

    # =========================
    # 3. SELECT VIS
    # =========================
    run_command([
        sys.executable,
        SELECT_SCRIPT,
        TEMP_CONFIG
    ])

    # =========================
    # 4. VISUALIZE
    # =========================
    run_command([
        sys.executable,
        VIS_SCRIPT,
        TEMP_CONFIG
    ])

    print(f"\n✅ DONE: {cfg['experiment_id']} Fold {cfg['fold']}\n")


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    for exp in EXPERIMENTS:
        run_experiment(exp)