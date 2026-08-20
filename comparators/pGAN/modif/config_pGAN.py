# DESTINATION: configs/config_pGAN.py
#
# PENTING -- konvensi proyek Anda: dataset_2p5d.py melakukan
# `from configs.config import CONFIG` (path FIXED, bukan config_pGAN).
# Artinya sebelum menjalankan train_pGAN.py, file ini harus disalin/di-rename
# jadi configs/config.py terlebih dahulu -- SAMA PERSIS dengan cara Anda
# menjalankan config_exp602_EDM.py / config_exp703rerun3.py.
#
#   copy configs\config_pGAN.py configs\config.py   (Windows)
#   cp configs/config_pGAN.py configs/config.py     (Linux/WSL)
#
# Nilai default di bawah mengikuti strategi "faithful baseline + fair budget"
# (lihat strategi_adaptasi_pGAN_cGAN.md bagian 2e):
#   - lr, beta1, lambda_A, lambda_vgg, lambda_adv, warmup_epochs = default RESMI
#     pGAN (Dar et al.), TIDAK disamakan ke angka EDMSynth Anda.
#   - epochs, early_stop_patience, batch_size, seed, fold, dataset_root = SAMA
#     dgn EXP-703 (budget training & data split yang sebanding).

CONFIG = {

    # =========================
    # Experiment
    # =========================
    "experiment_id": "EXP-PGAN01",
    "experiment_name": "pGAN_baseline",
    "experiment_root": "experiments",

    # =========================
    # Reproducibility
    # =========================
    "seed": 42,

    # =========================
    # Dataset (SAMA dgn EXP-602/703 -- wajib utk perbandingan fold yang identik)
    # =========================
    "dataset_root": r"./data/dataset_5fold_final_v4",
    "fold": 0,
    "modalities": ["T1", "T2", "FLAIR"],  # -> input_nc = 3*len(modalities) = 9

    # =========================
    # Training budget (SAMA dgn EXP-703 -- fair training budget)
    # =========================
    "batch_size": 4,
    "epochs": 200,
    "early_stop_patience": 40,
    "num_workers": 4,
    "pin_memory": True,
    "amp": False,   # GAN training pGAN dimatikan AMP-nya, sama pertimbangan dgn EXP-703

    # =========================
    # Optimizer (default RESMI pGAN -- Adam lr=2e-4, beta1=0.5, TANPA lr_D terpisah)
    # =========================
    "lr": 0.0002,
    "lr_D": 0.0002,
    "beta1": 0.5,
    "weight_decay": 0.0,
    "optimizer": "Adam",

    # =========================
    # Model (generator)
    # =========================
    "model": "pGAN",
    "ngf": 64,          # default resmi pGAN (BUKAN base_channels=32 milik EDMSynth)
    "n_blocks": 9,       # jumlah ResNet block, hardcoded di kode resmi (define_G)
    "norm": "instance",
    "init_type": "normal",
    "final_activation": "identity",  # bukan "tanh" -- lihat catatan adaptasi 2c

    # =========================
    # Discriminator (PatchGAN, default resmi pGAN)
    # =========================
    "n_layers_D": 3,
    "discriminator": {
        "base_channels": 64,
        "use_sigmoid": False,   # LSGAN -> tanpa sigmoid (skor mentah)
    },

    # =========================
    # GAN (default resmi pGAN: aktif sejak epoch 1, TANPA warmup)
    # =========================
    "gan": {
        "use_gan": True,
        "lambda_adv": 1.0,
        "warmup_epochs": 0,
    },

    # =========================
    # Loss recon (default resmi demo command pGAN: lambda_A=100, lambda_vgg=100)
    # =========================
    "loss": {
        "lambda_A": 100.0,
        "lambda_vgg": 100.0,
    },

    # =========================
    # Kriteria early-stopping / model selection
    # "psnr_lpips" -> score = val_psnr - 0.5*val_lpips (SAMA dgn EXP-703)
    # "psnr"       -> score = val_psnr saja (SAMA dgn EXP-602)
    # Default: psnr_lpips, karena pGAN (GAN+perceptual) paling sebanding dgn EXP-703.
    # =========================
    "early_stop_metric": "psnr_lpips",
}
