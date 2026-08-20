CONFIG = {

    # =========================
    # Experiment
    # =========================
    "experiment_id": "EXP-DDRESUNET",
    "experiment_name": "OsmanTamam2023",
    "experiment_root": "experiments",


    # =========================
    # Reproducibility
    # =========================
    "seed": 42,

    # =========================
    # Dataset
    # =========================
    "dataset_root": r"./data/dataset_5fold_final_v4",
    "fold": 0,

    # =========================
    # Training
    # =========================
    "batch_size": 4,
    "epochs": 200,
    "lr": 1e-4,
    "weight_decay": 0.0,
    "optimizer": "Adam",

    # =========================
    # Early stopping
    # =========================
    # NOTE: original DD-Res U-Net paper did not report an early-stopping
    # patience; nilai ini disamakan dengan EXP-602 (25) untuk budget training
    # yang setara antar baseline yang dibandingkan. Sesuaikan kalau ada
    # justifikasi lain yang mau didokumentasikan di Methods.
    "early_stop_patience": 25,

    # =========================
    # Model
    # =========================
    "model": "DDResUNet",
    "base_channels": 16,  # sesuai init_filters=16 di kode TF asli (bukan 32 seperti EDMSynth)

    "modalities": ["T1", "T2", "FLAIR"],

    # =========================
    # Loss
    # =========================
    "loss": "DDResUNetLoss",  # L1 + 5*(1-SSIM), sesuai custom_loss paper asli

    # =========================
    # Engineering
    # =========================
    "amp": True,
    "num_workers": 4,
    "pin_memory": True
}
