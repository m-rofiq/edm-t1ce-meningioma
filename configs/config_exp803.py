CONFIG = {

    # =========================
    # Experiment
    # =========================
    "experiment_id": "EXP-803",
    "experiment_name": "SharpFreq_LSGAN_Iter1",
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
    "early_stop_patience": 25,

    # =========================
    # Model
    # =========================
    "model": "EDMSynth",          # sama dengan EXP-715A
    "base_channels": 32,

    "modalities": ["T1","T2","FLAIR"],

    "discriminator": {
        "name": "patchgan",
        "in_channels": 10,
        "base_channels": 64,
        "use_spectral_norm": True,
    },

    "gan": {
        "use_gan": True,

        # [CHANGED] 1.0 → 2.5
        # GAN signal harus cukup kuat untuk mendorong ketajaman
        # setelah pixel loss diturunkan ~40%
        "lambda_gan": 2.5,

        # [CHANGED] "hinge" → "lsgan"
        # LSGAN: gradient tidak saturate meski D sangat kuat,
        # sehingga signal ke G tetap informatif sepanjang training.
        # Gunakan lsgan_generator_loss / lsgan_discriminator_loss
        # dari loss_registry_exp803.py di trainer Anda.
        "gan_loss": "lsgan",

        "d_updates_per_g": 1,
        "warmup_epochs": 20,      # sama, beri waktu rekonstruksi dulu
    },

    # =========================
    # Loss
    # =========================
    # Gunakan loss_registry_exp803.py
    # Perubahan utama vs EXP-715A:
    #   - Pixel losses -40%
    #   - VGG: 0.08 → 0.25
    #   - Focal Frequency Loss (baru): 0.20
    #   - GAN: hinge → lsgan, lambda 1.0 → 2.5
    "loss": "EDMLoss",

    # =========================
    # Engineering
    # =========================
    "amp": True,
    "num_workers": 2,
    "pin_memory": True,
}
