CONFIG = {

    "experiment_id"  : "EXP-813",
    "experiment_name": "StratifiedEnh_Sharpness",
    "experiment_root": "experiments",

    "seed": 42,

    "dataset_root": r"./data/dataset_5fold_final_v4",
    "fold": 0,

    "batch_size"   : 4,
    "epochs"       : 200,
    "lr"           : 1e-4,
    "weight_decay" : 0.0,
    "optimizer"    : "Adam",

    "early_stop_patience": 45,

    "model"        : "EDMSynth",
    "base_channels": 32,

    "modalities": ["T1", "T2", "FLAIR"],

    "discriminator": {
        "name"             : "patchgan",
        "in_channels"      : 10,
        "base_channels"    : 64,
        "use_spectral_norm": True,
    },

    "gan": {
        "use_gan"        : False,
        "lambda_gan"     : 0.0,
        "gan_loss"       : "lsgan",
        "warmup_epochs"  : 35,
        "d_updates_per_g": 1,
        "g_loss_clip"    : 0.5,
    },

    "scheduler": {
        "use_scheduler": True,
        "type"         : "cosine",
        "eta_min"      : 2e-5,
    },

    "grad_clip": {
        "use_clip"       : True,
        "max_norm"       : 1.0,
        "warmup_norm"    : 0.5,
        "warmup_epochs"  : 10,
    },

    "optimizer_groups": {
        "use_groups"     : True,
        "projection_lr"  : 5e-5,
        "projection_keys": ["bridge", "skip_proj"],
    },

    "loss": "EDMLoss",

    # [NEW EXP-813] Stratified enhancement thresholds
    # Menggantikan flat threshold diff > 0.025
    # Tiga tier: weak / strong / very_strong enhancement
    "enhancement": {
        "weak_thr"        : 0.025,   # diff > ini   → weak enhancement
        "strong_thr"      : 0.10,    # diff > ini   → strong enhancement
        "very_strong_thr" : 0.25,    # diff > ini   → very strong (gadolinium peak)
        "w_weak"          : 1.0,     # bobot tier weak
        "w_strong"        : 3.0,     # bobot tier strong
        "w_very_strong"   : 5.0,     # bobot tier very_strong
    },

    # [NEW EXP-813] Sharpness loss config
    "sharpness": {
        "use_sharpness"   : True,
        "weight"          : 0.30,    # bobot dalam total loss
        "patch_size"      : 8,       # ukuran patch untuk local contrast
    },

    "amp"        : True,
    "num_workers": 2,
    "pin_memory" : True,

    "nan_guard": {
        "loss_upper_bound" : 50.0,
        "max_nan_per_epoch": 10,
    },
}
