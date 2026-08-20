CONFIG = {

    "experiment_id"  : "EXP-806",
    "experiment_name": "GroupNorm_CosineLR_Iter4",
    "experiment_root": "experiments",

    "seed": 42,

    "dataset_root": r"./data/dataset_5fold_final_v4",
    "fold": 0,

    "batch_size"   : 4,
    "epochs"       : 200,
    "lr"           : 1e-4,       # peak lr — cosine akan turunkan ke eta_min
    "weight_decay" : 0.0,
    "optimizer"    : "Adam",

    "early_stop_patience": 40,   # lebih panjang — beri ruang cosine LR bekerja

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

    # [NEW] scheduler config
    "scheduler": {
        "use_scheduler": True,
        "type"         : "cosine",   # CosineAnnealingLR
        "eta_min"      : 2e-5,       # lr floor di akhir training
    },

    # [NEW] gradient clipping
    "grad_clip": {
        "use_clip" : True,
        "max_norm" : 1.0,
    },

    "loss": "EDMLoss",

    "amp"        : True,
    "num_workers": 2,
    "pin_memory" : True,
}
