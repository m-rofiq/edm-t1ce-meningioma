CONFIG = {

    "experiment_id"  : "EXP-815",
    "experiment_name": "EnhSegHead_EagerInit_FastTrain",
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
        "projection_keys": ["bridge", "skip_proj", "enh_head", "enh_gate"],
    },

    "loss": "EDMLoss",

    "enhancement": {
        "weak_thr"        : 0.025,
        "strong_thr"      : 0.10,
        "very_strong_thr" : 0.25,
        "w_weak"          : 1.0,
        "w_strong"        : 3.0,
        "w_very_strong"   : 5.0,
    },

    "sharpness": {
        "use_sharpness"   : True,
        "weight"          : 0.30,
        "patch_size"      : 8,
    },

    "enh_seg": {
        "use_enh_seg"    : True,
        "seg_loss_weight": 1.50,
        "gate_weight"    : 0.70,
        "enh_threshold"  : 0.025,
    },

    "amp"        : True,
    "num_workers": 2,
    "pin_memory" : True,

    "nan_guard": {
        "loss_upper_bound" : 50.0,
        "max_nan_per_epoch": 10,
    },
}
