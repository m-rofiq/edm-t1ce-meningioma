CONFIG = {

    # =========================
    # Experiment
    # =========================
    "experiment_id": "EXP-602",
    "experiment_name": "EDM", 
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
    
    #"model": "DMECNetStep1",
    #"model": "DMECNetStep2",
    #"model": "DMECNetStep3",
    #"model": "GuidedResAttentionUNet",
    "model": "EDMSynth",
    "base_channels": 32,

    "modalities": ["T1","T2","FLAIR"], #  ["T1"], ["T1","T2"], ["T1","FLAIR"], ["T1","T2","FLAIR"]

    "discriminator": {
    "name": "patchgan",
    #"in_channels": 4,
    "in_channels": 10,
    "base_channels": 64,
    "use_spectral_norm": True,
    },

    "gan": {
    "use_gan": True,
    "lambda_gan": 0.01,
    "gan_loss": "hinge",
    "d_updates_per_g": 1,
    "warmup_epochs": 20
    },

    # =========================
    # Loss
    # =========================
    #"loss": "MaskedL1", # "MaskedL1", "MaskedMSE", "Charbonnier", "EdgeLoss"
    "loss": "EDMLoss",
  

    # =========================
    # Engineering
    # =========================
    "amp": True,
    "num_workers": 4,
    "pin_memory": True
}