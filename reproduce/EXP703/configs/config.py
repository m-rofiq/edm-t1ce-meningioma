CONFIG = {

    # =========================
    # Experiment
    # =========================
    "experiment_id": "EXP-703",
    "experiment_name": "EDM_GAN_VGG_LPIPS", 
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
    "lr_D": 5e-5,
    "lr": 1e-4,
    "weight_decay": 0.0,
    "optimizer": "Adam",

    # =========================
    # Early stopping
    # =========================
    "early_stop_patience": 40,

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
    "in_channels": 10,
    "base_channels": 64,
    "use_spectral_norm": True,
    },

    "gan": {
    "use_gan": True,
    "lambda_gan": 1.0,  #0.01 (tune 1), 0.5 (tune 2), 1.0 (tune 3 dan tune 4)
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
    "amp": False,
    "num_workers": 4,
    "pin_memory": True
}