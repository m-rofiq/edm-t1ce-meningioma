CONFIG = {

    # =========================
    # Experiment
    # =========================
    "experiment_id": "EXP-100", #"EXP-100"
    "experiment_name": "UNET_L1", # "UNET_L1"
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
    "model": "UNetBaseline", # "UNetBaseline", "ResUNet", "AttentionUNet", "ResAttentionUNet", "DenseUNet"
    "base_channels": 32,
    "modalities": ["T1"],

    # =========================
    # Loss
    # =========================
    "loss": "MaskedL1", # "MaskedL1", "MaskedMSE", "Charbonnier", "EdgeLoss"
    #"loss": ["MaskedL1", "Perceptual", "EdgeLoss"]
    #"loss_weights": {
    #    "MaskedL1": 1.0,
    #    "Perceptual": 0.1,
    #    "EdgeLoss": 0.2
    #}

    # =========================
    # Engineering
    # =========================
    "amp": True,
    "num_workers": 4,
    "pin_memory": True
}