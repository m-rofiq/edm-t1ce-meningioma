CONFIG = {

    # =========================
    # Experiment
    # =========================
    "experiment_id": "EXP-303", # "EXP-300", "EXP-301", "EXP-302", "EXP-303"
    "experiment_name": "T1_T2_FLAIR_TO_T1CE", #"T1_TO_T1CE", "T1_T2_TO_T1CE", "T1_FLAIR_TO_T1CE", "T1_T2_FLAIR_TO_T1CE"
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
    "epochs": 1,
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
    "model": "ResAttentionUNet", # "UNetBaseline", "ResUNet", "AttentionUNet", "ResAttentionUNet", "DenseUNet"
    "base_channels": 32,

    "modalities": ["T1","T2","FLAIR"], #  ["T1"], ["T1","T2"], ["T1","FLAIR"], ["T1","T2","FLAIR"]

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