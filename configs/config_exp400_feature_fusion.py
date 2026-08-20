CONFIG = {

    # =========================
    # Experiment
    # =========================
    "experiment_id": "EXP-401", 
    # "EXP-400","EXP-401","EXP-402","EXP-403","EXP-404","EXP-405","EXP-406","EXP-407","EXP-408" 
    "experiment_name": "MULTI_ENCODER_STEP1_T1_T2", 
    # "MULTI_ENCODER_STEP1_T1","MULTI_ENCODER_STEP1_T1_T2","MULTI_ENCODER_STEP1_T1_T2_F", 
    # "MULTI_ENCODER_STEP2_T2","MULTI_ENCODER_STEP2_T1_T2","MULTI_ENCODER_STEP2_T1_T2_F", 
    # "MULTI_ENCODER_STEP3_T1","MULTI_ENCODER_STEP3_T1_T2","MULTI_ENCODER_STEP3_T1_T2_F", 
    "experiment_root": "experiments",
   

    # =========================
    # Reproducibility
    # =========================
    "seed": 42,

    # =========================
    # Dataset
    # =========================
    "dataset_root": r"./data/dataset_5fold_final_v4",
    "fold": 4,

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
    
    "model": "DMECNetStep1",    #"EXP-400","EXP-401","EXP-402"
    #"model": "DMECNetStep2",   #"EXP-403","EXP-404","EXP-405"
    #"model": "DMECNetStep3",   #"EXP-406","EXP-407","EXP-408" 
    "base_channels": 32,

    "modalities": ["T1","T2"], 
    # ["T1"],                  "EXP-400","EXP-401","EXP-402"
    # ["T1","T2"],              "EXP-403","EXP-404","EXP-405"
    # ["T1","T2","FLAIR"]       "EXP-406","EXP-407","EXP-408" 

    # =========================
    # Loss
    # =========================
    "loss": "MaskedL1", # "MaskedL1", "MaskedMSE", "Charbonnier", "EdgeLoss"
  

    # =========================
    # Engineering
    # =========================
    "amp": True,
    "num_workers": 4,
    "pin_memory": True
}