CONFIG = {

    "experiment_id": "SANITY-002",
    "experiment_name": "DATASET_IDENTITY",
    "experiment_root": "experiments",

    "seed": 42,

    "dataset_root": r"./data/dataset_sanity",
    "fold": "0",

    "batch_size": 2,
    "epochs": 150,
    "lr": 3e-4,
    "weight_decay": 0.0,

    "early_stop_patience": 50,

    "model": "UNetBaseline",
    "base_channels": 64,

    "loss": "MaskedL1",

    "amp": True,
    "num_workers": 2,
    "pin_memory": True
}