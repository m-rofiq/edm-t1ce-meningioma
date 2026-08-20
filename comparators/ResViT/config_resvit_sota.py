CONFIG = {

    # =========================
    # Experiment
    # =========================
    "experiment_id": "EXP-RESVIT",
    "experiment_name": "Dalmaz2022",
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
    # Training (default sesuai train_options.py ResViT asli)
    # =========================
    "batch_size": 4,
    "niter": 100,        # epoch dengan lr konstan
    "niter_decay": 100,  # epoch dengan lr decay linear ke 0
    "lr": 0.0002,
    "beta1": 0.5,

    # =========================
    # Loss weights (default sesuai ResViT/pix2pix asli)
    # =========================
    "lambda_A": 10.0,   # bobot L1
    "lambda_adv": 1.0,  # bobot adversarial
    "use_lsgan": True,  # least-square GAN (default ResViT, no_lsgan=False)
    "pool_size": 50,     # ImagePool untuk stabilitas training discriminator

    # =========================
    # Model
    # =========================
    "model": "ResViT",
    "base_channels": 64,  # ngf, default ResViT asli
    "modalities": ["T1", "T2", "FLAIR"],
    "vit_name": "Res-ViT-B_16",
    # CATATAN: link resmi Google untuk checkpoint ini SUDAH EXPIRED (per awal
    # 2026, dikonfirmasi publik oleh proyek lain seperti TransUNet). Kalau
    # Anda tidak punya salinan valid, set ke None untuk training dari bobot
    # acak (deviasi dari resep asli paper, dokumentasikan di Methods).
    # WAJIB 256 -- hyperparameter arsitektural yang dipatok (grid transformer
    # 16x16 + upsample tetap 2x), BUKAN resolusi native MRI Anda (512).
    # ResViTDataset otomatis resize 512->256 untuk training; evaluasi final
    # nanti upsample kembali ke 512 untuk dibandingkan dengan DD-Res U-Net.
    "img_size": 256,
    # Path ke checkpoint ImageNet-21k GENERIK (bukan task-specific), download dari
    # https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz
    "pretrained_vit_path": r"./models/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz",
    #"pretrained_vit_path": None,
    "pretrained_resnet": False,
    #"early_stop_patience": 30, #optional

    # =========================
    # Engineering
    # =========================
    "num_workers": 4,
    "pin_memory": True,
}
