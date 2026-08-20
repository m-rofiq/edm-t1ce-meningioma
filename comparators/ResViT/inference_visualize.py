import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from model_registry import MODEL_REGISTRY
from config_resvit_sota import CONFIG  # <-- beda dari versi DD-Res U-Net
from resvit_dataset import to_tanh_range, from_tanh_range, _load_global_range, RESVIT_IMG_SIZE


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_min, global_max = _load_global_range()

    FOLD = CONFIG["fold"]
    EXP_DIR = os.path.join(CONFIG["experiment_root"], f'{CONFIG["experiment_id"]}_{CONFIG["experiment_name"]}_fold{FOLD}')
    CHECKPOINT_PATH = os.path.join(EXP_DIR, "best_model.pth")
    DATASET_ROOT = CONFIG["dataset_root"]
    HOLDOUT_PATH = os.path.join(DATASET_ROOT, "holdout_test")

    t1_dir = os.path.join(HOLDOUT_PATH, "T1")
    t1ce_dir = os.path.join(HOLDOUT_PATH, "T1CE")
    cases_file = os.path.join(EXP_DIR, "selected_visualization_cases.csv")

    def radiology_view(img):
        return np.flip(img)

    # =========================
    # LOAD MODEL (konvensi ResViT: in_channels=len(modalities), img_size dari
    # CONFIG, pretrained_vit_path=None karena checkpoint sendiri akan ditimpa
    # lewat load_state_dict tepat sesudahnya)
    # =========================
    model_name = CONFIG["model"]
    model = MODEL_REGISTRY[model_name](
        base_channels=CONFIG["base_channels"],
        in_channels=len(CONFIG["modalities"]),
        img_size=CONFIG["img_size"],
        vit_name=CONFIG["vit_name"],
        pretrained_vit_path=None,
    ).to(device)

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    cases = pd.read_csv(cases_file)
    print("\nSelected visualization cases:")
    print(cases)

    def norm(img, vmin, vmax):
        img_clipped = np.clip(img, vmin, vmax)
        return (img_clipped - vmin) / (vmax - vmin + 1e-8)

    for _, row in cases.iterrows():
        case_type = row["type"]
        patient_id = row["patient_id"]
        print(f"\nProcessing {case_type} case:", patient_id)

        files = sorted([f for f in os.listdir(t1_dir) if f.startswith(patient_id) and f.endswith(".npy")])
        if len(files) == 0:
            print("No slices found for:", patient_id)
            continue

        best_file = None
        best_score = -1
        for f in files:
            t1ce_temp = np.load(os.path.join(t1ce_dir, f))
            gt_slice = t1ce_temp[1]
            brain_mask_temp = gt_slice != 0
            if brain_mask_temp.sum() > 0:
                mu = np.mean(gt_slice[brain_mask_temp])
                sigma = np.std(gt_slice[brain_mask_temp])
                enh_mask = (gt_slice > (mu + 1.5 * sigma)) & brain_mask_temp
                score = enh_mask.sum()
            else:
                score = 0
            if score > best_score:
                best_score = score
                best_file = f

        if best_file is None:
            best_file = files[len(files) // 2]

        print("Selected slice:", best_file)

        # =========================
        # LOAD DATA (native, z-score asli -- untuk visualisasi & GT)
        # =========================
        t1 = np.load(os.path.join(t1_dir, best_file))
        t1ce = np.load(os.path.join(t1ce_dir, best_file))
        input_center = t1[1]
        gt = t1ce[1]

        # =========================
        # INFERENCE (native ResViT: 1 slice tengah per modalitas, rescale
        # ke [-1,1], resize ke 256, infer, upsample balik ke 512,
        # inverse-transform kembali ke z-score)
        # =========================
        inputs = []
        for m in CONFIG["modalities"]:
            mod_dir = os.path.join(HOLDOUT_PATH, m)
            vol = np.load(os.path.join(mod_dir, best_file))
            center = vol[1:2]  # (1, H, W) -- slice tengah saja, BUKAN triplet
            inputs.append(to_tanh_range(center, global_min, global_max))

        x_native = torch.from_numpy(np.concatenate(inputs, axis=0)).unsqueeze(0).float().to(device)
        x_256 = F.interpolate(x_native, size=(RESVIT_IMG_SIZE, RESVIT_IMG_SIZE),
                               mode="bilinear", align_corners=False)

        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                pred_256 = model(x_256)
        pred_256 = pred_256.float()

        pred_native_tanh = F.interpolate(pred_256, size=x_native.shape[-2:],
                                          mode="bilinear", align_corners=False)
        pred_t = from_tanh_range(pred_native_tanh, global_min, global_max)
        pred = pred_t.cpu().numpy()[0, 0]  # kembali ke skala z-score asli, sama seperti gt

        # =========================
        # SISANYA IDENTIK dengan versi DD-Res U-Net (pred sudah di skala yang sama)
        # =========================
        brain_mask = gt != 0
        if np.sum(brain_mask) == 0:
            print("Empty brain mask:", best_file)
            continue

        t1_pixels = input_center[brain_mask]
        t1ce_pixels = gt[brain_mask]
        t1_min, t1_max = np.percentile(t1_pixels, 1), np.percentile(t1_pixels, 99)
        t1ce_min, t1ce_max = np.percentile(t1ce_pixels, 1), np.percentile(t1ce_pixels, 99)

        input_vis = norm(input_center, t1_min, t1_max) * brain_mask
        gt_vis = norm(gt, t1ce_min, t1ce_max) * brain_mask
        pred_vis = norm(pred, t1ce_min, t1ce_max) * brain_mask

        error_map = np.abs(pred - gt) * brain_mask
        p = np.percentile(error_map[brain_mask], 99) if np.sum(brain_mask) > 10 else np.percentile(error_map, 99)
        error_vis = np.clip(error_map / (p + 1e-8), 0, 1) * brain_mask

        enh_gt = (gt - input_center) * brain_mask
        enh_pred = (pred - input_center) * brain_mask
        enh_diff = np.abs(enh_pred - enh_gt) * brain_mask
        p = np.percentile(enh_diff[brain_mask], 99) if np.sum(brain_mask) > 10 else np.percentile(enh_diff, 99)
        enh_diff = np.clip(enh_diff / (p + 1e-8), 0, 1) * brain_mask

        input_vis = radiology_view(input_vis)
        gt_vis = radiology_view(gt_vis)
        pred_vis = radiology_view(pred_vis)
        error_vis = radiology_view(error_vis)
        enh_diff = radiology_view(enh_diff)
        brain_mask_rad = radiology_view(brain_mask)
        gt_rad = radiology_view(gt)

        mu_rad = np.mean(gt_rad[brain_mask_rad])
        sigma_rad = np.std(gt_rad[brain_mask_rad])
        tumor_mask = (gt_rad > (mu_rad + 1.5 * sigma_rad)) & brain_mask_rad
        ys, xs = np.where(tumor_mask)

        if len(xs) > 0:
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
        else:
            x1, x2 = 200, 320
            y1, y2 = 200, 320

        pad = 20
        H, W = gt.shape
        x1, x2 = max(0, x1 - pad), min(W, x2 + pad)
        y1, y2 = max(0, y1 - pad), min(H, y2 + pad)

        zoom_pred = pred_vis[y1:y2, x1:x2]
        zoom_gt = gt_vis[y1:y2, x1:x2]

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for ax in axes.flatten():
            ax.axis("off")

        axes[0, 0].imshow(input_vis, cmap="gray"); axes[0, 0].set_title("Input T1")
        axes[0, 1].imshow(pred_vis, cmap="gray"); axes[0, 1].set_title("Generated T1CE")
        axes[0, 2].imshow(gt_vis, cmap="gray"); axes[0, 2].set_title("Ground Truth T1CE")
        axes[0, 3].imshow(error_vis, cmap="hot"); axes[0, 3].set_title("Absolute Error")
        axes[1, 0].imshow(enh_diff, cmap="hot"); axes[1, 0].set_title("Enhancement Difference Map")
        axes[1, 1].imshow(zoom_pred, cmap="gray"); axes[1, 1].set_title("Tumor Zoom (Pred)")
        axes[1, 2].imshow(zoom_gt, cmap="gray"); axes[1, 2].set_title("Tumor Zoom (GT)")
        axes[1, 3].axis("off")

        plt.tight_layout()
        save_path = os.path.join(EXP_DIR, f"{case_type}_case_visualization.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved:", save_path)


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            new_cfg = json.load(f)
        CONFIG.clear()
        CONFIG.update(new_cfg)
    main()
