import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# ADD PROJECT ROOT
# =========================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from models.model_registry import MODEL_REGISTRY
from configs.config import CONFIG

def main():
    # =========================
    # CONFIG
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FOLD = CONFIG["fold"]
    EXP_DIR = os.path.join(CONFIG["experiment_root"], f'{CONFIG["experiment_id"]}_{CONFIG["experiment_name"]}_fold{FOLD}')
    CHECKPOINT_PATH = os.path.join(EXP_DIR, "best_model.pth")
    DATASET_ROOT = CONFIG["dataset_root"]
    HOLDOUT_PATH = os.path.join(DATASET_ROOT, "holdout_test")

    t1_dir = os.path.join(HOLDOUT_PATH, "T1")
    t1ce_dir = os.path.join(HOLDOUT_PATH, "T1CE")
    cases_file = os.path.join(EXP_DIR, "selected_visualization_cases.csv")

    # =========================
    # RADIOLOGY VIEW
    # =========================
    def radiology_view(img):
        return np.flip(img)

    # =========================
    # LOAD MODEL
    # =========================
    model_name = CONFIG["model"]

    if model_name in ["DMECNetStep1", "DMECNetStep2", "DMECNetStep3"]:

        model = MODEL_REGISTRY[model_name](
            use_t2 = "T2" in CONFIG["modalities"],
            use_flair = "FLAIR" in CONFIG["modalities"],
            base_ch = CONFIG["base_channels"]
        ).to(device)

    else:
        model = MODEL_REGISTRY[model_name](
            base_channels=CONFIG["base_channels"],
            in_channels = 3 * len(CONFIG["modalities"])
        ).to(device)

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    # =========================
    # LOAD CASE LIST
    # =========================
    cases = pd.read_csv(cases_file)
    print("\nSelected visualization cases:")
    print(cases)

    # =========================
    # HELPER
    # =========================
    def norm(img, vmin, vmax):
        img_clipped = np.clip(img, vmin, vmax)
        return (img_clipped - vmin) / (vmax - vmin + 1e-8)

    # =========================
    # LOOP CASES
    # =========================
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

        # =========================
        # FIX 1: PENCARIAN IRISAN TUMOR YANG SAH
        # =========================
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
            best_file = files[len(files)//2]

        print("Selected slice:", best_file)

        # =========================
        # LOAD DATA
        # =========================

        # KODE YANG BENAR:
        t1 = np.load(os.path.join(t1_dir, best_file))
        t1ce = np.load(os.path.join(t1ce_dir, best_file))

        input_center = t1[1] # Tetap simpan ini untuk visualisasi plot T1 saja
        gt = t1ce[1]

        # =========================
        # INFERENCE (MULTIMODAL EARLY FUSION)
        # =========================
        inputs = []
        for m in CONFIG["modalities"]:
            mod_dir = os.path.join(HOLDOUT_PATH, m)
            vol = np.load(os.path.join(mod_dir, best_file))
            inputs.append(vol)
            
        x_np = np.concatenate(inputs, axis=0) # Membentuk (9, 512, 512)
        x = torch.from_numpy(x_np).unsqueeze(0).float().to(device)

        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=(device.type=="cuda" and CONFIG["amp"])):
                output = model(x)

        if isinstance(output, tuple):
            pred = output[0]
        else:
            pred = output

        pred = pred.float().cpu().numpy()[0,0]

        # =========================
        # BRAIN MASK & NORMALIZATION (FIX 2: Cegah Background Abu-abu)
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

        # =========================
        # ERROR MAP
        # =========================
        error_map = np.abs(pred - gt) * brain_mask
    
        if np.sum(brain_mask) > 10:
            p = np.percentile(error_map[brain_mask], 99)
        else:
            p = np.percentile(error_map, 99)

        error_vis = error_map / (p + 1e-8)
        error_vis = np.clip(error_vis, 0, 1) * brain_mask

        # =========================
        # ENHANCEMENT DIFFERENCE MAP (DIKEMBALIKAN)
        # =========================
        enh_gt = (gt - input_center) * brain_mask
        enh_pred = (pred - input_center) * brain_mask

        enh_diff = np.abs(enh_pred - enh_gt) * brain_mask

        if np.sum(brain_mask) > 10:
            p = np.percentile(enh_diff[brain_mask], 99)
        else:
            p = np.percentile(enh_diff, 99)

        enh_diff = enh_diff / (p + 1e-8)
        enh_diff = np.clip(enh_diff, 0, 1) * brain_mask

        # =========================
        # RADIOLOGY VIEW
        # =========================
        input_vis = radiology_view(input_vis)
        gt_vis = radiology_view(gt_vis)
        pred_vis = radiology_view(pred_vis)
        error_vis = radiology_view(error_vis)
        enh_diff = radiology_view(enh_diff)

        brain_mask_rad = radiology_view(brain_mask)
        gt_rad = radiology_view(gt) 

        # =========================
        # TUMOR ROI ZOOM (FIX 3: Presisi Tumor)
        # =========================
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

        x1 = max(0, x1-pad)
        x2 = min(W, x2+pad)
        y1 = max(0, y1-pad)
        y2 = min(H, y2+pad)

        zoom_pred = pred_vis[y1:y2, x1:x2]
        zoom_gt = gt_vis[y1:y2, x1:x2]

        # =========================
        # PLOT FIGURE (KEMBALI KE 2x4 FORMAT)
        # =========================
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        for ax in axes.flatten():
            ax.axis("off")

        axes[0,0].imshow(input_vis, cmap="gray")
        axes[0,0].set_title("Input T1")

        axes[0,1].imshow(pred_vis, cmap="gray")
        axes[0,1].set_title("Generated T1CE")

        axes[0,2].imshow(gt_vis, cmap="gray")
        axes[0,2].set_title("Ground Truth T1CE")

        axes[0,3].imshow(error_vis, cmap="hot")
        axes[0,3].set_title("Absolute Error")

        axes[1,0].imshow(enh_diff, cmap="hot")
        axes[1,0].set_title("Enhancement Difference Map")

        axes[1,1].imshow(zoom_pred, cmap="gray")
        axes[1,1].set_title("Tumor Zoom (Pred)")

        axes[1,2].imshow(zoom_gt, cmap="gray")
        axes[1,2].set_title("Tumor Zoom (GT)")

        axes[1,3].axis("off") # Sisa kanvas kosong sesuai aslinya

        plt.tight_layout()
        save_path = os.path.join(EXP_DIR, f"{case_type}_case_visualization.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print("Saved:", save_path)

        pass

if __name__ == "__main__":
    import sys, json
    from configs.config import CONFIG

    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            new_cfg = json.load(f)
        CONFIG.clear()
        CONFIG.update(new_cfg)

    main()