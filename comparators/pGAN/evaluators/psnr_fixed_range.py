import torch
import torch.nn.functional as F
import json
import os


CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "configs",
    "metric_config.json"
)


def load_data_range():
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    return float(cfg["data_range"])


# ===============================
# ROI PSNR (NO BACKGROUND)
# ===============================
def psnr_fixed(pred, target, data_range=None):

    if data_range is None:
        data_range = load_data_range()

    mse = torch.mean((pred - target) ** 2)

    if mse == 0:
        return torch.tensor(99.0, device=pred.device)

    psnr = 10 * torch.log10((data_range ** 2) / (mse + 1e-8))

    return psnr

def psnr_roi(pred, target, mask, data_range=None):

    if data_range is None:
        data_range = load_data_range()

    pred_roi = pred[mask == 1]
    target_roi = target[mask == 1]

    mse = torch.mean((pred_roi - target_roi) ** 2)

    psnr = 10 * torch.log10((data_range ** 2) / (mse + 1e-8))

    return psnr


# ===============================
# CALIBRATION BIAS
# ===============================
def calibration_bias(pred, target, mask):

    pred_roi = pred[mask == 1]
    target_roi = target[mask == 1]

    return torch.mean(pred_roi - target_roi)


# ===============================
# VARIANCE RATIO
# ===============================
def variance_ratio(pred, target, mask):

    pred_roi = pred[mask == 1]
    target_roi = target[mask == 1]

    var_pred = torch.var(pred_roi)
    var_gt = torch.var(target_roi)

    return var_pred / (var_gt + 1e-8)


# ===============================
# GRADIENT DIFFERENCE
# ===============================
def gradient_difference_tumor(pred, target, tumor_mask):
    """
    Gradient difference evaluated only on CENTER SLICE inside tumor mask.
    Avoid skull-strip edge bias.
    """

    # Expect shape (B,C,H,W)
    if pred.ndim == 4:
        pred = pred[0]
        target = target[0]
        tumor_mask = tumor_mask[0]

    # Take center slice (2.5D → C=3)
    center = pred.shape[0] // 2
    pred = pred[center].unsqueeze(0).unsqueeze(0)
    target = target[center].unsqueeze(0).unsqueeze(0)
    tumor_mask = tumor_mask[center]

    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], 
                           dtype=torch.float32,
                           device=pred.device).view(1,1,3,3)

    sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], 
                           dtype=torch.float32,
                           device=pred.device).view(1,1,3,3)

    grad_pred_x = torch.nn.functional.conv2d(pred, sobel_x, padding=1)
    grad_pred_y = torch.nn.functional.conv2d(pred, sobel_y, padding=1)

    grad_gt_x = torch.nn.functional.conv2d(target, sobel_x, padding=1)
    grad_gt_y = torch.nn.functional.conv2d(target, sobel_y, padding=1)

    grad_diff = torch.abs(grad_pred_x - grad_gt_x) + \
                torch.abs(grad_pred_y - grad_gt_y)

    grad_diff = grad_diff.squeeze()

    tumor_bool = tumor_mask.bool()

    if tumor_bool.sum() == 0:
        return torch.tensor(float("nan"))

    return torch.mean(grad_diff[tumor_bool])

# ===============================
# TUMOR PSNR (ANTI DILUTION BIAS)
# ===============================
def psnr_tumor(pred, target, tumor_mask, data_range=None):

    if data_range is None:
        data_range = load_data_range()

    pred_roi = pred[tumor_mask == 1]
    target_roi = target[tumor_mask == 1]

    mse = torch.mean((pred_roi - target_roi) ** 2)

    if mse == 0:
        return torch.tensor(99.0, device=pred.device)

    psnr = 10 * torch.log10((data_range ** 2) / (mse + 1e-8))

    return psnr


# ===============================
# ENHANCEMENT ERROR
# ===============================
def enhancement_error(pred, target, tumor_mask):

    pred_roi = pred[tumor_mask == 1]
    target_roi = target[tumor_mask == 1]

    return torch.mean(torch.abs(pred_roi - target_roi))