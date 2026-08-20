import torch
import torch.nn as nn

class EDMLoss(torch.nn.Module):
    def __init__(self, lambda_enh=0.5):
        super().__init__()
        self.lambda_enh = lambda_enh

    def forward(self, pred_tuple, target, mask, input_tensor):

        pred, base, enh = pred_tuple

        # =========================
        # MAIN LOSS
        # =========================
        loss_main = torch.mean(torch.abs(pred - target)[mask])

        # =========================
        # FIX: ENHANCEMENT GT DARI T1
        # =========================
        # ambil T1 center slice (channel ke-2 dari 3 slice)
        t1_center = input_tensor[:, 1:2, :, :]

        enh_gt = target - t1_center

        loss_enh = torch.mean(torch.abs(enh - enh_gt)[mask])

        return loss_main + self.lambda_enh * loss_enh