import torch
import torch.nn as nn

class EDMLoss(nn.Module):
    def __init__(self, lambda_enh=0.5):
        super().__init__()
        self.lambda_enh = lambda_enh

    def forward(self, pred_tuple, target, mask):

        pred, base, enh = pred_tuple

        # main loss
        loss_main = torch.mean(torch.abs(pred - target)[mask])

        # enhancement GT
        enh_gt = target - base.detach()

        loss_enh = torch.mean(torch.abs(enh - enh_gt)[mask])

        return loss_main + self.lambda_enh * loss_enh