import torch
import torch.nn as nn

from models.resunet import ResUNet
from models.resattention_unet import ResAttentionUNet


class EDMSynth(nn.Module):
    def __init__(self, base_channels=32,
                 in_channels=None,
                 use_t2=True,
                 use_flair=True):
        super().__init__()

        self.use_t2 = use_t2
        self.use_flair = use_flair

        # =========================
        # BASE NET
        # =========================
        self.base_net = ResUNet(
            in_channels=3,
            base_channels=base_channels
        )

        # =========================
        # ENH NET
        # =========================
        cond_ch = 0

        if use_t2:
            cond_ch += 3

        if use_flair:
            cond_ch += 3

        self.has_cond = cond_ch > 0

        if self.has_cond:
            self.enh_net = ResAttentionUNet(
                in_channels=cond_ch,
                base_channels=base_channels
            )

    def forward(self, x):

        if x.shape[1] == 3:
            t1 = x
            cond = None
        else:
            t1 = x[:, :3]
            cond = x[:, 3:]

        # =========================
        # BASE
        # =========================
        base = self.base_net(t1)

        # =========================
        # ENH
        # =========================
        if self.has_cond and cond is not None:
            enh = self.enh_net(cond)
        else:
            enh = torch.zeros_like(base)

        # =========================
        # FINAL
        # =========================
        pred = base + enh

        return pred, base, enh