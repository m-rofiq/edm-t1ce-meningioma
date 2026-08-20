import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Reuse blocks dari step1
# =========================

from models.dmec_net_step1 import ConvBlock, DownBlock, UpBlock, ModalityEncoder, SimpleDecoder


# =========================
# Cross Attention Fusion
# =========================

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()

        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, f_t1, f_list):

        B, C, H, W = f_t1.shape

        def flatten(x):
            return x.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)

        q = flatten(f_t1)

        # gabungkan modality lain
        k = torch.cat([flatten(f) for f in f_list], dim=1)
        v = k

        out, _ = self.attn(q, k, v)
        out = self.norm(out + q)

        out = out.permute(0, 2, 1).view(B, C, H, W)

        return out


# =========================
# DMEC STEP 2
# =========================

class DMECNetStep2(nn.Module):
    def __init__(self, use_t2=True, use_flair=True, base_ch=32):
        super().__init__()

        self.use_t2 = use_t2
        self.use_flair = use_flair

        # encoders
        self.enc_t1 = ModalityEncoder(3, base_ch)

        if use_t2:
            self.enc_t2 = ModalityEncoder(3, base_ch)

        if use_flair:
            self.enc_flair = ModalityEncoder(3, base_ch)

        # cross-attention
        self.fusion = CrossAttentionFusion(dim=base_ch * 8)

        # decoder
        self.decoder = SimpleDecoder(base_ch * 8, base_ch)

    def forward(self, x):

        B, C, H, W = x.shape

        # split
        if C == 3:
            t1 = x
            t2 = flair = None

        elif C == 6:
            t1, t2 = x[:, :3], x[:, 3:6]
            flair = None

        elif C == 9:
            t1, t2, flair = x[:, :3], x[:, 3:6], x[:, 6:9]

        else:
            raise ValueError(f"Unsupported input channel: {C}")

        # encode
        skips_t1, b_t1 = self.enc_t1(t1)

        context = []

        if self.use_t2 and t2 is not None:
            _, b_t2 = self.enc_t2(t2)
            context.append(b_t2)

        if self.use_flair and flair is not None:
            _, b_flair = self.enc_flair(flair)
            context.append(b_flair)

        # fusion
        if len(context) > 0:
            b = self.fusion(b_t1, context)
        else:
            b = b_t1

        # decode
        out = self.decoder(b, skips_t1)

        return out