import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dmec_net_step1 import ConvBlock, DownBlock, UpBlock, ModalityEncoder, SimpleDecoder


# =========================
# Gated Cross Attention
# =========================

class GatedCrossAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()

        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

        # gating mechanism (KEY)
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, f_t1, f_list):

        B, C, H, W = f_t1.shape

        def flatten(x):
            return x.view(B, C, -1).permute(0, 2, 1)

        q = flatten(f_t1)
        k = torch.cat([flatten(f) for f in f_list], dim=1)
        v = k

        attn_out, _ = self.attn(q, k, v)
        attn_out = self.norm(attn_out + q)

        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)

        # Gating mechanism
        gate = self.gate(torch.cat([f_t1, attn_out], dim=1))
        out = attn_out * gate

        # Tambahkan kembali fondasi T1 (Residual Connection)
        return f_t1 + out

# =========================
# DMEC STEP 3
# =========================

class DMECNetStep3(nn.Module):
    def __init__(self, use_t2=True, use_flair=True, base_ch=32):
        super().__init__()

        self.use_t2 = use_t2
        self.use_flair = use_flair

        self.enc_t1 = ModalityEncoder(3, base_ch)

        if use_t2:
            self.enc_t2 = ModalityEncoder(3, base_ch)

        if use_flair:
            self.enc_flair = ModalityEncoder(3, base_ch)

        self.fusion = GatedCrossAttention(dim=base_ch * 8)

        self.decoder = SimpleDecoder(base_ch * 8, base_ch)

    def forward(self, x):

        B, C, H, W = x.shape

        if C == 3:
            t1 = x
            t2 = flair = None

        elif C == 6:
            t1, t2 = x[:, :3], x[:, 3:6]
            flair = None

        elif C == 9:
            t1, t2, flair = x[:, :3], x[:, 3:6], x[:, 6:9]

        else:
            raise ValueError

        skips_t1, b_t1 = self.enc_t1(t1)

        context = []

        if self.use_t2 and t2 is not None:
            _, b_t2 = self.enc_t2(t2)
            context.append(b_t2)

        if self.use_flair and flair is not None:
            _, b_flair = self.enc_flair(flair)
            context.append(b_flair)

        if len(context) > 0:
            b = self.fusion(b_t1, context)
        else:
            b = b_t1

        out = self.decoder(b, skips_t1)

        return out