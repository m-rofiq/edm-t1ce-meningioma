import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================
# RES BLOCK
# [CHANGED] InstanceNorm2d → GroupNorm
# GroupNorm tidak menghapus informasi intensitas
# absolut antar-channel → lebih akurat untuk MRI
# synthesis di mana magnitude sinyal penting.
# num_groups=min(8, out_ch) agar aman untuk ch kecil.
# =====================================================
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)

        self.act  = nn.LeakyReLU(0.2, inplace=True)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + identity)


# =====================================================
# CROSS ATTENTION  (tidak berubah)
# =====================================================
class CrossAttention2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q     = nn.Conv2d(dim, dim, 1)
        self.k     = nn.Conv2d(dim, dim, 1)
        self.v     = nn.Conv2d(dim, dim, 1)
        self.proj  = nn.Conv2d(dim, dim, 1)
        self.scale = dim ** -0.5

    def forward(self, q_feat, kv_feat):
        B, C, H, W = q_feat.shape
        q    = self.q(q_feat).flatten(2).transpose(1, 2)
        k    = self.k(kv_feat).flatten(2)
        v    = self.v(kv_feat).flatten(2).transpose(1, 2)
        attn = torch.softmax((q @ k) * self.scale, dim=-1)
        out  = (attn @ v).transpose(1, 2).reshape(B, C, H, W)
        return q_feat + self.proj(out)


# =====================================================
# UPSAMPLE BLOCK  (tidak berubah)
# =====================================================
class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 1)
        )
        self.block = ResBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


# =====================================================
# MODEL  (tidak berubah kecuali ResBlock sudah GroupNorm)
# =====================================================
class EDMSynth(nn.Module):
    def __init__(self, base_channels=32, in_channels=None, use_t2=True, use_flair=True):
        super().__init__()

        c = base_channels

        self.e1 = ResBlock(3, c);      self.p1 = nn.MaxPool2d(2)
        self.e2 = ResBlock(c, c*2);    self.p2 = nn.MaxPool2d(2)
        self.e3 = ResBlock(c*2, c*4);  self.p3 = nn.MaxPool2d(2)
        self.e4 = ResBlock(c*4, c*8);  self.p4 = nn.MaxPool2d(2)
        self.b1 = ResBlock(c*8, c*16)

        cond_ch = 0
        if use_t2:    cond_ch += 3
        if use_flair: cond_ch += 3
        self.has_cond = cond_ch > 0

        if self.has_cond:
            self.ce1 = ResBlock(cond_ch, c);   self.cp1 = nn.MaxPool2d(2)
            self.ce2 = ResBlock(c, c*2);        self.cp2 = nn.MaxPool2d(2)
            self.ce3 = ResBlock(c*2, c*4);      self.cp3 = nn.MaxPool2d(2)
            self.ce4 = ResBlock(c*4, c*8);      self.cp4 = nn.MaxPool2d(2)
            self.b2  = ResBlock(c*8, c*16)
            self.attn_b = CrossAttention2D(c*16)
            self.attn4  = CrossAttention2D(c*8)
            self.attn3  = CrossAttention2D(c*4)

        self.up4 = UpBlock(c*16, c*8, c*8)
        self.up3 = UpBlock(c*8,  c*4, c*4)
        self.up2 = UpBlock(c*4,  c*2, c*2)
        self.up1 = UpBlock(c*2,  c,   c)

        self.final = nn.Conv2d(c, 1, 1)
        self.ds4   = nn.Conv2d(c*8, 1, 1)
        self.ds3   = nn.Conv2d(c*4, 1, 1)

    def forward(self, x):
        t1   = x[:, :3]
        cond = x[:, 3:] if x.shape[1] > 3 else None

        e1 = self.e1(t1)
        e2 = self.e2(self.p1(e1))
        e3 = self.e3(self.p2(e2))
        e4 = self.e4(self.p3(e3))
        b1 = self.b1(self.p4(e4))

        if self.has_cond and cond is not None:
            c1 = self.ce1(cond)
            c2 = self.ce2(self.cp1(c1))
            c3 = self.ce3(self.cp2(c2))
            c4 = self.ce4(self.cp3(c3))
            b2 = self.b2(self.cp4(c4))
            e4 = self.attn4(e4, c4)
            e3 = self.attn3(e3, c3)
            bottleneck = self.attn_b(b1, b2)
        else:
            bottleneck = b1

        d4 = self.up4(bottleneck, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        pred = self.final(d1)

        _aux4 = F.interpolate(self.ds4(d4), size=pred.shape[-2:], mode="bilinear", align_corners=False)
        _aux3 = F.interpolate(self.ds3(d3), size=pred.shape[-2:], mode="bilinear", align_corners=False)

        base = pred
        enh  = torch.zeros_like(pred)

        return pred, base, enh
