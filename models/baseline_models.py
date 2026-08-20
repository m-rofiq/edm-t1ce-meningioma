"""
Baseline models untuk perbandingan dengan EDMSynth.

1. Pix2Pix  : UNet Generator standar (Isola et al., 2017)
              in_channels = 9 (T1×3 + T2×3 + FLAIR×3)
              out_channels = 1 (T1CE)

2. SimpleUNet: UNet sederhana tanpa CBAM, tanpa cross-attention,
               tanpa skip projection — murni encoder-decoder dasar.
               Digunakan sebagai lower-bound baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================
# BUILDING BLOCKS
# ======================================================

class ConvBlock(nn.Module):
    """Conv → InstanceNorm → LeakyReLU"""
    def __init__(self, in_ch, out_ch, stride=1, use_norm=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1, bias=not use_norm)
        ]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=False))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpConvBlock(nn.Module):
    """ConvTranspose → InstanceNorm → ReLU, dengan skip concat"""
    def __init__(self, in_ch, out_ch, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=False),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.block(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)


# ======================================================
# PIX2PIX GENERATOR (UNet-based)
# ======================================================
class Pix2PixGenerator(nn.Module):
    """
    UNet Generator dari Pix2Pix (Isola et al., 2017).
    Disesuaikan untuk input multi-modal MRI:
      - in_channels  = 9 (T1×3 + T2×3 + FLAIR×3)
      - out_channels = 1 (T1CE)
    Architecture: 8-level UNet dengan skip connections.
    """
    def __init__(self, in_channels=9, out_channels=1, base_ch=64):
        super().__init__()

        # Encoder (downsampling)
        self.e1 = nn.Conv2d(in_channels, base_ch, 4, stride=2, padding=1)    # no norm on first
        self.e2 = ConvBlock(base_ch,     base_ch*2,  stride=2)
        self.e3 = ConvBlock(base_ch*2,   base_ch*4,  stride=2)
        self.e4 = ConvBlock(base_ch*4,   base_ch*8,  stride=2)
        self.e5 = ConvBlock(base_ch*8,   base_ch*8,  stride=2)
        self.e6 = ConvBlock(base_ch*8,   base_ch*8,  stride=2)
        self.e7 = ConvBlock(base_ch*8,   base_ch*8,  stride=2)
        # Bottleneck
        self.e8 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(base_ch*8, base_ch*8, 4, stride=2, padding=1)
        )

        # Decoder (upsampling + skip concat)
        # in_ch = prev_out + skip_ch
        self.d1 = UpConvBlock(base_ch*8,   base_ch*8, dropout=True)
        self.d2 = UpConvBlock(base_ch*8*2, base_ch*8, dropout=True)
        self.d3 = UpConvBlock(base_ch*8*2, base_ch*8, dropout=True)
        self.d4 = UpConvBlock(base_ch*8*2, base_ch*8)
        self.d5 = UpConvBlock(base_ch*8*2, base_ch*4)
        self.d6 = UpConvBlock(base_ch*4*2, base_ch*2)
        self.d7 = UpConvBlock(base_ch*2*2, base_ch)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(base_ch*2, out_channels, 4, stride=2, padding=1),
            nn.Tanh()   # output [-1, 1], akan di-rescale ke [0,1] saat evaluasi
        )

    def forward(self, x):
        # Encode
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)

        # Decode dengan skip
        d1 = self.d1(e8, e7)
        d2 = self.d2(d1, e6)
        d3 = self.d3(d2, e5)
        d4 = self.d4(d3, e4)
        d5 = self.d5(d4, e3)
        d6 = self.d6(d5, e2)
        d7 = self.d7(d6, e1)

        out = self.final(d7)
        # Rescale dari [-1,1] ke [0,1]
        return (out + 1.0) / 2.0


# ======================================================
# SIMPLE UNET
# ======================================================
class SimpleUNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.skip(x)


class SimpleUNet(nn.Module):
    """
    UNet sederhana tanpa CBAM, tanpa cross-attention,
    tanpa dual encoder. Lower-bound baseline murni.
    in_channels = 9 (T1×3 + T2×3 + FLAIR×3)
    """
    def __init__(self, in_channels=9, out_channels=1, base_ch=64):
        super().__init__()

        # Encoder
        self.e1 = SimpleUNetBlock(in_channels, base_ch)
        self.p1 = nn.MaxPool2d(2)
        self.e2 = SimpleUNetBlock(base_ch,   base_ch*2)
        self.p2 = nn.MaxPool2d(2)
        self.e3 = SimpleUNetBlock(base_ch*2, base_ch*4)
        self.p3 = nn.MaxPool2d(2)
        self.e4 = SimpleUNetBlock(base_ch*4, base_ch*8)
        self.p4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = SimpleUNetBlock(base_ch*8, base_ch*16)

        # Decoder
        self.u4 = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, stride=2)
        self.d4 = SimpleUNetBlock(base_ch*16, base_ch*8)

        self.u3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.d3 = SimpleUNetBlock(base_ch*8, base_ch*4)

        self.u2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.d2 = SimpleUNetBlock(base_ch*4, base_ch*2)

        self.u1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.d1 = SimpleUNetBlock(base_ch*2, base_ch)

        self.final = nn.Conv2d(base_ch, out_channels, 1)

    def forward(self, x):
        # Encode
        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        e3 = self.e3(self.p2(e2))
        e4 = self.e4(self.p3(e3))
        bn = self.bottleneck(self.p4(e4))

        # Decode
        d4 = self.d4(torch.cat([self._up(self.u4, bn, e4), e4], dim=1))
        d3 = self.d3(torch.cat([self._up(self.u3, d4, e3), e3], dim=1))
        d2 = self.d2(torch.cat([self._up(self.u2, d3, e2), e2], dim=1))
        d1 = self.d1(torch.cat([self._up(self.u1, d2, e1), e1], dim=1))

        return torch.sigmoid(self.final(d1))

    def _up(self, upsample, x, target):
        x = upsample(x)
        if x.shape[-2:] != target.shape[-2:]:
            x = F.interpolate(x, size=target.shape[-2:],
                              mode="bilinear", align_corners=False)
        return x


# ======================================================
# REGISTRY
# ======================================================
BASELINE_REGISTRY = {
    "Pix2Pix"   : Pix2PixGenerator,
    "SimpleUNet" : SimpleUNet,
}
