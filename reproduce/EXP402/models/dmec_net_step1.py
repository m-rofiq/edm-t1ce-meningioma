import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Basic Blocks (reuse jika sudah ada)
# =========================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x_down = self.pool(x)
        return x, x_down


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# =========================
# Encoder (per modality)
# =========================

class ModalityEncoder(nn.Module):
    def __init__(self, in_channels=3, base_ch=32):
        super().__init__()
        self.down1 = DownBlock(in_channels, base_ch)        # 32
        self.down2 = DownBlock(base_ch, base_ch * 2)        # 64
        self.down3 = DownBlock(base_ch * 2, base_ch * 4)    # 128
        self.down4 = DownBlock(base_ch * 4, base_ch * 8)    # 256

    def forward(self, x):
        s1, x = self.down1(x)
        s2, x = self.down2(x)
        s3, x = self.down3(x)
        s4, x = self.down4(x)
        return [s1, s2, s3, s4], x  # skips, bottleneck


# =========================
# Decoder (reuse structure UNet)
# =========================

class SimpleDecoder(nn.Module):
    def __init__(self, in_ch, base_ch=32):
        super().__init__()
        self.up1 = UpBlock(in_ch + base_ch * 8, base_ch * 8)
        self.up2 = UpBlock(base_ch * 8 + base_ch * 4, base_ch * 4)
        self.up3 = UpBlock(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.up4 = UpBlock(base_ch * 2 + base_ch, base_ch)

        self.out = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x, skips):
        s1, s2, s3, s4 = skips

        x = self.up1(x, s4)
        x = self.up2(x, s3)
        x = self.up3(x, s2)
        x = self.up4(x, s1)

        return self.out(x)


# =========================
# DMEC-Net STEP 1
# =========================

class DMECNetStep1(nn.Module):
    def __init__(self, use_t2=True, use_flair=True, base_ch=32):
        super().__init__()

        self.use_t2 = use_t2
        self.use_flair = use_flair

        # encoders
        self.enc_t1 = ModalityEncoder(3, base_ch)

        if self.use_t2:
            self.enc_t2 = ModalityEncoder(3, base_ch)
        if self.use_flair:
            self.enc_flair = ModalityEncoder(3, base_ch)

        # bottleneck fusion (concat → reduce channel)
        n_modal = 1 + int(use_t2) + int(use_flair)
        self.fusion_conv = nn.Conv2d(base_ch * 8 * n_modal, base_ch * 8, kernel_size=1)

        # decoder (gunakan skip T1 saja → stabil)
        self.decoder = SimpleDecoder(base_ch * 8, base_ch)

    # =========================
    # Forward
    # =========================

    def forward(self, x):
        B, C, H, W = x.shape

        # split input
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

        bottlenecks = [b_t1]

        if self.use_t2 and t2 is not None:
            _, b_t2 = self.enc_t2(t2)
            bottlenecks.append(b_t2)

        if self.use_flair and flair is not None:
            _, b_flair = self.enc_flair(flair)
            bottlenecks.append(b_flair)

        # concat fusion
        b = torch.cat(bottlenecks, dim=1)
        b = self.fusion_conv(b)

        # decode (skip dari T1 saja)
        out = self.decoder(b, skips_t1)

        return out