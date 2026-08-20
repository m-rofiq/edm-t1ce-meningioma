# DD-Res U-Net (Osman & Tamam, 2023) -- PyTorch port for SOTA baseline comparison.
#
# FAITHFUL TO ORIGINAL PAPER/KODE (github afiosman, model.py yang sudah dianalisis
# sebelumnya):
#   - Residual conv block: Conv-ReLU-Conv(-Dropout) + 1x1 shortcut, add, ReLU.
#     TIDAK ada normalization layer (BatchNorm/InstanceNorm) -- paper asli memang
#     tidak memakainya, jadi sengaja tidak ditambahkan di sini walau codebase
#     GuidedResAttentionUNet Anda pakai InstanceNorm (itu desain EDMSynth sendiri,
#     bukan bagian dari arsitektur yang sedang dibandingkan).
#   - Dropout schedule sama persis dengan versi asli: 0.10 / 0.15 / 0.20 (encoder),
#     0.25 (bridge dense-dilated), 0.20 / 0.15 / 0.10 (decoder, simetris).
#   - Bridge dense-dilated: 3 conv dengan dilation rate 1, 2, 5 (concat bertahap),
#     persis desain "dense-dilated" di judul paper.
#
# ADAPTASI (2D, bukan 3D) -- KARENA FORMAT DATASET, BUKAN MENGUBAH METODE:
#   - Conv3D/MaxPool3D/ConvTranspose3D -> Conv2d/MaxPool2d/ConvTranspose2d,
#     karena dataset Anda 2.5D per-slice (bukan volume 3D utuh), sesuai
#     kesepakatan protokol adaptasi input yang sudah dibahas.
#   - Aktivasi output LINEAR (bukan ReLU seperti versi asli): data Anda
#     ter-normalisasi z-score (nilai negatif ada), ReLU akan memotong semua
#     prediksi negatif jadi 0 -- ini bug fatal untuk data Anda, bukan pilihan gaya.

import torch
import torch.nn as nn


class DDResConvBlock(nn.Module):
    """Residual conv block -- identik dengan res_conv_block versi TF asli."""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        h = self.dropout(h)
        return self.act(h + residual)


class DDResUNet(nn.Module):
    """
    3D Dense-Dilated Residual U-Net (Osman & Tamam, 2023), diporting ke 2D.

    Constructor signature MENGIKUTI konvensi model_registry.py Anda (lihat
    train_edm_exp602.py: MODEL_REGISTRY[model_name](base_channels=...,
    in_channels=...)) -- out_channels punya default sendiri (1) karena skrip
    training tidak mengirim argumen ini secara eksplisit.
    """

    def __init__(self, in_channels=9, out_channels=1, base_channels=16):
        super().__init__()
        f = base_channels

        # Encoder
        self.enc1 = DDResConvBlock(in_channels, f, dropout=0.10)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DDResConvBlock(f, f * 2, dropout=0.15)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DDResConvBlock(f * 2, f * 4, dropout=0.20)
        self.pool3 = nn.MaxPool2d(2)

        # Dense-dilated bridge (dilation rates 1, 2, 5)
        self.bridge_d1 = nn.Conv2d(f * 4, f * 8, kernel_size=3, padding=1, dilation=1)
        self.bridge_d2 = nn.Conv2d(f * 8, f * 8, kernel_size=3, padding=2, dilation=2)
        self.bridge_d5 = nn.Conv2d(f * 16, f * 8, kernel_size=3, padding=5, dilation=5)
        self.bridge_act = nn.ReLU(inplace=True)
        self.bridge_dropout = nn.Dropout2d(0.25)

        # Decoder
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = DDResConvBlock(f * 4 + f * 4, f * 4, dropout=0.20)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = DDResConvBlock(f * 2 + f * 2, f * 2, dropout=0.15)

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = DDResConvBlock(f + f, f, dropout=0.10)

        # Output linear (BUKAN relu -- lihat catatan di atas)
        self.final = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b1 = self.bridge_act(self.bridge_d1(p3))
        b1 = self.bridge_dropout(b1)
        b2 = self.bridge_act(self.bridge_d2(b1))
        b2 = self.bridge_dropout(b2)
        conc = torch.cat([b1, b2], dim=1)
        b5 = self.bridge_act(self.bridge_d5(conc))
        b5 = self.bridge_dropout(b5)

        d3 = self.up3(b5)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1)
