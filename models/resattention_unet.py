import torch
import torch.nn as nn
import torch.nn.functional as F

class FDRBlock(nn.Module):
    """
    Blok Feature Double Reuse (FDR).
    Menggabungkan konsep Residual (addition) dan Dense (concatenation) 
    untuk meminimalisir kehilangan fitur spasial pada encoder.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Pre-activation style
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_ch)
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_ch)

        # Skip connection untuk bagian Residual
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        identity = self.skip(x)
        
        # Aliran fitur utama
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        
        # Penjumlahan residu (FDR core: reuse fitur input ke output)
        return self.act(out + identity)

class AttentionGate(nn.Module):
    def __init__(self, Fg, Fl, Fint):
        super().__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(Fg, Fint, kernel_size=1, bias=True),
            nn.InstanceNorm2d(Fint)
        )
        self.Wx = nn.Sequential(
            nn.Conv2d(Fl, Fint, kernel_size=1, bias=True),
            nn.InstanceNorm2d(Fint)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(Fint, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.Wg(g)
        x1 = self.Wx(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class BilinearUp(nn.Module):
    """Mekanisme Upsampling Bilinear untuk mencegah checkerboard artifacts"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(self.up(x))

class ResAttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super().__init__()
        c = base_channels

        # ==========================================
        # ENCODER (Menggunakan FDRBlock)
        # ==========================================
        self.enc1 = FDRBlock(in_channels, c)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = FDRBlock(c, c * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = FDRBlock(c * 2, c * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = FDRBlock(c * 4, c * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = FDRBlock(c * 8, c * 16)

        # ==========================================
        # DECODER (Bilinear + Attention + Skip)
        # ==========================================
        self.up4 = BilinearUp(c * 16, c * 8)
        self.att4 = AttentionGate(c * 8, c * 8, c * 4)
        self.dec4 = FDRBlock(c * 16, c * 8)

        self.up3 = BilinearUp(c * 8, c * 4)
        self.att3 = AttentionGate(c * 4, c * 4, c * 2)
        self.dec3 = FDRBlock(c * 8, c * 4)

        self.up2 = BilinearUp(c * 4, c * 2)
        self.att2 = AttentionGate(c * 2, c * 2, c)
        self.dec2 = FDRBlock(c * 4, c * 2)

        self.up1 = BilinearUp(c * 2, c)
        self.att1 = AttentionGate(c, c, c // 2)
        self.dec1 = FDRBlock(c * 2, c)

        # Final Output
        self.final = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder dengan Skip Connections (torch.cat)
        d4 = self.up4(b)
        e4_att = self.att4(d4, e4) # Attention Gate
        d4 = torch.cat([d4, e4_att], dim=1) # Skip Connection
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        e3_att = self.att3(d3, e3)
        d3 = torch.cat([d3, e3_att], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        e2_att = self.att2(d2, e2)
        d2 = torch.cat([d2, e2_att], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        e1_att = self.att1(d1, e1)
        d1 = torch.cat([d1, e1_att], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)