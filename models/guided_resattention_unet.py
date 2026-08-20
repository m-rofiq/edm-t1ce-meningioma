import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_ch)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        identity = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + identity)

class GuidedAttentionGate(nn.Module):
    def __init__(self, Fg, Fl, Fc, Fint):
        super().__init__()
        self.Wg = nn.Sequential(nn.Conv2d(Fg, Fint, kernel_size=1, bias=True), nn.InstanceNorm2d(Fint))
        self.Wx = nn.Sequential(nn.Conv2d(Fl, Fint, kernel_size=1, bias=True), nn.InstanceNorm2d(Fint))
        
        self.use_cond = Fc > 0
        if self.use_cond:
            self.Wc = nn.Sequential(nn.Conv2d(Fc, Fint, kernel_size=1, bias=True), nn.InstanceNorm2d(Fint))
            
        self.psi = nn.Sequential(nn.Conv2d(Fint, 1, kernel_size=1, bias=True), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x, c=None):
        g1 = self.Wg(g)
        x1 = self.Wx(x)
        
        if self.use_cond and c is not None:
            c1 = self.Wc(c)
            psi = self.relu(g1 + x1 + c1)
        else:
            psi = self.relu(g1 + x1)
            
        psi = self.psi(psi)
        return x * psi

class GuidedResAttentionUNet(nn.Module):
    def __init__(self, in_channels=9, out_channels=1, base_channels=32):
        super().__init__()
        
        c = base_channels
        self.cond_channels = in_channels - 3 # T1 mengambil 3 channel, sisanya cond

        # =========================
        # MAIN T1 ENCODER
        # =========================
        self.enc1 = ResidualBlock(3, c)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(c, c * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResidualBlock(c * 2, c * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ResidualBlock(c * 4, c * 8)
        self.pool4 = nn.MaxPool2d(2)

        # =========================
        # CONDITION ENCODER (T2/FLAIR)
        # =========================
        if self.cond_channels > 0:
            self.cond1 = ResidualBlock(self.cond_channels, c)
            self.c_pool1 = nn.MaxPool2d(2)
            self.cond2 = ResidualBlock(c, c * 2)
            self.c_pool2 = nn.MaxPool2d(2)
            self.cond3 = ResidualBlock(c * 2, c * 4)
            self.c_pool3 = nn.MaxPool2d(2)
            self.cond4 = ResidualBlock(c * 4, c * 8)

        # =========================
        # BOTTLENECK & DECODER
        # =========================
        self.bottleneck = ResidualBlock(c * 8, c * 16)

        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=2, stride=2)
        self.att4 = GuidedAttentionGate(Fg=c*8, Fl=c*8, Fc=c*8 if self.cond_channels > 0 else 0, Fint=c*4)
        self.dec4 = ResidualBlock(c * 16, c * 8)

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.att3 = GuidedAttentionGate(Fg=c*4, Fl=c*4, Fc=c*4 if self.cond_channels > 0 else 0, Fint=c*2)
        self.dec3 = ResidualBlock(c * 8, c * 4)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.att2 = GuidedAttentionGate(Fg=c*2, Fl=c*2, Fc=c*2 if self.cond_channels > 0 else 0, Fint=c)
        self.dec2 = ResidualBlock(c * 4, c * 2)

        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.att1 = GuidedAttentionGate(Fg=c, Fl=c, Fc=c if self.cond_channels > 0 else 0, Fint=c//2)
        self.dec1 = ResidualBlock(c * 2, c)

        self.final = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x):
        # Split input
        t1 = x[:, :3, :, :]
        cond = x[:, 3:, :, :] if self.cond_channels > 0 else None

        # Main Encoder
        e1 = self.enc1(t1)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        # Cond Encoder
        if cond is not None:
            c1 = self.cond1(cond)
            c2 = self.cond2(self.c_pool1(c1))
            c3 = self.cond3(self.c_pool2(c2))
            c4 = self.cond4(self.c_pool3(c3))
        else:
            c1 = c2 = c3 = c4 = None

        # Decoder with Guided Attention
        d4 = self.up4(b)
        e4 = self.att4(d4, e4, c4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        e3 = self.att3(d3, e3, c3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        e2 = self.att2(d2, e2, c2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        e1 = self.att1(d1, e1, c1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1)