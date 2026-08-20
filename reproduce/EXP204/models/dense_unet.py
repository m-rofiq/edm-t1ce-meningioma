import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):

    def __init__(self, in_ch, growth):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, growth, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(growth)

        self.conv2 = nn.Conv2d(in_ch + growth, growth, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(growth)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(torch.cat([x, x1], dim=1))))

        out = torch.cat([x, x1, x2], dim=1)

        return out


class TransitionDown(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):

        x = self.bn(self.conv(x))
        x = self.pool(x)

        return x


class TransitionUp(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

    def forward(self, x):
        return self.up(x)


class DenseUNet(nn.Module):

    def __init__(self, base_channels=32, in_channels=3, out_channels=1):

        super().__init__()

        g = base_channels // 2

        # =========================
        # Encoder
        # =========================

        self.db1 = DenseBlock(in_channels, g)
        ch1 = in_channels + 2 * g

        self.td1 = TransitionDown(ch1, base_channels)

        self.db2 = DenseBlock(base_channels, g)
        ch2 = base_channels + 2 * g

        self.td2 = TransitionDown(ch2, base_channels * 2)

        self.db3 = DenseBlock(base_channels * 2, g)
        ch3 = base_channels * 2 + 2 * g

        self.td3 = TransitionDown(ch3, base_channels * 4)

        self.db4 = DenseBlock(base_channels * 4, g)
        ch4 = base_channels * 4 + 2 * g

        self.td4 = TransitionDown(ch4, base_channels * 8)

        # =========================
        # Bottleneck
        # =========================

        self.center = DenseBlock(base_channels * 8, g)
        ch_center = base_channels * 8 + 2 * g

        # =========================
        # Decoder
        # =========================

        self.up4 = TransitionUp(ch_center, base_channels * 4)
        self.db_up4 = DenseBlock(base_channels * 4 + ch4, g)

        ch_up4 = base_channels * 4 + ch4 + 2 * g

        self.up3 = TransitionUp(ch_up4, base_channels * 2)
        self.db_up3 = DenseBlock(base_channels * 2 + ch3, g)

        ch_up3 = base_channels * 2 + ch3 + 2 * g

        self.up2 = TransitionUp(ch_up3, base_channels)
        self.db_up2 = DenseBlock(base_channels + ch2, g)

        ch_up2 = base_channels + ch2 + 2 * g

        self.up1 = TransitionUp(ch_up2, base_channels)
        self.db_up1 = DenseBlock(base_channels + ch1, g)

        ch_up1 = base_channels + ch1 + 2 * g

        # =========================
        # Output
        # =========================

        self.final = nn.Conv2d(ch_up1, out_channels, 1)

    def forward(self, x):

        # encoder
        e1 = self.db1(x)
        e2 = self.db2(self.td1(e1))
        e3 = self.db3(self.td2(e2))
        e4 = self.db4(self.td3(e3))

        # bottleneck
        center = self.center(self.td4(e4))

        # decoder
        d4 = self.up4(center)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.db_up4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.db_up3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.db_up2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.db_up1(d1)

        out = self.final(d1)

        return out