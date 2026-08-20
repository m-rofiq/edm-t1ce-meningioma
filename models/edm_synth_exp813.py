import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================
# RES BLOCK
# =====================================================
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_ch)
        self.act   = nn.LeakyReLU(0.2, inplace=True)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + identity)


# =====================================================
# CBAM
# =====================================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=8):
        super().__init__()
        mid = max(channels // reduction_ratio, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return x * torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        nn.init.xavier_normal_(self.conv.weight)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * scale


class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=8, spatial_kernel=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction_ratio)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x):
        return self.spatial_att(self.channel_att(x))


# =====================================================
# CROSS ATTENTION
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
# UP BLOCK + CBAM
# =====================================================
class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, use_cbam=True):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 1)
        )
        self.block = ResBlock(out_ch + skip_ch, out_ch)
        self.cbam  = CBAM(out_ch) if use_cbam else nn.Identity()

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.cbam(self.block(torch.cat([x, skip], dim=1)))


# =====================================================
# TWO-STAGE BRIDGE
# =====================================================
class TwoStageBridge(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.InstanceNorm2d(mid_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu', a=0.2)
                if m.weight.data.shape[1] > 256:
                    m.weight.data *= 0.5

    def forward(self, x):
        return self.stage2(self.stage1(x))


# =====================================================
# SAFE SKIP PROJECTION
# =====================================================
class SkipProjection(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.norm = nn.InstanceNorm2d(out_ch)
        self.act  = nn.LeakyReLU(0.2, inplace=True)
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out',
                                nonlinearity='leaky_relu', a=0.2)
        if in_ch > 128:
            self.proj.weight.data *= 0.5

    def forward(self, x):
        return self.act(self.norm(self.proj(x)))


# =====================================================
# MODEL — EDMSynth EXP-813
# Identik dengan EXP-812 / EXP-811.
# Semua perubahan ada di loss.
# =====================================================
class EDMSynth(nn.Module):
    def __init__(self, base_channels=32, in_channels=None, use_t2=True, use_flair=True):
        super().__init__()

        ec = 48
        dc = 32

        self.e1 = ResBlock(3,    ec);     self.p1 = nn.MaxPool2d(2)
        self.e2 = ResBlock(ec,   ec*2);   self.p2 = nn.MaxPool2d(2)
        self.e3 = ResBlock(ec*2, ec*4);   self.p3 = nn.MaxPool2d(2)
        self.e4 = ResBlock(ec*4, ec*8);   self.p4 = nn.MaxPool2d(2)
        self.b1 = ResBlock(ec*8, ec*16)

        cond_ch = 0
        if use_t2:    cond_ch += 3
        if use_flair: cond_ch += 3
        self.has_cond = cond_ch > 0

        if self.has_cond:
            self.ce1 = ResBlock(cond_ch, ec);   self.cp1 = nn.MaxPool2d(2)
            self.ce2 = ResBlock(ec,   ec*2);     self.cp2 = nn.MaxPool2d(2)
            self.ce3 = ResBlock(ec*2, ec*4);     self.cp3 = nn.MaxPool2d(2)
            self.ce4 = ResBlock(ec*4, ec*8);     self.cp4 = nn.MaxPool2d(2)
            self.b2  = ResBlock(ec*8, ec*16)
            self.attn_b = CrossAttention2D(ec*16)
            self.attn4  = CrossAttention2D(ec*8)
            self.attn3  = CrossAttention2D(ec*4)

        self.bridge = TwoStageBridge(ec*16, ec*16//2, dc*8)

        self.skip_proj4 = SkipProjection(ec*8,  dc*8)
        self.skip_proj3 = SkipProjection(ec*4,  dc*4)
        self.skip_proj2 = SkipProjection(ec*2,  dc*2)
        self.skip_proj1 = SkipProjection(ec,    dc)

        self.up4 = UpBlock(dc*8, dc*8, dc*8, use_cbam=True)
        self.up3 = UpBlock(dc*8, dc*4, dc*4, use_cbam=True)
        self.up2 = UpBlock(dc*4, dc*2, dc*2, use_cbam=True)
        self.up1 = UpBlock(dc*2, dc,   dc,   use_cbam=True)

        self.final = nn.Conv2d(dc, 1, 1)
        self.ds4   = nn.Conv2d(dc*8, 1, 1)
        self.ds3   = nn.Conv2d(dc*4, 1, 1)

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

        bn = self.bridge(bottleneck)
        s4 = self.skip_proj4(e4)
        s3 = self.skip_proj3(e3)
        s2 = self.skip_proj2(e2)
        s1 = self.skip_proj1(e1)

        d4 = self.up4(bn, s4)
        d3 = self.up3(d4, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)

        pred = self.final(d1)

        _aux4 = F.interpolate(self.ds4(d4), size=pred.shape[-2:],
                              mode="bilinear", align_corners=False)
        _aux3 = F.interpolate(self.ds3(d3), size=pred.shape[-2:],
                              mode="bilinear", align_corners=False)

        return pred, pred, torch.zeros_like(pred)
