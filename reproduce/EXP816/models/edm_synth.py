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
# [NEW EXP-814] ENHANCEMENT SEGMENTATION HEAD
# ------------------------------------------------------
# Motivasi (dari analisis EXP-812/813):
#   Model gagal mensintesis focal bright enhancement
#   karena tidak ada mekanisme eksplisit untuk mengetahui
#   *di mana* enhancement terjadi secara spasial.
#
# Solusi: Enhancement Segmentation Head (EnhSegHead)
#   - Dijalankan dari feature bottleneck + decoder d3/d4
#   - Memprediksi binary enhancement probability map [0,1]
#   - Ditraining dengan BCEWithLogitsLoss vs GT diff mask
#   - Outputnya digunakan sebagai SPATIAL GATE di decoder:
#     decoder feature × (1 + gate_weight × enh_map)
#     → region enhancement mendapat amplifikasi sinyal
#     → region non-enhancement tetap normal
#
# Ini secara eksplisit mengajarkan model:
#   "di sini ada enhancement (segmentasi) →
#    fokus lebih kuat di sini saat decode (gating)"
# =====================================================
class EnhSegHead(nn.Module):
    """
    Memprediksi enhancement probability map dari
    gabungan fitur bottleneck dan decoder menengah.
    Output: [B, 1, H, W] logits (sebelum sigmoid)
    """
    def __init__(self, in_ch):
        super().__init__()
        mid = max(in_ch // 4, 16)
        self.enh_head = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1, bias=False),
            nn.InstanceNorm2d(mid),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid, mid // 2, 3, padding=1, bias=False),
            nn.InstanceNorm2d(mid // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid // 2, 1, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu', a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Inisialisasi bias layer akhir ke -2.0
        # agar sigmoid awal ≈ 0.12 → model mulai dengan prior
        # bahwa enhancement jarang (mayoritas background)
        last_conv = self.enh_head[-1]
        if last_conv.bias is not None:
            nn.init.constant_(last_conv.bias, -2.0)

    def forward(self, x):
        return self.enh_head(x)


# =====================================================
# [NEW EXP-814] ENHANCEMENT GATE
# ------------------------------------------------------
# Mengaplikasikan enhancement map sebagai spatial gate
# pada feature map decoder.
#
# Formula:
#   gated = feat × (1 + gate_weight × enh_prob)
#
# Interpretasi:
#   - enh_prob ≈ 0  → gated ≈ feat (tidak ada efek)
#   - enh_prob ≈ 1  → gated ≈ feat × (1 + gate_weight)
#     → amplifikasi sinyal di region enhancement
#
# Gate diterapkan di up1 (resolusi penuh) karena di sinilah
# detail spasial yang menentukan kualitas visual.
# =====================================================
class EnhancementGate(nn.Module):
    def __init__(self, feat_ch, gate_weight=0.70):
        super().__init__()
        self.gate_weight = gate_weight
        # Lightweight projection untuk menyesuaikan
        # channel enh_prob (1ch) ke feat_ch — lebih ekspresif
        # daripada broadcast langsung
        self.enh_gate = nn.Sequential(
            nn.Conv2d(1, feat_ch, 1, bias=False),
            nn.Sigmoid()
        )
        nn.init.constant_(self.enh_gate[0].weight,
                          gate_weight / feat_ch)

    def forward(self, feat, enh_logits):
        # Resize enh_logits ke resolusi feat jika berbeda
        if enh_logits.shape[-2:] != feat.shape[-2:]:
            enh_logits = F.interpolate(
                enh_logits, size=feat.shape[-2:],
                mode="bilinear", align_corners=False
            )
        enh_prob = torch.sigmoid(enh_logits)
        gate     = self.enh_gate(enh_prob)
        return feat * (1.0 + gate)


# =====================================================
# MODEL — EDMSynth EXP-814
# [CHANGED vs EXP-813]:
#   - Tambah EnhSegHead: prediksi enhancement mask
#     dari gabungan fitur d3+d4 (decoder menengah)
#   - Tambah EnhancementGate: gate spasial di d1
#     menggunakan output EnhSegHead
#   - forward() sekarang return 4-tuple:
#     (pred, enh_logits, pred_as_base, zeros_as_enh)
#     enh_logits digunakan loss BCE di trainer
#   - Semua komponen lama (encoder, bridge, skip_proj,
#     up1-4, CBAM, CrossAttention) identik EXP-813
# =====================================================
class EDMSynth(nn.Module):
    def __init__(self, base_channels=32, in_channels=None,
                 use_t2=True, use_flair=True,
                 gate_weight=0.70):
        super().__init__()

        ec = 48
        dc = 32

        # --- main encoder ---
        self.e1 = ResBlock(3,    ec);     self.p1 = nn.MaxPool2d(2)
        self.e2 = ResBlock(ec,   ec*2);   self.p2 = nn.MaxPool2d(2)
        self.e3 = ResBlock(ec*2, ec*4);   self.p3 = nn.MaxPool2d(2)
        self.e4 = ResBlock(ec*4, ec*8);   self.p4 = nn.MaxPool2d(2)
        self.b1 = ResBlock(ec*8, ec*16)

        # --- conditional encoder ---
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

        # --- bridge + skip proj ---
        self.bridge     = TwoStageBridge(ec*16, ec*16//2, dc*8)
        self.skip_proj4 = SkipProjection(ec*8,  dc*8)
        self.skip_proj3 = SkipProjection(ec*4,  dc*4)
        self.skip_proj2 = SkipProjection(ec*2,  dc*2)
        self.skip_proj1 = SkipProjection(ec,    dc)

        # --- decoder ---
        self.up4 = UpBlock(dc*8, dc*8, dc*8, use_cbam=True)
        self.up3 = UpBlock(dc*8, dc*4, dc*4, use_cbam=True)
        self.up2 = UpBlock(dc*4, dc*2, dc*2, use_cbam=True)
        self.up1 = UpBlock(dc*2, dc,   dc,   use_cbam=True)

        # --- [NEW] Enhancement Segmentation Head ---
        # Input: concat(d3, d4_upsampled) = dc*4 + dc*8 = 128 + 256 = 384ch
        self.enh_head = EnhSegHead(in_ch=dc*4 + dc*8)

        # --- [NEW] Enhancement Gate di d1 ---
        self.enh_gate = EnhancementGate(feat_ch=dc, gate_weight=gate_weight)

        # --- output heads ---
        self.final = nn.Conv2d(dc, 1, 1)
        self.ds4   = nn.Conv2d(dc*8, 1, 1)
        self.ds3   = nn.Conv2d(dc*4, 1, 1)

    def forward(self, x):
        t1   = x[:, :3]
        cond = x[:, 3:] if x.shape[1] > 3 else None

        # --- encode ---
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

        # --- bridge + project ---
        bn = self.bridge(bottleneck)
        s4 = self.skip_proj4(e4)
        s3 = self.skip_proj3(e3)
        s2 = self.skip_proj2(e2)
        s1 = self.skip_proj1(e1)

        # --- decode ---
        d4 = self.up4(bn, s4)
        d3 = self.up3(d4, s3)
        d2 = self.up2(d3, s2)

        # --- [NEW] Enhancement Segmentation Head ---
        # Gabungkan d3 dan d4 (upsample d4 ke resolusi d3)
        d4_up = F.interpolate(d4, size=d3.shape[-2:],
                              mode="bilinear", align_corners=False)
        enh_feat    = torch.cat([d3, d4_up], dim=1)   # [B, dc*8, H/4, W/4]
        enh_logits  = self.enh_head(enh_feat)          # [B, 1, H/4, W/4]

        # --- [NEW] Enhancement Gate di d1 ---
        d1_raw = self.up1(d2, s1)                      # [B, dc, H, W]
        d1     = self.enh_gate(d1_raw, enh_logits)     # gated feature

        pred = self.final(d1)

        # Return 4-tuple: (pred, enh_logits, base, dummy_enh)
        # enh_logits digunakan loss BCE di trainer
        return pred, enh_logits, pred, torch.zeros_like(pred)
