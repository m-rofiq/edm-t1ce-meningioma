# DESTINATION: models/pgan_networks.py (letakkan di package `models/` proyek Anda)
#
# Generator + Discriminator resmi pGAN/cGAN (Dar et al., 2019, IEEE TMI),
# diporting dari icon-lab/pGAN-cGAN (models/networks.py) ke PyTorch modern.
#
# ADAPTASI vs kode resmi (lihat strategi_adaptasi_pGAN_cGAN.md bagian 2):
#   1) API PyTorch 0.2.0 -> modern (init.normal_, dst.), Python 2.7 -> 3.
#   2) input_nc/output_nc default disesuaikan ke setup 2.5D multi-modal Anda
#      (9 channel input = 3 modalitas x 3 slice tetangga, 1 channel output = T1CE center).
#   3) Tanh() akhir generator DIHAPUS (final_activation="identity") karena target
#      Anda z-score (unbounded), bukan dinormalisasi ke [-1,1] seperti dataset asli
#      pGAN (IXI, 0-1 -> (x-0.5)/0.5). Tanh dgn data z-score akan memotong sinyal
#      intensitas tinggi (mis. area tumor hyperintense) -> perbandingan tidak adil.
#
# Arsitektur inti (ResNet 9-block generator, PatchGAN 70x70 discriminator,
# InstanceNorm2d, reflection padding) TIDAK diubah dari kode resmi.

import torch
import torch.nn as nn
from torch.nn import init
import functools


def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == "none":
        return None
    else:
        raise NotImplementedError(f"normalization layer [{norm_type}] is not found")


def init_weights(net, init_type="normal", gain=0.02):
    """Porting init_weights resmi pGAN (models/networks.py) ke API PyTorch modern
    (init.normal -> init.normal_, dst.)."""

    def _init(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and ("Conv" in classname or "Linear" in classname):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(f"init method [{init_type}] not implemented")
            if getattr(m, "bias", None) is not None:
                init.constant_(m.bias.data, 0.0)
        elif "InstanceNorm2d" in classname or "BatchNorm2d" in classname:
            if getattr(m, "weight", None) is not None:
                init.normal_(m.weight.data, 1.0, gain)
            if getattr(m, "bias", None) is not None:
                init.constant_(m.bias.data, 0.0)

    net.apply(_init)
    return net


class ResnetBlock(nn.Module):
    """Residual block Johnson et al., identik dgn kode resmi pGAN/cGAN."""

    def __init__(self, dim, padding_type="reflect", norm_layer=nn.InstanceNorm2d,
                 use_dropout=False, use_bias=True):
        super().__init__()
        self.conv_block = self._build(dim, padding_type, norm_layer, use_dropout, use_bias)

    def _pad(self, padding_type, layers):
        if padding_type == "reflect":
            layers += [nn.ReflectionPad2d(1)]
            p = 0
        elif padding_type == "replicate":
            layers += [nn.ReplicationPad2d(1)]
            p = 0
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"padding [{padding_type}] is not implemented")
        return p

    def _build(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        layers = []
        p = self._pad(padding_type, layers)
        layers += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                   norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            layers += [nn.Dropout(0.5)]

        p = self._pad(padding_type, layers)
        layers += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                   norm_layer(dim)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv_block(x)


class PGANGenerator(nn.Module):
    """
    ResNet generator (9 residual block default) dari pGAN/cGAN.

    Args:
        input_nc: default 9 = 3 modalitas (T1,T2,FLAIR) x 3 slice tetangga (2.5D).
                  Sesuaikan otomatis di train_pGAN.py via 3*len(CONFIG["modalities"]).
        output_nc: default 1 = T1CE center slice.
        final_activation: "identity" (default, sesuai adaptasi utk data z-score)
                           atau "tanh" (kalau ingin replikasi 100% kode asli dgn
                           data dinormalisasi ulang ke [-1,1] -- TIDAK direkomendasikan
                           utk pipeline Anda, lihat strategi doc bagian 2c).
    """

    def __init__(self, input_nc=9, output_nc=1, ngf=64, norm="instance",
                 use_dropout=False, n_blocks=9, final_activation="identity"):
        super().__init__()
        assert n_blocks >= 0
        norm_layer = get_norm_layer(norm)
        use_bias = (norm == "instance")

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf), nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2,
                                 padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                   use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3,
                                          stride=2, padding=1, output_padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]

        if final_activation == "tanh":
            model += [nn.Tanh()]
        elif final_activation != "identity":
            raise NotImplementedError(f"final_activation [{final_activation}] not implemented")

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class PGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator (NLayerDiscriminator) dari pGAN/cGAN.
    Conditional: menerima concat(cond, img) mengikuti konvensi pix2pix/pGAN.

    in_channels HARUS = input_nc (kondisi/modalitas) + output_nc (target/prediksi).
    Utk setup Anda: 9 + 1 = 10 -- persis sama dgn discriminator.in_channels=10
    di EXP-703 Anda (konsistensi informasi discriminator antar baseline).
    """

    def __init__(self, in_channels=10, base_channels=64, n_layers=3, norm="instance",
                 use_sigmoid=False):
        super().__init__()
        norm_layer = get_norm_layer(norm)
        use_bias = (norm == "instance")

        kw, padw = 4, 1
        sequence = [nn.Conv2d(in_channels, base_channels, kernel_size=kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2 ** n, 8)
            sequence += [nn.Conv2d(base_channels * nf_mult_prev, base_channels * nf_mult,
                                    kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                         norm_layer(base_channels * nf_mult), nn.LeakyReLU(0.2, True)]

        nf_mult_prev, nf_mult = nf_mult, min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(base_channels * nf_mult_prev, base_channels * nf_mult,
                                kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                     norm_layer(base_channels * nf_mult), nn.LeakyReLU(0.2, True)]

        sequence += [nn.Conv2d(base_channels * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, cond, img):
        x = torch.cat([cond, img], dim=1)
        return self.model(x)
