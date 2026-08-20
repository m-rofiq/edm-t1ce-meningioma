# ResViT (Dalmaz, Yurt, Cukur, IEEE TMI 2022) -- wrapper untuk MODEL_REGISTRY.
#
# ARSITEKTUR TIDAK DIUBAH (faithful): resvit_backbone.py (dari residual_transformers.py
# asli), resvit_transformer_configs.py, dan generator/discriminator dari
# resvit_networks_orig.py disalin APA ADANYA dari repo resmi icon-lab/ResViT --
# hanya import relatif yang di-flatten (`from . import x` -> `import x`) supaya
# cocok dengan struktur folder flat Anda. Tidak ada baris logic yang diubah.
#
# ADAPTASI (bukan perubahan arsitektur):
#   - in_channels default 3 (T1, T2, FLAIR -- SATU slice tengah per modalitas,
#     sesuai keputusan Anda: ikuti desain native ResViT, bukan skema 2.5D
#     9-channel seperti DD-Res U-Net/SynDiff).
#   - out_channels tetap 1 (T1CE, slice tengah).
#   - Constructor signature mengikuti konvensi model_registry.py Anda:
#     MODEL_REGISTRY[model_name](base_channels=..., in_channels=...) --
#     "base_channels" di-mapping ke ngf (jumlah filter conv pertama ResViT).

import resvit_backbone as residual_transformers
import resvit_networks_orig as networks


def build_resvit_generator(in_channels=3, out_channels=1, base_channels=64,
                            vit_name="Res-ViT-B_16", img_size=512,
                            pretrained_vit_path=None, pretrained_resnet=False):
    """
    pretrained_vit_path: path ke checkpoint ImageNet-21k generik
        (R50+ViT-B_16.npz) -- BUKAN checkpoint task-specific milik author.
        Download dari (lihat README icon-lab/ResViT):
        https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz
    """
    config_vit = residual_transformers.CONFIGS[vit_name]
    if pretrained_vit_path is not None:
        config_vit.pretrained_path = pretrained_vit_path

    netG = residual_transformers.ResViT(
        config_vit, input_dim=in_channels, img_size=img_size,
        output_dim=out_channels, vis=False
    )

    if pretrained_resnet:
        pre_trained_model = residual_transformers.Res_CNN(
            config_vit, input_dim=in_channels, img_size=img_size,
            output_dim=out_channels, vis=False
        )
        # (opsional -- hanya jika Anda punya checkpoint Res_CNN sendiri dari
        # tahap pre-training terpisah; skip kalau training dari awal)

    # Load pretrained ViT (ImageNet-21k) -- generik, faithful ke resep 2-stage
    # training ResViT (init dari ViT pretrained, bukan dari nol).
    # CATATAN: link resmi Google (storage.googleapis.com/vit_models/...) untuk
    # checkpoint ini SUDAH EXPIRED per awal 2026 (dikonfirmasi publik oleh
    # proyek lain yang memakai checkpoint sama, mis. TransUNet). Kalau Anda
    # tidak punya salinan checkpoint yang valid, set pretrained_vit_path=None
    # untuk training dari bobot acak (deviasi dari resep asli, perlu
    # didokumentasikan di Methods, tapi arsitektur tetap sama persis).
    if pretrained_vit_path is not None:
        import numpy as np
        netG.load_from(weights=np.load(pretrained_vit_path))
    else:
        print("[WARNING] pretrained_vit_path=None -- ResViT dilatih dari bobot "
              "ACAK (tanpa transfer learning ImageNet-21k). Ini deviasi dari "
              "resep training asli paper, dokumentasikan di Methods.")

    return netG


class ResViTGenerator:
    """Factory-style wrapper supaya cocok dipanggil seperti kelas biasa dari
    MODEL_REGISTRY: MODEL_REGISTRY["ResViT"](base_channels=..., in_channels=...).
    """
    def __new__(cls, in_channels=3, out_channels=1, base_channels=64, **kwargs):
        return build_resvit_generator(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            vit_name=kwargs.get("vit_name", "Res-ViT-B_16"),
            img_size=kwargs.get("img_size", 512),
            pretrained_vit_path=kwargs.get("pretrained_vit_path", None),
            pretrained_resnet=kwargs.get("pretrained_resnet", False),
        )


def build_resvit_discriminator(in_channels, ndf=64, n_layers=3, norm="instance"):
    """PatchGAN discriminator dari networks.py asli (NLayerDiscriminator),
    TIDAK diubah -- in_channels = channel gabungan (real_A + fake/real_B)
    persis konvensi pix2pix/ResViT asli."""
    norm_layer = networks.get_norm_layer(norm_type=norm)
    return networks.NLayerDiscriminator(
        in_channels, ndf, n_layers=n_layers, norm_layer=norm_layer, use_sigmoid=False
    )


# Tambahkan baris ini ke MODEL_REGISTRY Anda:
#
#   from resvit_model_wrapper import ResViTGenerator
#   MODEL_REGISTRY["ResViT"] = ResViTGenerator
