import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """
    Minimal PatchGAN Discriminator (Stable Version)

    Supports:
    - Conditional GAN (default)
    - Unconditional GAN (optional)

    Input:
        x: input modalities (B, C_in, H, W)   -> optional
        y: image (real/fake) (B, 1, H, W)

    Output:
        patch map (B, 1, H/16, W/16)
    """

    def __init__(
        self,
        in_channels=10,   # modalities + 1 (T1CE)
        base_channels=64,
        use_spectral_norm=True
    ):
        super().__init__()

        def conv_block(in_c, out_c, stride, normalize=True):
            layers = []

            conv = nn.Conv2d(
                in_c,
                out_c,
                kernel_size=4,
                stride=stride,
                padding=1
            )

            if use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)

            layers.append(conv)

            if normalize:
                layers.append(nn.InstanceNorm2d(out_c))

            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return nn.Sequential(*layers)

        # ---- Architecture ----
        self.model = nn.Sequential(
            # (B, in_channels, H, W)
            conv_block(in_channels, base_channels, stride=2, normalize=False),  # no norm first layer
            conv_block(base_channels, base_channels * 2, stride=2),
            conv_block(base_channels * 2, base_channels * 4, stride=2),
            conv_block(base_channels * 4, base_channels * 8, stride=1),

            # final patch output
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x, y):
        """
        x: modalities (B, C, H, W) OR None
        y: image (B, 1, H, W)
        """

        if x is not None:
            inp = torch.cat([x, y], dim=1)
        else:
            inp = y

        return self.model(inp)