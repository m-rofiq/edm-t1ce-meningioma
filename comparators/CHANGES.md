# External comparators

Three published methods were evaluated alongside the configurations developed in
this study. pGAN and ResViT were trained locally from the implementations
released by their authors, adapted to the input format and the intensity
convention of this study. DDResUNet was implemented for this study from the
architecture description of its publication. No released task-specific
checkpoint was used.

| Comparator | Reference | Upstream licence | Included here |
|---|---|---|---|
| pGAN | Dar et al. 2019 | MIT, ICON Lab 2019, plus BSD for pix2pix, CycleGAN and DCGAN | yes, in full |
| ResViT | Dalmaz et al. 2022 | MIT, ICON Lab 2021, plus BSD for pix2pix | yes, in full |
| DDResUNet | Osman and Tamam 2023 | upstream states none, so no upstream code is redistributed | own implementation, see `DDResUNet/README.md` |

## Adaptations

**pGAN.** The generator ends with a Tanh in the upstream implementation, which
bounds the output range and cannot represent z-scored targets. It was replaced
with a linear output. The network received nine input channels and one output
channel, with ten channels at its discriminator. The optimiser keeps the
published values, a learning rate of 2e-4 with beta_1 of 0.5. The checkpoint is
selected on validation PSNR minus half the LPIPS rather than on PSNR_ROI alone.
Mixed precision was disabled. Early stopping used a patience of 40.
The reported pGAN runs were launched from `pGAN/modif/`. The files at the root of
`pGAN/` are the upstream release together with later edits that postdate those
runs and were not used.

**ResViT.** The architecture was left unchanged. Four protocol deviations apply,
all stated in Section 2.3 and Supplementary Section S17 of the manuscript. No
early stopping was in force. Evaluation-path inference ran under automatic mixed
precision. Inputs were bilinearly downsampled to the native 256x256 grid and
predictions upsampled back before any metric was computed. The transformer stage
was initialised from the ImageNet-21k weights released with the architecture,
whereas the fourteen remaining configurations were trained from random
initialisation.

**DDResUNet.** Implemented in two dimensions rather than three, because the
dataset of this study is 2.5D per slice. The final ReLU of the published
architecture was replaced with a linear output, because z-scored targets are
unbounded and may be negative. The published learning rate of 1e-4 was
retained, so the protocol deviates in nothing else.

## Licence notice

The contents of `pGAN/` and `ResViT/` remain under the licences of their
upstream repositories, reproduced in `pGAN/LICENSE` and `ResViT/LICENSE`. The
Apache License 2.0 at the root of this repository does not apply to them.

## Upstream files that do not parse under current Python

Some files in `pGAN/` and `ResViT/` are reproduced as released and target an
older Python. `models/pgan_model.py`, `models/cgan_model.py`,
`models/test_model.py`, `models/resvit_one.py` and `models/resvit_many.py`
use `async=True`, which is a syntax error from Python 3.7 onwards. These files
were not used in this study, which ran through the adapted entry points, and
they are left unmodified so that the upstream code remains distinguishable from
the adaptations.