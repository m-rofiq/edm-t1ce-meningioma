# DDResUNet

The implementation evaluated in this study is `ddresunet_model.py`, written for
this study from the architecture description given in the publication below. It
is original work and is covered by the licence at the root of this repository.

- Reference: Osman AFI, Tamam NM. Contrast-enhanced MRI synthesis using
  dense-dilated residual convolutions based 3D network toward elimination of
  gadolinium in neuro-oncology. J Appl Clin Med Phys. 2023;e14120.
  https://doi.org/10.1002/acm2.14120
- Repository released by the authors of that publication:
  https://github.com/afiosman/dense-dilated-residual-convolutions-for-contrast-enhanced-MRI-synthesis
  (commit 3aaecc5 at the time of this study)

**No code from that repository is redistributed here.** It carries no licence,
so under default copyright its authors retain all rights. It was consulted
during development and is cited for provenance only. The two remaining external
comparators, pGAN and ResViT, are released by their authors under the MIT
licence and are included in full, together with the licence files of their
upstream repositories.

## Differences from the published architecture

Two changes were required, both stated in Section 2.3 of the manuscript.

The network operates in two dimensions rather than three, because the dataset of
this study is 2.5D per slice.

The final ReLU activation was replaced with a linear output. This study
z-scores intensities, so target values are unbounded and may be negative, and a
ReLU output cannot represent them.

The training protocol fixed the data, the patient-level partition, the batch
size, the epoch budget and the metric definitions. The publication states a
learning rate of 1e-4, which equals the rate used by the twelve configurations
developed here, so this comparator deviates from the uniform protocol in
nothing.

## Contents

| File | Role |
|---|---|
| `ddresunet_model.py` | the architecture as implemented and evaluated here |
| `ddresunet_loss.py`, `loss_registry.py`, `model_registry.py` | objective and registries |
| `config_ddresunet_sota.py` | configuration of the reported runs |
| `train_ddresunet_sota_pytorch.py`, `run_experiments_all_fold.py` | training entry points |
| `dataset_2p5d.py`, `psnr_fixed_range.py`, `ssim_masked.py`, and similar | copies of the shared modules of this repository |
| `experiments/` | per-fold numerical results, no weights and no images |
