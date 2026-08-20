# EXP-716A

Self-contained snapshot of the code used for this run.

## Provenance

| File in this folder | Source file in the repository | md5 |
|---|---|---|
| `configs/config.py` | `configs/config_exp716A.py` | 2f7a0456 |
| `models/edm_synth.py` | `models/edm_synth_exp716A.py` | a23d99ba |
| `models/model_registry.py` | `models/model_registry.py` | ccf48ffb |
| `losses/loss_registry.py` | `losses/loss_registry_exp716A.py` | ef8c459f |
| `train.py` | `train_gan_juga_vgg.py` | deac9287 |

## Note

The training script is inferred from file timestamps and is not archived under this identifier.

## How to run

Linux and macOS:

```bash
cd reproduce/EXP716A
PYTHONPATH=../.. python train.py
```

Windows PowerShell:

```powershell
cd reproduce\EXP716A
$env:PYTHONPATH = (Resolve-Path ..\..).Path
python train.py
```

The current directory is searched before the repository root, so
`configs.config`, `models.edm_synth` and `losses.loss_registry` resolve to
the files in this folder. The shared modules `datasets, utils, evaluators` resolve
from the repository root. No import statement was edited.

The imaging data are not included. The run requires the restricted
archive in the layout described in `docs/DATA_LAYOUT.md`.

