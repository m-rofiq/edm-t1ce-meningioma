# EXP-500

Self-contained snapshot of the code used for this run.

## Provenance

| File in this folder | Source file in the repository | md5 |
|---|---|---|
| `configs/config.py` | `configs/config_exp500_guided_resattention.py` | 49cdf379 |
| `models/guided_resattention_unet.py` | `models/guided_resattention_unet.py` | bf3fde63 |
| `models/model_registry.py` | `models/model_registry.py` | ccf48ffb |
| `losses/loss_registry.py` | `losses/loss_registry.py` | 5322a2c9 |
| `train.py` | `train_guided_resattention.py` | fc038d11 |

## Note

Only the configuration was archived for this run.

## How to run

Linux and macOS:

```bash
cd reproduce/EXP500
PYTHONPATH=../.. python train.py
```

Windows PowerShell:

```powershell
cd reproduce\EXP500
$env:PYTHONPATH = (Resolve-Path ..\..).Path
python train.py
```

The current directory is searched before the repository root, so
`configs.config`, `models.edm_synth` and `losses.loss_registry` resolve to
the files in this folder. The shared modules `datasets, utils, evaluators` resolve
from the repository root. No import statement was edited.

The imaging data are not included. The run requires the restricted
archive in the layout described in `docs/DATA_LAYOUT.md`.

