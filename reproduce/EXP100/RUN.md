# EXP-100

Self-contained snapshot of the code used for this run.

## Provenance

| File in this folder | Source file in the repository | md5 |
|---|---|---|
| `configs/config.py` | `configs/config_exp100_evaluasi_metric.py` | c58a245a |
| `models/unet_baseline.py` | `models/unet_baseline.py` | 7587293f |
| `models/model_registry.py` | `models/model_registry.py` | ccf48ffb |
| `losses/loss_registry.py` | `losses/loss_registry.py` | 5322a2c9 |
| `train.py` | `train_any_loss.py` | a4e922d4 |

## Note

Only the configuration was archived for this run. The model definition is the canonical file, unchanged across the baseline experiments. The training script is train_any_loss.py. A second file, train_baseline_awal.py, is identical to it except for one commented-out import line, so the two cannot be distinguished and produce the same result.

## How to run

Linux and macOS:

```bash
cd reproduce/EXP100
PYTHONPATH=../.. python train.py
```

Windows PowerShell:

```powershell
cd reproduce\EXP100
$env:PYTHONPATH = (Resolve-Path ..\..).Path
python train.py
```

The current directory is searched before the repository root, so
`configs.config`, `models.edm_synth` and `losses.loss_registry` resolve to
the files in this folder. The shared modules `datasets, utils, evaluators` resolve
from the repository root. No import statement was edited.

The imaging data are not included. The run requires the restricted
archive in the layout described in `docs/DATA_LAYOUT.md`.

