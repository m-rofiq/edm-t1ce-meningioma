# EXP-602

Self-contained snapshot of the code used for this run.

## Provenance

| File in this folder | Source file in the repository | md5 |
|---|---|---|
| `configs/config.py` | `configs/config_exp602.py` | bd6f61e4 |
| `models/edm_synth.py` | `models/edm_synth_exp602.py` | 47a0436b |
| `models/model_registry.py` | `models/model_registry_exp601_exp602.py` | af3c586c |
| `models/resattention_unet.py` | `models/resattention_unet_exp303.py` | 70857791 |
| `models/resunet.py` | `models/resunet.py` | 50136a57 |
| `losses/loss_registry.py` | `losses/loss_registry_exp602.py` | 879b6854 |
| `losses/edm_loss.py` | `losses/edm_loss.py` | 9044532c |
| `train.py` | `train_edm_exp602.py` | 80e3cc80 |

## Note

The training script is byte-identical to the one used for EXP-601.

## How to run

Linux and macOS:

```bash
cd reproduce/EXP602
PYTHONPATH=../.. python train.py
```

Windows PowerShell:

```powershell
cd reproduce\EXP602
$env:PYTHONPATH = (Resolve-Path ..\..).Path
python train.py
```

The current directory is searched before the repository root, so
`configs.config`, `models.edm_synth` and `losses.loss_registry` resolve to
the files in this folder. The shared modules `datasets, utils, evaluators` resolve
from the repository root. No import statement was edited.

The imaging data are not included. The run requires the restricted
archive in the layout described in `docs/DATA_LAYOUT.md`.

