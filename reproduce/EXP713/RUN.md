# EXP-713

Self-contained snapshot of the code used for this run.

## Provenance

| File in this folder | Source file in the repository | md5 |
|---|---|---|
| `configs/config.py` | `configs/config_exp713.py` | 15570933 |
| `models/edm_synth.py` | `models/edm_synth_exp713.py` | 1d10a1d8 |
| `models/model_registry.py` | `models/model_registry.py` | ccf48ffb |
| `losses/loss_registry.py` | `losses/loss_registry_exp713.py` | 01caa918 |
| `train.py` | `train_exp713.py` | 2fcb81b5 |

## Note

The archived files ending in _cek are byte-identical duplicates and are not reproduced here.

## How to run

Linux and macOS:

```bash
cd reproduce/EXP713
PYTHONPATH=../.. python train.py
```

Windows PowerShell:

```powershell
cd reproduce\EXP713
$env:PYTHONPATH = (Resolve-Path ..\..).Path
python train.py
```

The current directory is searched before the repository root, so
`configs.config`, `models.edm_synth` and `losses.loss_registry` resolve to
the files in this folder. The shared modules `datasets, utils, evaluators` resolve
from the repository root. No import statement was edited.

The imaging data are not included. The run requires the restricted
archive in the layout described in `docs/DATA_LAYOUT.md`.

