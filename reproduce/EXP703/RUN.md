# EXP-703

Self-contained snapshot of the code used for this run.

## Provenance

| File in this folder | Source file in the repository | md5 |
|---|---|---|
| `configs/config.py` | `configs/config_exp703rerun3.py` | b94e8c47 |
| `models/edm_synth.py` | `models/edm_synth_exp703rerun3.py` | 4dafac48 |
| `models/model_registry.py` | `models/model_registry.py` | ccf48ffb |
| `losses/loss_registry.py` | `losses/loss_registry_exp703rerun3.py` | 5322a2c9 |
| `train.py` | `train_exp703rerun3.py` | 1a10d0b1 |

## Note

Three archived versions exist. The version reproduced here matches config_snapshot.json on early_stop_patience 40, amp False and lr_D 5e-05. The two earlier versions differ and were not used.

## How to run

Linux and macOS:

```bash
cd reproduce/EXP703
PYTHONPATH=../.. python train.py
```

Windows PowerShell:

```powershell
cd reproduce\EXP703
$env:PYTHONPATH = (Resolve-Path ..\..).Path
python train.py
```

The current directory is searched before the repository root, so
`configs.config`, `models.edm_synth` and `losses.loss_registry` resolve to
the files in this folder. The shared modules `datasets, utils, evaluators` resolve
from the repository root. No import statement was edited.

The imaging data are not included. The run requires the restricted
archive in the layout described in `docs/DATA_LAYOUT.md`.

