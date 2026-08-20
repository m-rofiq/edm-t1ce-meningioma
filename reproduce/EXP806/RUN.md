# EXP-806

Self-contained snapshot of the code used for this run.

## Provenance

| File in this folder | Source file in the repository | md5 |
|---|---|---|
| `configs/config.py` | `configs/config_exp806.py` | 8718444d |
| `models/edm_synth.py` | `models/edm_synth_exp806.py` | e232aac2 |
| `models/model_registry.py` | `models/model_registry.py` | ccf48ffb |
| `losses/loss_registry.py` | `losses/loss_registry_exp806.py` | f19db591 |
| `train.py` | `train_exp806.py` | b34ea524 |

## How to run

Linux and macOS:

```bash
cd reproduce/EXP806
PYTHONPATH=../.. python train.py
```

Windows PowerShell:

```powershell
cd reproduce\EXP806
$env:PYTHONPATH = (Resolve-Path ..\..).Path
python train.py
```

The current directory is searched before the repository root, so
`configs.config`, `models.edm_synth` and `losses.loss_registry` resolve to
the files in this folder. The shared modules `datasets, utils, evaluators` resolve
from the repository root. No import statement was edited.

The imaging data are not included. The run requires the restricted
archive in the layout described in `docs/DATA_LAYOUT.md`.

