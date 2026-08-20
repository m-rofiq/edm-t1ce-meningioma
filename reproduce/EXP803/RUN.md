# EXP-803

Self-contained snapshot of the code used for this run.

## Provenance

| File in this folder | Source file in the repository | md5 |
|---|---|---|
| `configs/config.py` | `configs/config_exp803.py` | b62fb250 |
| `models/edm_synth.py` | `models/edm_synth_exp803_sd_exp805.py` | 1d10a1d8 |
| `models/model_registry.py` | `models/model_registry.py` | ccf48ffb |
| `losses/loss_registry.py` | `losses/loss_registry_exp803.py` | da217b6b |
| `train.py` | `train_exp803.py` | 47d30655 |

## Note

The archived model file is named edm_synth_exp803_sd_exp805.py. The suffix refers to a later exploration variant that is not reported in the manuscript.

## How to run

Linux and macOS:

```bash
cd reproduce/EXP803
PYTHONPATH=../.. python train.py
```

Windows PowerShell:

```powershell
cd reproduce\EXP803
$env:PYTHONPATH = (Resolve-Path ..\..).Path
python train.py
```

The current directory is searched before the repository root, so
`configs.config`, `models.edm_synth` and `losses.loss_registry` resolve to
the files in this folder. The shared modules `datasets, utils, evaluators` resolve
from the repository root. No import statement was edited.

The imaging data are not included. The run requires the restricted
archive in the layout described in `docs/DATA_LAYOUT.md`.

