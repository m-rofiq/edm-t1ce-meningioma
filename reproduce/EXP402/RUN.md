# EXP-402

Self-contained snapshot of the code used for this run.

## Provenance

| File in this folder | Source file in the repository | md5 |
|---|---|---|
| `configs/config.py` | `configs/config_exp400_feature_fusion.py` | 52acd956 |
| `models/dmec_net_step1.py` | `models/dmec_net_step1.py` | ffe22eaf |
| `models/model_registry.py` | `models/model_registry.py` | ccf48ffb |
| `losses/loss_registry.py` | `losses/loss_registry.py` | 5322a2c9 |
| `train.py` | `train_feature_fusion.py` | fc038d11 |

## Note

No file was archived under this identifier. The run altered the configuration only. The configuration file shown here also defines EXP-400, EXP-401 and EXP-403 to EXP-408, none of which is reported in the manuscript.

## How to run

Linux and macOS:

```bash
cd reproduce/EXP402
PYTHONPATH=../.. python train.py
```

Windows PowerShell:

```powershell
cd reproduce\EXP402
$env:PYTHONPATH = (Resolve-Path ..\..).Path
python train.py
```

The current directory is searched before the repository root, so
`configs.config`, `models.edm_synth` and `losses.loss_registry` resolve to
the files in this folder. The shared modules `datasets, utils, evaluators` resolve
from the repository root. No import statement was edited.

The imaging data are not included. The run requires the restricted
archive in the layout described in `docs/DATA_LAYOUT.md`.

