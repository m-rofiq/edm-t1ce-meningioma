# EXP-706

Self-contained snapshot of the code used for this run.

## Provenance

| File in this folder | Source file in the repository | md5 |
|---|---|---|
| `configs/config.py` | `configs/config_exp706_eval_metric.py` | 7160de96 |
| `models/edm_synth.py` | `models/edm_synth_exp706_eval_metric.py` | d79fa27e |
| `models/model_registry.py` | `models/model_registry.py` | ccf48ffb |
| `losses/loss_registry.py` | `losses/loss_registry_sd_exp706.py` | 08b4d285 |
| `train.py` | `train_gan_juga_vgg.py` | deac9287 |

## Note

The archived loss registry was named loss_registry_sd_exp706.py and was renamed to loss_registry.py at run time. The training script is inferred from file timestamps and is not archived under this identifier.

## How to run

Linux and macOS:

```bash
cd reproduce/EXP706
PYTHONPATH=../.. python train.py
```

Windows PowerShell:

```powershell
cd reproduce\EXP706
$env:PYTHONPATH = (Resolve-Path ..\..).Path
python train.py
```

The current directory is searched before the repository root, so
`configs.config`, `models.edm_synth` and `losses.loss_registry` resolve to
the files in this folder. The shared modules `datasets, utils, evaluators` resolve
from the repository root. No import statement was edited.

The imaging data are not included. The run requires the restricted
archive in the layout described in `docs/DATA_LAYOUT.md`.

