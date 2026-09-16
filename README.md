# Enhancement-Disentangled Modelling for T1-CE Meningioma Synthesis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22042980.svg)](https://doi.org/10.5281/zenodo.22042980)

Analysis and modelling code for the study *Enhancement-Disentangled Modelling for
T1-CE Meningioma Synthesis: Enhancement-Region Fidelity and Intensity Calibration
at the Cost of Global Structural Similarity*.

The repository contains the architecture definitions of the configurations
evaluated on the locked hold-out cohort, the training and evaluation pipelines,
the loss functions, the metric definitions, and the statistical analysis that
produces the patient-level significance table reported as Supplementary Table S7.

**No patient imaging data are included.** The magnetic resonance examinations
analysed in the study are restricted by the approving ethics committee, approval
number 069/KEP/2025, and by institutional policy. **No trained model weights are
included**, because they are derived from that restricted imaging.

## Layout

| Path | Contents |
|---|---|
| `configs/` | configuration files, one per run, archived under the run identifier |
| `models/` | architecture definitions |
| `losses/` | loss functions and the loss registry |
| `datasets/` | the 2.5D slice dataset and the loader |
| `evaluators/` | metric definitions, masks, bootstrap intervals, statistical tests |
| `utils/` | early stopping, experiment bookkeeping, alignment audits |
| `tools/` | dataset integrity and slice-level audits |
| `experiments/` | per-run numerical results, one folder per run, no weights and no images |
| `reproduce/` | one self-contained folder per configuration, see below |
| `comparators/` | the external comparators, see `comparators/CHANGES.md` |
| `preprocessing/` | the chain that built the dataset, five numbered stages, see `docs/PREPROCESSING.md` |
| `docs/` | identifier mapping, data layout, preprocessing chain, residual-trace report |
| `bangun_reproduce.py` | script that assembled `reproduce/` |
| `ganti_jalur.py` | first pass, replaced absolute paths under the project root |
| `ganti_jalur_preprosesing.py` | second pass, added the preprocessing and comparator roots |
| `verify_S7.py` | regenerates Supplementary Table S7 and compares it with the deposited reference |
| `NOTICE` | copyright notice and the third-party licence exceptions |

`bangun_reproduce.py`, `ganti_jalur.py`, `ganti_jalur_preprosesing.py` and
`verify_S7.py` are the scripts that assembled and verified this deposit. They are
included for transparency and are not part of the training or evaluation
pipeline.

## Reproducing a single configuration

During the study the working files carried canonical names such as
`configs/config.py`, and each finished run was archived by appending the run
identifier to the filename. A reviewer reading the flat repository therefore
cannot tell which archived file belongs to which run.

`reproduce/` removes that ambiguity. Each folder restores the archived files of
one run to their canonical names, so the import statements of the training
script apply unchanged. No import statement was edited.

```bash
cd reproduce/EXP602
PYTHONPATH=../.. python train.py
```

```powershell
cd reproduce\EXP602
$env:PYTHONPATH = (Resolve-Path ..\..).Path
python train.py
```

Every folder carries a `RUN.md` giving the source file and the md5 of each file
it contains. `docs/MAPPING.md` maps every run identifier to the configuration
name used in the manuscript.

## Preprocessing

The preprocessing scripts are provided for inspection and audit. They cannot be
executed without the restricted imaging archive and are included so that the
preprocessing and masking decisions behind every reported number can be verified
line by line.

## Data

Place the dataset at `./data`. The expected layout is described in
`docs/DATA_LAYOUT.md`. Paths in `experiments/*/config_snapshot.json` are the
original machine paths of the run and are left unedited as a historical record.

## Citation

See `CITATION.cff`.

## Licence

Apache License 2.0, see `LICENSE`. The three external comparators under
`comparators/` retain the licences of their upstream repositories.
