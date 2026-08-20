# Data layout

The imaging data are not distributed with this repository. They are restricted by
the approving ethics committee, approval number 069/KEP/2025, and by
institutional policy. This document states the layout the code expects, so that
the preprocessing and evaluation decisions can be verified line by line and so
that an equivalent dataset can be substituted.

## Directory tree

Place the dataset at `./data/dataset_5fold_final_v4` relative to the repository
root, or set the path in the configuration file of the run being reproduced.

```
data/dataset_5fold_final_v4/
├── fold_0/
│   ├── train/
│   │   ├── T1/
│   │   ├── T2/
│   │   ├── FLAIR/
│   │   └── T1CE/
│   └── val/
│       ├── T1/
│       ├── T2/
│       ├── FLAIR/
│       └── T1CE/
├── fold_1/          same structure
├── fold_2/          same structure
├── fold_3/          same structure
├── fold_4/          same structure
└── holdout_test/
    ├── T1/
    ├── T2/
    ├── FLAIR/
    └── T1CE/
```

`T1CE` is the prediction target. `T1`, `T2` and `FLAIR` are the inputs. The four
modality directories of any split must contain **identical filenames**. A slice
is admitted only if all four modalities are present and valid for it.

## File naming

```
<patient>_slice<NNN>.npy
```

Example: `03d50111_slice001.npy`

`<patient>` is the anonymised patient key, the first eight hexadecimal characters
of the SHA-1 hash of the original identifier. `<NNN>` is the slice index within
the patient, zero-padded to three digits and starting at `001`. The same key is
used as `patient_id` in every results table in `experiments/`, so a metric value
can be traced to the slices that produced it without any identifiable
information.

## Array format

Each `.npy` file holds one 2D slice as a single-precision float array. Intensities
are z-scored per modality and per patient, so values are unbounded and may be
negative. Two consequences follow, both of which the code depends on.

The brain region `M_ROI` is defined as `arr > 0` on the reference volume of the
modality concerned. It is an intensity threshold, not a skull-stripping mask.

The enhancement region `M_Enh` thresholds the reference at
`mu_ROI(y) + 1.5 * sigma_ROI(y)`. It is an intensity threshold and not a tumour
segmentation, which is why the manuscript refers to a hyperintense region
throughout.

Networks receive a 2.5D input: the slice together with its two neighbours, for
each of the three input modalities, giving nine input channels and one output
channel.

## Expected counts

The cohort reported in the manuscript gives the following. Use these to check a
substituted dataset for shape, not for identity.

| Quantity | Value |
|---|---|
| Patients acquired | 64 |
| Patients excluded | 1 |
| Patients analysed | 63 |
| Cross-validation patients | 53 |
| Hold-out patients | 10 |
| Slices admitted, total | 777 |
| Slices in the five folds | 638 |
| Slices in `holdout_test` | 139 |

The reduction from 1408 candidate triplets to 777 admitted slices is described in
the manuscript and in the supplementary material. No bias field correction was
applied and no registration was applied.

## Verifying a dataset before running

```powershell
$D = ".\data\dataset_5fold_final_v4"

# every split has the four modality directories
Get-ChildItem $D -Directory -Recurse -Depth 2 |
    Where-Object { $_.Name -in "T1","T2","FLAIR","T1CE" } |
    Group-Object { Split-Path $_.Parent -Leaf } | Select-Object Name, Count

# hold-out slice count, expected 139 per modality
"T1","T2","FLAIR","T1CE" | ForEach-Object {
    "{0}: {1}" -f $_, (Get-ChildItem "$D\holdout_test\$_" -Filter *.npy).Count
}

# the four modality directories must hold identical filenames
$ref = (Get-ChildItem "$D\holdout_test\T1" -Filter *.npy).Name
"T2","FLAIR","T1CE" | ForEach-Object {
    $cur = (Get-ChildItem "$D\holdout_test\$_" -Filter *.npy).Name
    "{0}: beda {1}" -f $_, (Compare-Object $ref $cur).Count
}
```

Expected: four directories per split, 139 files per hold-out modality, and zero
differences in the filename comparison.
