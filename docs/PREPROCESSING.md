# Preprocessing chain

The imaging data are not distributed. This document states the order in which
the scripts under `preprocessing/` were run, so that every preprocessing and
masking decision behind a reported number can be traced.

## Stages

| Stage | Directory | What it produces |
|---|---|---|
| 1 | `01_dicom_to_resampled/` | DICOM audit, reorientation, resampling to a common grid |
| 2 | `02_registration_sensitivity/` | rigid registration and the paired before-and-after tests reported as the registration sensitivity analysis |
| 3 | `03_patient_split/` | the patient-level partition |
| 4 | `04_uniform_normalise_2p5d/` | uniform spacing, standardised volume size, masked normalisation, 2.5D slice construction |
| 5 | `05_folds_and_pseudonymisation/` | the five folds and the locked hold-out, the identifier map, the pseudonymisation, the leakage check |

## Order

```
01  1_dicom_audit                     -> dicom_audit.csv
    1a_analisis_distribusi_geometri
    1b_cek_orientation_per_patient
    2_reorientation_resampling_volume -> work/temp_resampled
    2a_cek_shape_spacing
    2b_resampling_grid_T1             -> work/temp_resampled_fixed

02  3_evaluasi_3D_aligment
    4_3D_rigid_registration           -> work/temp_registered
    5_uji_statistik_per_case          -> paired tests, registered against unregistered
    6_distribusi_delta_dan_outlier_analysis
    98_hitung_MC_NCC

03  9_split_patient                   -> patient-level partition

04  24a_audit_temp_resampled_fixed
    24b_check_spacing_global
    25_resample_global_spacing
    26_verify_uniform_spacing
    27_check_uniform_size
    28_standardize_volume_size_v2     -> work/temp_uniform_final_v2
    29_verify_final_volume_size
    30_intensity_audit_final_uniform
    31_normalize_with_mask_uniform    -> work/temp_uniform_normalized_v2
    32_verify_normalization_uniform
    33_verify_background_zero
    34_build_2p5D_from_uniform
    35_audit_2p5D_base

05  44_build_dataset_5fold_final      -> the five folds and the hold-out
    45_audit_final_dataset
    46_generate_patient_mapping       -> the identifier map, not distributed
    47_apply_anonymization            -> filenames replaced by the identifier
    48_check_phi_leakage
    49_rename_test_to_val
    50_rename_sacred_to_holdout
```

A final cleaning step follows outside `preprocessing/`. The scripts
`tools/clean_dataset_final_v4.py`, `tools/clean_corrupted_slices_v4.py` and
`tools/clean_small_brain_slices.py` remove slices whose variance falls below
`1e-6` and slices carrying a reconstruction artefact. This is the step that
reduces the admitted slice count to the number reported in the manuscript.

## Two decisions that the manuscript depends on

**The brain region is an intensity threshold, not a skull-stripping mask.**
`31_normalize_with_mask_uniform.py` defines it as `arr > 0` on the reference
volume of the modality concerned. The z-score is computed inside that region,
after clipping to the 0.5th and 99.5th percentiles of the in-region intensities,
and the background is left at zero.

**No registration was applied to the distributed dataset.** Stage 2 produces
`work/temp_registered`, but stage 4 starts again from
`work/temp_resampled_fixed`. The registered volumes are used only for the
sensitivity analysis reported in the manuscript, and never for training or
evaluation.

## What is not distributed

The identifier map produced by `46_generate_patient_mapping.py`, the patient
split table produced by `9_split_patient.py`, and every imaging volume. Those
files relate an identifier to a patient or contain patient anatomy. Scripts that
consume them are included, because they are method. Their outputs are not.

## Manual step

`work/dataset_5fold_final_v2` was copied to the dataset release directory and
renamed before the final cleaning step. That copy was performed by hand and has
no script.
