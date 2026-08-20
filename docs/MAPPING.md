# Identifier mapping

Generated from each experiment's `config_snapshot.json`.

| Identifier | Name in the manuscript | Folder | Model class | Loss | Patience | AMP | lr | Replicates |
|---|---|---|---|---|---|---|---|---|
| EXP-100 | Baseline_U-Net | `EXP-100_UNET_L1_fold0` | UNetBaseline | MaskedL1 | 25 | True | 0.0001 | 5 |
| EXP-202 | Baseline_ResAttn | `EXP-202_ATTENTION_UNET_L1_fold0` | AttentionUNet | MaskedL1 | 25 | True | 0.0001 | 5 |
| EXP-204 | Baseline_Dense | `EXP-204_DENSE_UNET_L1_fold0` | DenseUNet | MaskedL1 | 25 | True | 0.0001 | 5 |
| EXP-303 | Fusion_Trimodal | `EXP-303_T1_T2_FLAIR_TO_T1CE_fold0` | ResAttentionUNet | MaskedL1 | 25 | True | 0.0001 | 5 |
| EXP-402 | FeatureFusion_MultiEnc (MultiEnc) | `EXP-402_MULTI_ENCODER_STEP1_T1_T2_F_fold0` | DMECNetStep1 | MaskedL1 | 25 | True | 0.0001 | 5 |
| EXP-500 | GuidedAttn | `EXP-500_GUIDED_RESATTENTION_T1T2F_fold0` | GuidedResAttentionUNet | MaskedL1 | 25 | True | 0.0001 | 5 |
| EXP-601 | EDM w/o SC | `EXP-601_EDM_fold0` | EDMSynth | EDMLoss | 25 | True | 0.0001 | 5 |
| EXP-602 | EDM | `EXP-602_EDM_fold0` | EDMSynth | EDMLoss | 25 | True | 0.0001 | 5 |
| EXP-703 | EDM-AP | `EXP-703_EDM_GAN_VGG_LPIPS_fold0` | EDMSynth | EDMLoss | 40 | False | 0.0001 | 5 |
| EXP-706 | excluded, single replicate, Table S10 | `EXP-706_single_replicate_fold0` | EDMSynth | EDMLoss | 25 | True | 0.0001 |  |
| EXP-713 | MSCA | `EXP-713_MULTISCALE_CROSS_ATTENTION_fold0` | EDMSynth | EDMLoss | 25 | True | 0.0001 | 5 |
| EXP-716A | XAttn-Decomp, excluded | `EXP-716A_CROSS_ATTENTION_DECOMPOSITION_fold0` | EDMSynth | EDMLoss | 25 | True | 0.0001 |  |
| EXP-803 | SharpFreq LSGAN, excluded | `EXP-803_SharpFreq_LSGAN_Iter1_fold0` | EDMSynth | EDMLoss | 25 | True | 0.0001 |  |
| EXP-806 | GroupNorm cosine LR, excluded | `EXP-806_GroupNorm_CosineLR_Iter4_fold0` | EDMSynth | EDMLoss | 40 | True | 0.0001 |  |
| EXP-813 | StratEnh | `EXP-813_StratifiedEnh_Sharpness_fold0` | EDMSynth | EDMLoss | 45 | True | 0.0001 | 5 |
| EXP-815 | Enh. seg. head, excluded | `EXP-815_EnhSegHead_EagerInit_FastTrain_fold0` | EDMSynth | EDMLoss | 45 | True | 0.0001 |  |
| EXP-816 | CalGate | `EXP-816_CalibratedGate_BalancedSeg_fold0` | EDMSynth | EDMLoss | 45 | True | 0.0001 | 5 |
| EXP-PGAN01 | pGAN | `EXP-PGAN01_pGAN_baseline` | pGAN | GAN + VGG | 40 | False | 0.0002 | 5 |
| EXP-RESVIT | ResViT | `EXP-RESVIT_Dalmaz2022` | ResViT | LSGAN + L1 | none | n/a | 0.0002 | 5 |
| EXP-DDRESUNET | DDResUNet | `EXP-DDRESUNET_OsmanTamam2023` | DDResUNet | L1 | see config | see config | 0.0001 | 5 |

This repository contains the configurations evaluated on the locked hold-out
cohort. Trained weights and predicted images are not included. Paths inside
`config_snapshot.json` are the original machine paths and are left unedited
as a historical record.
