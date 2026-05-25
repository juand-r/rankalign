# Morning Results — ValROC — 2026-05-25T13:52:33Z

All cells: **ValROC × 100  ±  SE** (mean ± SE across the eval-task split for the corresponding (model × dataset)).

Train column: ✓ ep=N done · ⏳ jobid R elapsed (≤remaining) in-flight · – not started.
Empty cells (—): no eval CSV with that prefix yet.

## gemma-2-2b × membership

Source: [rosch_v7_2b_val_roc_table_cells.csv](metrics-from-scores/rosch_v7_2b_val_roc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 89.68 ± 2.58 | — | 89.68 ± 2.58 | — | 89.68 ± 2.58 |
| 1 SFT labelonly 10% | – | — | — | — | — | — |
| 2 RankAlign | – | — | — | — | — | — |
| 3 New + fsx [-TC] | – | — | — | — | — | — |
| 4 New + PMI + fsx | – | — | — | — | — | — |
| 5 RA + PMI + fsx [-NLL] | – | — | — | — | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | – | — | — | — | — | — |
| 12 New + NegTC [-fsx] | – | — | — | — | — | — |
| 13 SFT + CFT | ✓ ep=2 | 88.26 ± 3.13 | 88.26 ± 3.13 | — | 88.26 ± 3.13 | — |

## gemma-2-2b-it × membership

Source: [rosch_v7_2b-it_val_roc_table_cells.csv](metrics-from-scores/rosch_v7_2b-it_val_roc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 90.27 ± 2.44 | — | 90.27 ± 2.44 | — | 90.27 ± 2.44 |
| 1 SFT labelonly 10% | ✓ ep=2 | 78.06 ± 4.25 | 78.06 ± 4.25 | 78.06 ± 4.25 | 78.06 ± 4.25 | 78.06 ± 4.25 |
| 2 RankAlign | ✓ ep=2 | 88.51 ± 2.76 | 88.51 ± 2.76 | 88.51 ± 2.76 | 88.51 ± 2.76 | 88.51 ± 2.76 |
| 3 New + fsx [-TC] | ✓ ep=2 | 88.07 ± 2.93 | 88.07 ± 2.93 | 88.07 ± 2.93 | 88.07 ± 2.93 | 88.07 ± 2.93 |
| 4 New + PMI + fsx | ✓ ep=2 | 81.70 ± 4.01 | 81.70 ± 4.01 | 81.70 ± 4.01 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 90.81 ± 2.35 | 90.81 ± 2.35 | 90.81 ± 2.35 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 89.26 ± 2.39 | 89.26 ± 2.39 | 89.26 ± 2.39 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 90.17 ± 2.80 | — | — | 90.17 ± 2.80 | 90.17 ± 2.80 |
| 11 New + PMI [-fsx] | ✓ ep=2 | 89.35 ± 2.75 | 89.35 ± 2.75 | 89.35 ± 2.75 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 87.15 ± 2.62 | — | — | 87.15 ± 2.62 | 87.15 ± 2.62 |
| 13 SFT + CFT | ✓ ep=2 | 89.73 ± 3.05 | 89.73 ± 3.05 | 89.73 ± 3.05 | 89.73 ± 3.05 | 89.73 ± 3.05 |

## gemma-2-9b-it × membership

Source: [rosch_v7_9b-it_val_roc_table_cells.csv](metrics-from-scores/rosch_v7_9b-it_val_roc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 94.90 ± 1.93 | — | 94.90 ± 1.93 | — | 94.90 ± 1.93 |
| 1 SFT labelonly 10% | ✓ ep=1 | 94.99 ± 1.81 | 94.99 ± 1.81 | 94.99 ± 1.81 | 94.99 ± 1.81 | 94.99 ± 1.81 |
| 2 RankAlign | ✓ ep=2 | 95.06 ± 1.82 | 95.06 ± 1.82 | 95.06 ± 1.82 | 95.06 ± 1.82 | 95.06 ± 1.82 |
| 3 New + fsx [-TC] | ✓ ep=2 | 94.52 ± 1.91 | 94.52 ± 1.91 | 94.52 ± 1.91 | 94.52 ± 1.91 | 94.52 ± 1.91 |
| 4 New + PMI + fsx | ✓ ep=2 | 94.50 ± 1.85 | 94.50 ± 1.85 | 94.50 ± 1.85 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 95.12 ± 1.90 | 95.12 ± 1.90 | 95.12 ± 1.90 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 94.76 ± 2.01 | 94.76 ± 2.01 | 94.76 ± 2.01 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 94.63 ± 1.97 | — | — | 94.63 ± 1.97 | 94.63 ± 1.97 |
| 11 New + PMI [-fsx] | ✓ ep=2 | 94.76 ± 1.92 | 94.76 ± 1.92 | 94.76 ± 1.92 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 94.69 ± 1.91 | — | — | 94.69 ± 1.91 | 94.69 ± 1.91 |
| 13 SFT + CFT | ✓ ep=2 | 94.72 ± 1.85 | 94.72 ± 1.85 | — | 94.72 ± 1.85 | — |

## gemma-2-2b × persona

Source: [persona_v1_v7_gemma-2-2b_all_val_roc_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-2b_all_val_roc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 96.71 ± 2.90 | — | 96.71 ± 2.90 | — | 96.71 ± 2.90 |
| 1 SFT labelonly 10% | – | — | — | — | — | — |
| 2 RankAlign | ✓ ep=1 | — | — | — | — | — |
| 3 New + fsx [-TC] | – | — | — | — | — | — |
| 4 New + PMI + fsx | – | — | — | — | — | — |
| 5 RA + PMI + fsx [-NLL] | – | — | — | — | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=1 | 96.17 ± 2.86 | 96.17 ± 2.86 | 96.17 ± 2.86 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 90.29 ± 5.98 | — | — | 90.29 ± 5.98 | — |
| 13 SFT + CFT | ✓ ep=2 | 86.20 ± 7.66 | 86.09 ± 7.70 | 86.20 ± 7.66 | 86.09 ± 7.70 | 86.20 ± 7.66 |

## gemma-2-2b-it × persona

Source: [persona_v1_v7_gemma-2-2b-it_all_val_roc_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-2b-it_all_val_roc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 96.95 ± 1.35 | — | 96.95 ± 1.35 | — | 96.95 ± 1.35 |
| 1 SFT labelonly 10% | ✓ ep=2 | 91.73 ± 4.02 | 91.73 ± 4.02 | 91.73 ± 4.02 | 91.73 ± 4.02 | 91.73 ± 4.02 |
| 2 RankAlign | ✓ ep=2 | 92.07 ± 5.06 | 92.07 ± 5.06 | 92.07 ± 5.06 | 92.07 ± 5.06 | 92.07 ± 5.06 |
| 3 New + fsx [-TC] | ✓ ep=2 | 94.64 ± 3.09 | 94.64 ± 3.09 | 94.64 ± 3.09 | 94.64 ± 3.09 | 94.64 ± 3.09 |
| 4 New + PMI + fsx | ✓ ep=2 | 97.28 ± 1.85 | 97.28 ± 1.85 | 97.28 ± 1.85 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 97.88 ± 1.20 | 97.88 ± 1.20 | 97.88 ± 1.20 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 98.59 ± 0.86 | 98.59 ± 0.86 | 98.59 ± 0.86 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 91.92 ± 3.95 | — | — | 91.92 ± 3.95 | 91.92 ± 3.95 |
| 11 New + PMI [-fsx] | ✓ ep=2 | 95.72 ± 3.64 | 95.72 ± 3.64 | 94.81 ± 2.78 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 95.45 ± 3.57 | — | — | 95.45 ± 3.57 | 94.68 ± 2.62 |
| 13 SFT + CFT | ✓ ep=2 | 91.30 ± 4.64 | 90.95 ± 4.81 | 91.30 ± 4.64 | 90.95 ± 4.81 | 91.30 ± 4.64 |

## gemma-2-9b-it × persona

Source: [persona_v1_v7_gemma-2-9b-it_all_val_roc_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-9b-it_all_val_roc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 95.59 ± 1.49 | — | 95.59 ± 1.49 | — | 95.59 ± 1.49 |
| 1 SFT labelonly 10% | ⏳ 42307 R 7:12:45 (≤47:15) | 98.07 ± 1.85 | 98.07 ± 1.85 | 98.07 ± 1.85 | 98.07 ± 1.85 | 98.07 ± 1.85 |
| 2 RankAlign | ✓ ep=2 | 99.35 ± 0.43 | 99.35 ± 0.43 | 99.35 ± 0.43 | 99.35 ± 0.43 | 99.35 ± 0.43 |
| 3 New + fsx [-TC] | ⏳ 42377 R 4:05:22 (≤4:54:38) | 96.92 ± 2.97 | 96.92 ± 2.97 | 96.92 ± 2.97 | 96.92 ± 2.97 | 96.92 ± 2.97 |
| 4 New + PMI + fsx | ✓ ep=1 | 96.74 ± 3.21 | 96.74 ± 3.21 | 96.74 ± 3.21 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 99.41 ± 0.33 | 99.41 ± 0.33 | 99.41 ± 0.33 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 99.40 ± 0.36 | 99.40 ± 0.36 | 99.40 ± 0.36 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 97.76 ± 2.22 | — | — | 97.76 ± 2.22 | 97.76 ± 2.22 |
| 11 New + PMI [-fsx] | ✓ ep=1 | 98.03 ± 1.96 | 98.03 ± 1.96 | 97.66 ± 2.32 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 98.16 ± 1.84 | — | — | 98.16 ± 1.84 | 97.85 ± 2.10 |
| 13 SFT + CFT | ✓ ep=2 | 98.94 ± 1.00 | 98.94 ± 1.00 | 98.94 ± 1.00 | 98.94 ± 1.00 | 98.94 ± 1.00 |

## (pooled models) × ifeval (ID)

Source: [ifeval_id_val_roc_table_cells.csv](metrics-from-scores/ifeval_id_val_roc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 84.04 ± 1.59 | — | 84.04 ± 1.59 | — | 84.04 ± 1.59 |
| 1 SFT labelonly 10% | (pooled) | 82.78 ± 1.65 | 82.78 ± 1.65 | 82.78 ± 1.65 | — | — |
| 2 RankAlign | (pooled) | 81.53 ± 1.75 | — | 81.53 ± 1.75 | — | 81.53 ± 1.75 |
| 3 New + fsx [-TC] | (pooled) | 84.51 ± 1.48 | 84.51 ± 1.48 | 84.51 ± 1.48 | — | 84.51 ± 1.48 |
| 4 New + PMI + fsx | (pooled) | 82.53 ± 1.57 | 82.53 ± 1.57 | 82.53 ± 1.57 | — | — |
| 5 RA + PMI + fsx [-NLL] | (pooled) | 81.95 ± 1.60 | 81.95 ± 1.60 | 81.95 ± 1.60 | — | — |
| 6 RA + PMI [+TC] | (pooled) | — | — | — | — | — |
| 7 New + NegTC + fsx | (pooled) | 84.65 ± 1.54 | — | — | — | 84.65 ± 1.54 |
| 11 New + PMI [-fsx] | (pooled) | — | — | — | — | — |
| 12 New + NegTC [-fsx] | (pooled) | — | — | — | — | — |
| 13 SFT + CFT | (pooled) | — | — | — | — | — |

## (pooled models) × ifeval (OOD)

Source: [ifeval_ood_val_roc_table_cells.csv](metrics-from-scores/ifeval_ood_val_roc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 78.41 ± 3.47 | — | 78.41 ± 3.47 | — | 78.41 ± 3.47 |
| 1 SFT labelonly 10% | (pooled) | 77.25 ± 3.40 | 77.25 ± 3.40 | 77.25 ± 3.40 | — | — |
| 2 RankAlign | (pooled) | 74.50 ± 3.79 | — | 74.50 ± 3.79 | — | 74.50 ± 3.79 |
| 3 New + fsx [-TC] | (pooled) | 79.80 ± 3.42 | 79.80 ± 3.42 | 79.80 ± 3.42 | — | 79.80 ± 3.42 |
| 4 New + PMI + fsx | (pooled) | 79.58 ± 3.24 | 79.58 ± 3.24 | 79.58 ± 3.24 | — | — |
| 5 RA + PMI + fsx [-NLL] | (pooled) | 76.34 ± 3.55 | 76.34 ± 3.55 | 76.34 ± 3.55 | — | — |
| 6 RA + PMI [+TC] | (pooled) | — | — | — | — | — |
| 7 New + NegTC + fsx | (pooled) | 79.21 ± 3.07 | — | — | — | 79.21 ± 3.07 |
| 11 New + PMI [-fsx] | (pooled) | — | — | — | — | — |
| 12 New + NegTC [-fsx] | (pooled) | — | — | — | — | — |
| 13 SFT + CFT | (pooled) | — | — | — | — | — |

## gemma-4-31B-it × humaneval

Source: [humaneval_v2.1correct-upper_g4-31B-it_val_roc_table_cells.csv](metrics-from-scores/humaneval_v2.1correct-upper_g4-31B-it_val_roc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | — | — | — | — | — |
| 1 SFT labelonly 10% | – | — | — | — | — | — |
| 2 RankAlign | – | — | — | — | — | — |
| 3 New + fsx [-TC] | ✓ (prior run) | 93.73 ± 0.86 | 93.73 ± 0.86 | — | 93.73 ± 0.86 | — |
| 4 New + PMI + fsx | ✓ (prior run) | 93.34 ± 0.90 | 93.34 ± 0.90 | — | — | — |
| 5 RA + PMI + fsx [-NLL] | – | — | — | — | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | ✓ (prior run) | 93.22 ± 0.93 | — | — | 93.22 ± 0.93 | — |
| 11 New + PMI [-fsx] | – | — | — | — | — | — |
| 12 New + NegTC [-fsx] | – | — | — | — | — | — |
| 13 SFT + CFT | ⏳ 42114 R 13:44:22 (≤10:15:38) | — | — | — | — | — |
