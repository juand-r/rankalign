# Morning Results — GenROC — 2026-05-25T13:52:33Z

All cells: **GenROC × 100  ±  SE** (mean ± SE across the eval-task split for the corresponding (model × dataset)).

Train column: ✓ ep=N done · ⏳ jobid R elapsed (≤remaining) in-flight · – not started.
Empty cells (—): no eval CSV with that prefix yet.

## gemma-2-2b × membership

Source: [rosch_v7_2b_gen_roc_table_cells.csv](metrics-from-scores/rosch_v7_2b_gen_roc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 74.61 ± 1.59 | — | 77.32 ± 1.76 | — | 77.71 ± 4.01 |
| 1 SFT labelonly 10% | – | — | — | — | — | — |
| 2 RankAlign | – | — | — | — | — | — |
| 3 New + fsx [-TC] | – | — | — | — | — | — |
| 4 New + PMI + fsx | – | — | — | — | — | — |
| 5 RA + PMI + fsx [-NLL] | – | — | — | — | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | – | — | — | — | — | — |
| 12 New + NegTC [-fsx] | – | — | — | — | — | — |
| 13 SFT + CFT | ✓ ep=2 | 80.88 ± 2.08 | 82.52 ± 1.98 | — | 81.05 ± 2.29 | — |

## gemma-2-2b-it × membership

Source: [rosch_v7_2b-it_gen_roc_table_cells.csv](metrics-from-scores/rosch_v7_2b-it_gen_roc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 64.67 ± 2.90 | — | 72.31 ± 3.67 | — | 79.83 ± 3.43 |
| 1 SFT labelonly 10% | ✓ ep=2 | 84.32 ± 1.51 | 83.26 ± 1.64 | 87.24 ± 1.54 | 80.55 ± 1.08 | 87.25 ± 2.55 |
| 2 RankAlign | ✓ ep=2 | 79.21 ± 2.80 | 78.17 ± 3.22 | 82.72 ± 2.56 | 74.85 ± 2.83 | 86.93 ± 2.67 |
| 3 New + fsx [-TC] | ✓ ep=2 | 81.18 ± 2.89 | 81.79 ± 2.39 | 88.16 ± 2.40 | 78.98 ± 1.62 | 86.41 ± 3.34 |
| 4 New + PMI + fsx | ✓ ep=2 | 72.87 ± 3.84 | 82.09 ± 2.68 | 86.08 ± 2.78 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 73.44 ± 2.90 | 82.34 ± 2.41 | 84.01 ± 2.47 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 76.66 ± 3.23 | 83.95 ± 2.59 | 86.16 ± 2.40 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 73.85 ± 3.26 | — | — | 81.69 ± 2.26 | 85.69 ± 3.04 |
| 11 New + PMI [-fsx] | ✓ ep=2 | 77.10 ± 3.68 | 84.97 ± 2.63 | 87.97 ± 2.77 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 77.93 ± 3.38 | — | — | 83.55 ± 1.74 | 90.02 ± 1.93 |
| 13 SFT + CFT | ✓ ep=2 | 84.95 ± 1.66 | 85.45 ± 1.29 | 88.07 ± 1.33 | 83.69 ± 1.00 | 87.10 ± 2.12 |

## gemma-2-9b-it × membership

Source: [rosch_v7_9b-it_gen_roc_table_cells.csv](metrics-from-scores/rosch_v7_9b-it_gen_roc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 66.91 ± 2.55 | — | 77.73 ± 2.18 | — | 81.20 ± 3.97 |
| 1 SFT labelonly 10% | ✓ ep=1 | 84.76 ± 1.60 | 85.37 ± 1.37 | 88.09 ± 1.29 | 83.48 ± 1.10 | 83.07 ± 3.23 |
| 2 RankAlign | ✓ ep=2 | 88.37 ± 1.46 | 89.11 ± 1.25 | 91.68 ± 1.33 | 84.76 ± 1.91 | 88.99 ± 2.56 |
| 3 New + fsx [-TC] | ✓ ep=2 | 88.78 ± 1.45 | 88.50 ± 1.11 | 90.81 ± 1.47 | 84.27 ± 1.43 | 88.44 ± 2.57 |
| 4 New + PMI + fsx | ✓ ep=2 | 88.16 ± 1.77 | 91.99 ± 1.12 | 92.50 ± 1.22 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 82.65 ± 2.15 | 90.68 ± 1.73 | 91.07 ± 1.69 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 82.60 ± 2.35 | 91.32 ± 1.78 | 91.64 ± 1.89 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 85.26 ± 2.01 | — | — | 92.32 ± 1.56 | 89.00 ± 2.45 |
| 11 New + PMI [-fsx] | ✓ ep=2 | 88.94 ± 1.57 | 92.39 ± 1.31 | 93.07 ± 1.46 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 86.70 ± 1.89 | — | — | 92.58 ± 1.48 | 88.64 ± 2.70 |
| 13 SFT + CFT | ✓ ep=2 | 81.94 ± 1.98 | 83.17 ± 1.44 | — | 81.88 ± 1.17 | — |

## gemma-2-2b × persona

Source: [persona_v1_v7_gemma-2-2b_all_gen_roc_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-2b_all_gen_roc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 40.01 ± 5.71 | — | 37.83 ± 3.89 | — | 73.89 ± 7.52 |
| 1 SFT labelonly 10% | – | — | — | — | — | — |
| 2 RankAlign | ✓ ep=1 | — | — | — | — | — |
| 3 New + fsx [-TC] | – | — | — | — | — | — |
| 4 New + PMI + fsx | – | — | — | — | — | — |
| 5 RA + PMI + fsx [-NLL] | – | — | — | — | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=1 | 62.21 ± 11.43 | 70.44 ± 10.37 | 68.27 ± 9.59 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 58.96 ± 11.71 | — | — | 72.11 ± 9.43 | — |
| 13 SFT + CFT | ✓ ep=2 | 51.64 ± 8.85 | 57.33 ± 8.03 | 51.78 ± 7.09 | 64.84 ± 6.20 | 66.60 ± 1.68 |

## gemma-2-2b-it × persona

Source: [persona_v1_v7_gemma-2-2b-it_all_gen_roc_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-2b-it_all_gen_roc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 48.56 ± 5.17 | — | 57.40 ± 3.01 | — | 75.82 ± 3.28 |
| 1 SFT labelonly 10% | ✓ ep=2 | 54.84 ± 8.84 | 59.41 ± 8.76 | 56.49 ± 5.14 | 68.84 ± 6.62 | 72.53 ± 4.80 |
| 2 RankAlign | ✓ ep=2 | 78.30 ± 10.26 | 86.43 ± 6.93 | 59.47 ± 4.69 | 89.75 ± 5.65 | 83.93 ± 2.61 |
| 3 New + fsx [-TC] | ✓ ep=2 | 66.87 ± 11.84 | 73.16 ± 10.87 | 68.76 ± 8.97 | 80.34 ± 8.10 | 72.60 ± 4.32 |
| 4 New + PMI + fsx | ✓ ep=2 | 60.51 ± 10.85 | 66.28 ± 11.29 | 62.31 ± 6.66 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 66.40 ± 13.27 | 72.57 ± 11.93 | 59.20 ± 8.60 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 74.02 ± 10.68 | 85.65 ± 6.82 | 74.05 ± 4.37 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 58.97 ± 11.77 | — | — | 70.62 ± 10.32 | 69.59 ± 2.59 |
| 11 New + PMI [-fsx] | ✓ ep=2 | 60.42 ± 11.93 | 65.31 ± 12.79 | 57.54 ± 8.54 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 61.05 ± 11.36 | — | — | 77.65 ± 8.15 | 61.53 ± 6.05 |
| 13 SFT + CFT | ✓ ep=2 | 53.61 ± 8.72 | 58.08 ± 9.21 | 56.21 ± 4.69 | 67.95 ± 6.99 | 64.40 ± 8.07 |

## gemma-2-9b-it × persona

Source: [persona_v1_v7_gemma-2-9b-it_all_gen_roc_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-9b-it_all_gen_roc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 42.96 ± 5.31 | — | 40.88 ± 5.40 | — | 57.19 ± 2.20 |
| 1 SFT labelonly 10% | ⏳ 42307 R 7:12:45 (≤47:15) | 57.17 ± 9.57 | 64.29 ± 9.49 | 60.86 ± 8.22 | 74.06 ± 6.17 | 83.73 ± 2.41 |
| 2 RankAlign | ✓ ep=2 | 85.19 ± 7.54 | 95.43 ± 2.55 | 90.06 ± 4.81 | 96.90 ± 1.78 | 97.24 ± 1.55 |
| 3 New + fsx [-TC] | ⏳ 42377 R 4:05:22 (≤4:54:38) | 63.95 ± 12.22 | 72.47 ± 11.22 | 68.94 ± 11.77 | 81.11 ± 7.48 | 91.52 ± 3.27 |
| 4 New + PMI + fsx | ✓ ep=1 | 60.92 ± 10.71 | 68.71 ± 10.76 | 65.11 ± 10.52 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 67.84 ± 11.06 | 85.91 ± 6.69 | 76.24 ± 9.53 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 69.08 ± 11.58 | 83.72 ± 7.72 | 69.10 ± 11.60 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 62.09 ± 10.51 | — | — | 76.39 ± 8.46 | 85.10 ± 2.30 |
| 11 New + PMI [-fsx] | ✓ ep=1 | 62.60 ± 10.87 | 71.83 ± 10.52 | 67.94 ± 9.69 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 62.04 ± 10.60 | — | — | 76.64 ± 8.17 | 85.44 ± 2.21 |
| 13 SFT + CFT | ✓ ep=2 | 52.36 ± 8.74 | 57.53 ± 8.85 | 53.91 ± 7.14 | 67.35 ± 5.94 | 72.95 ± 3.01 |

## (pooled models) × ifeval (ID)

Source: [ifeval_id_table_cells.csv](metrics-from-scores/ifeval_id_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 58.72 ± 2.13 | — | 81.64 ± 1.43 | — | 65.86 ± 2.11 |
| 1 SFT labelonly 10% | (pooled) | 56.98 ± 2.06 | 62.99 ± 2.00 | 75.39 ± 1.75 | — | — |
| 2 RankAlign | (pooled) | 59.54 ± 2.06 | — | 74.56 ± 1.83 | — | 65.21 ± 2.06 |
| 3 New + fsx [-TC] | (pooled) | 71.97 ± 2.22 | 78.29 ± 2.16 | 83.67 ± 1.78 | — | 71.01 ± 2.09 |
| 4 New + PMI + fsx | (pooled) | 69.46 ± 2.15 | 76.86 ± 2.07 | 82.98 ± 1.88 | — | — |
| 5 RA + PMI + fsx [-NLL] | (pooled) | 62.39 ± 2.13 | 79.36 ± 1.81 | 83.71 ± 1.52 | — | — |
| 6 RA + PMI [+TC] | (pooled) | — | — | — | — | — |
| 7 New + NegTC + fsx | (pooled) | 72.25 ± 2.08 | — | — | — | 56.59 ± 2.33 |
| 11 New + PMI [-fsx] | (pooled) | — | — | — | — | — |
| 12 New + NegTC [-fsx] | (pooled) | — | — | — | — | — |
| 13 SFT + CFT | (pooled) | — | — | — | — | — |

## (pooled models) × ifeval (OOD)

Source: [ifeval_ood_table_cells.csv](metrics-from-scores/ifeval_ood_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 52.81 ± 4.39 | — | 78.30 ± 2.22 | — | 59.27 ± 2.66 |
| 1 SFT labelonly 10% | (pooled) | 51.18 ± 4.25 | 57.64 ± 4.03 | 74.53 ± 2.60 | — | — |
| 2 RankAlign | (pooled) | 51.87 ± 4.28 | — | 74.88 ± 3.58 | — | 59.42 ± 2.68 |
| 3 New + fsx [-TC] | (pooled) | 60.16 ± 4.19 | 82.02 ± 2.41 | 83.53 ± 2.47 | — | 67.27 ± 3.59 |
| 4 New + PMI + fsx | (pooled) | 59.65 ± 4.39 | 82.03 ± 2.58 | 85.06 ± 2.71 | — | — |
| 5 RA + PMI + fsx [-NLL] | (pooled) | 54.01 ± 4.44 | 78.40 ± 3.24 | 83.39 ± 2.60 | — | — |
| 6 RA + PMI [+TC] | (pooled) | — | — | — | — | — |
| 7 New + NegTC + fsx | (pooled) | 60.77 ± 4.26 | — | — | — | 56.96 ± 3.79 |
| 11 New + PMI [-fsx] | (pooled) | — | — | — | — | — |
| 12 New + NegTC [-fsx] | (pooled) | — | — | — | — | — |
| 13 SFT + CFT | (pooled) | — | — | — | — | — |

## gemma-4-31B-it × humaneval

Source: [humaneval_v2.1correct-upper_g4-31B-it_gen_roc_table_cells.csv](metrics-from-scores/humaneval_v2.1correct-upper_g4-31B-it_gen_roc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | — | — | — | — | — |
| 1 SFT labelonly 10% | – | — | — | — | — | — |
| 2 RankAlign | – | — | — | — | — | — |
| 3 New + fsx [-TC] | ✓ (prior run) | 91.11 ± 0.96 | 86.09 ± 1.18 | — | 84.96 ± 1.67 | — |
| 4 New + PMI + fsx | ✓ (prior run) | 88.30 ± 0.99 | 92.16 ± 0.77 | — | — | — |
| 5 RA + PMI + fsx [-NLL] | – | — | — | — | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | ✓ (prior run) | 83.14 ± 1.09 | — | — | 94.16 ± 0.91 | — |
| 11 New + PMI [-fsx] | – | — | — | — | — | — |
| 12 New + NegTC [-fsx] | – | — | — | — | — | — |
| 13 SFT + CFT | ⏳ 42114 R 13:44:22 (≤10:15:38) | — | — | — | — | — |
