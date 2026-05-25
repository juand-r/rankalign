# Morning Results — ValAcc — 2026-05-25T13:52:33Z

All cells: **ValAcc × 100  ±  SE** (mean ± SE across the eval-task split for the corresponding (model × dataset)).

Train column: ✓ ep=N done · ⏳ jobid R elapsed (≤remaining) in-flight · – not started.
Empty cells (—): no eval CSV with that prefix yet.

## gemma-2-2b × membership

Source: [rosch_v7_2b_val_acc_table_cells.csv](metrics-from-scores/rosch_v7_2b_val_acc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 78.78 ± 3.39 | — | 78.78 ± 3.39 | — | 78.78 ± 3.39 |
| 1 SFT labelonly 10% | – | — | — | — | — | — |
| 2 RankAlign | – | — | — | — | — | — |
| 3 New + fsx [-TC] | – | — | — | — | — | — |
| 4 New + PMI + fsx | – | — | — | — | — | — |
| 5 RA + PMI + fsx [-NLL] | – | — | — | — | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | – | — | — | — | — | — |
| 12 New + NegTC [-fsx] | – | — | — | — | — | — |
| 13 SFT + CFT | ✓ ep=2 | 78.77 ± 2.73 | 78.77 ± 2.73 | — | 78.77 ± 2.73 | — |

## gemma-2-2b-it × membership

Source: [rosch_v7_2b-it_val_acc_table_cells.csv](metrics-from-scores/rosch_v7_2b-it_val_acc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 79.57 ± 2.25 | — | 79.57 ± 2.25 | — | 79.57 ± 2.25 |
| 1 SFT labelonly 10% | ✓ ep=2 | 59.89 ± 4.25 | 59.89 ± 4.25 | 59.89 ± 4.25 | 59.89 ± 4.25 | 59.89 ± 4.25 |
| 2 RankAlign | ✓ ep=2 | 75.43 ± 3.15 | 75.43 ± 3.15 | 75.43 ± 3.15 | 75.43 ± 3.15 | 75.43 ± 3.15 |
| 3 New + fsx [-TC] | ✓ ep=2 | 77.44 ± 2.19 | 77.44 ± 2.19 | 77.44 ± 2.19 | 77.44 ± 2.19 | 77.44 ± 2.19 |
| 4 New + PMI + fsx | ✓ ep=2 | 72.62 ± 3.58 | 72.62 ± 3.58 | 72.62 ± 3.58 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 72.32 ± 3.45 | 72.32 ± 3.45 | 72.32 ± 3.45 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 82.84 ± 3.01 | 82.84 ± 3.01 | 82.84 ± 3.01 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 78.23 ± 2.87 | — | — | 78.23 ± 2.87 | 78.23 ± 2.87 |
| 11 New + PMI [-fsx] | ✓ ep=2 | 71.00 ± 2.29 | 71.00 ± 2.29 | 71.00 ± 2.29 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 77.89 ± 3.05 | — | — | 77.89 ± 3.05 | 77.89 ± 3.05 |
| 13 SFT + CFT | ✓ ep=2 | 80.73 ± 3.02 | 80.73 ± 3.02 | 80.73 ± 3.02 | 80.73 ± 3.02 | 80.73 ± 3.02 |

## gemma-2-9b-it × membership

Source: [rosch_v7_9b-it_val_acc_table_cells.csv](metrics-from-scores/rosch_v7_9b-it_val_acc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 87.11 ± 2.46 | — | 87.11 ± 2.46 | — | 87.11 ± 2.46 |
| 1 SFT labelonly 10% | ✓ ep=1 | 85.29 ± 2.46 | 85.29 ± 2.46 | 85.29 ± 2.46 | 85.29 ± 2.46 | 85.29 ± 2.46 |
| 2 RankAlign | ✓ ep=2 | 85.12 ± 2.67 | 85.12 ± 2.67 | 85.12 ± 2.67 | 85.12 ± 2.67 | 85.12 ± 2.67 |
| 3 New + fsx [-TC] | ✓ ep=2 | 82.94 ± 2.90 | 82.94 ± 2.90 | 82.94 ± 2.90 | 82.94 ± 2.90 | 82.94 ± 2.90 |
| 4 New + PMI + fsx | ✓ ep=2 | 84.56 ± 2.43 | 84.56 ± 2.43 | 84.56 ± 2.43 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 86.92 ± 2.39 | 86.92 ± 2.39 | 86.92 ± 2.39 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 86.56 ± 2.42 | 86.56 ± 2.42 | 86.56 ± 2.42 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 84.85 ± 2.74 | — | — | 84.85 ± 2.74 | 84.85 ± 2.74 |
| 11 New + PMI [-fsx] | ✓ ep=2 | 84.66 ± 3.03 | 84.66 ± 3.03 | 84.66 ± 3.03 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 84.61 ± 2.92 | — | — | 84.61 ± 2.92 | 84.61 ± 2.92 |
| 13 SFT + CFT | ✓ ep=2 | 80.08 ± 3.05 | 80.08 ± 3.05 | — | 80.08 ± 3.05 | — |

## gemma-2-2b × persona

Source: [persona_v1_v7_gemma-2-2b_all_val_acc_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-2b_all_val_acc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 93.63 ± 3.90 | — | 93.63 ± 3.90 | — | 93.63 ± 3.90 |
| 1 SFT labelonly 10% | – | — | — | — | — | — |
| 2 RankAlign | ✓ ep=1 | — | — | — | — | — |
| 3 New + fsx [-TC] | – | — | — | — | — | — |
| 4 New + PMI + fsx | – | — | — | — | — | — |
| 5 RA + PMI + fsx [-NLL] | – | — | — | — | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=1 | 88.78 ± 4.99 | 88.78 ± 4.99 | 88.78 ± 4.99 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 81.10 ± 8.37 | — | — | 81.10 ± 8.37 | — |
| 13 SFT + CFT | ✓ ep=2 | 79.48 ± 8.36 | 79.35 ± 8.40 | 79.48 ± 8.36 | 79.35 ± 8.40 | 79.48 ± 8.36 |

## gemma-2-2b-it × persona

Source: [persona_v1_v7_gemma-2-2b-it_all_val_acc_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-2b-it_all_val_acc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 88.67 ± 2.18 | — | 88.67 ± 2.18 | — | 88.67 ± 2.18 |
| 1 SFT labelonly 10% | ✓ ep=2 | 84.37 ± 6.97 | 84.37 ± 6.97 | 84.37 ± 6.97 | 84.37 ± 6.97 | 84.37 ± 6.97 |
| 2 RankAlign | ✓ ep=2 | 69.05 ± 3.27 | 69.05 ± 3.27 | 69.05 ± 3.27 | 69.05 ± 3.27 | 69.05 ± 3.27 |
| 3 New + fsx [-TC] | ✓ ep=2 | 89.88 ± 4.79 | 89.88 ± 4.79 | 89.88 ± 4.79 | 89.88 ± 4.79 | 89.88 ± 4.79 |
| 4 New + PMI + fsx | ✓ ep=2 | 90.13 ± 4.82 | 90.13 ± 4.82 | 90.13 ± 4.82 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 92.52 ± 3.17 | 92.52 ± 3.17 | 92.52 ± 3.17 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 92.97 ± 2.25 | 92.97 ± 2.25 | 92.97 ± 2.25 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 85.20 ± 6.64 | — | — | 85.20 ± 6.64 | 85.20 ± 6.64 |
| 11 New + PMI [-fsx] | ✓ ep=2 | 93.27 ± 4.97 | 93.27 ± 4.97 | 87.93 ± 5.34 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 87.08 ± 5.66 | — | — | 87.08 ± 5.66 | 88.78 ± 5.17 |
| 13 SFT + CFT | ✓ ep=2 | 81.07 ± 8.84 | 80.83 ± 8.97 | 81.07 ± 8.84 | 80.83 ± 8.97 | 81.07 ± 8.84 |

## gemma-2-9b-it × persona

Source: [persona_v1_v7_gemma-2-9b-it_all_val_acc_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-9b-it_all_val_acc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 86.47 ± 3.48 | — | 86.47 ± 3.48 | — | 86.47 ± 3.48 |
| 1 SFT labelonly 10% | ⏳ 42307 R 7:12:45 (≤47:15) | 89.67 ± 5.94 | 89.67 ± 5.94 | 89.67 ± 5.94 | 89.67 ± 5.94 | 89.67 ± 5.94 |
| 2 RankAlign | ✓ ep=2 | 97.03 ± 1.83 | 97.03 ± 1.83 | 97.03 ± 1.83 | 97.03 ± 1.83 | 97.03 ± 1.83 |
| 3 New + fsx [-TC] | ⏳ 42377 R 4:05:22 (≤4:54:38) | 91.78 ± 4.57 | 91.78 ± 4.57 | 91.78 ± 4.57 | 91.78 ± 4.57 | 91.78 ± 4.57 |
| 4 New + PMI + fsx | ✓ ep=1 | 93.30 ± 3.90 | 93.30 ± 3.90 | 93.30 ± 3.90 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 95.07 ± 1.87 | 95.07 ± 1.87 | 95.07 ± 1.87 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 96.15 ± 1.77 | 96.15 ± 1.77 | 96.15 ± 1.77 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 94.67 ± 3.74 | — | — | 94.67 ± 3.74 | 94.67 ± 3.74 |
| 11 New + PMI [-fsx] | ✓ ep=1 | 95.57 ± 2.89 | 95.57 ± 2.89 | 94.20 ± 3.29 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 95.42 ± 3.25 | — | — | 95.42 ± 3.25 | 92.87 ± 3.88 |
| 13 SFT + CFT | ✓ ep=2 | 93.53 ± 3.63 | 93.53 ± 3.63 | 93.53 ± 3.63 | 93.53 ± 3.63 | 93.53 ± 3.63 |

## (pooled models) × ifeval (ID)

Source: [ifeval_id_val_acc_table_cells.csv](metrics-from-scores/ifeval_id_val_acc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 69.02 ± 1.88 | — | 69.02 ± 1.88 | — | 69.02 ± 1.88 |
| 1 SFT labelonly 10% | (pooled) | 67.37 ± 1.86 | 67.37 ± 1.86 | 67.37 ± 1.86 | — | — |
| 2 RankAlign | (pooled) | 66.01 ± 1.89 | — | 66.01 ± 1.89 | — | 66.01 ± 1.89 |
| 3 New + fsx [-TC] | (pooled) | 68.61 ± 1.81 | 68.61 ± 1.81 | 68.61 ± 1.81 | — | 68.61 ± 1.81 |
| 4 New + PMI + fsx | (pooled) | 68.80 ± 1.83 | 68.80 ± 1.83 | 68.80 ± 1.83 | — | — |
| 5 RA + PMI + fsx [-NLL] | (pooled) | 68.89 ± 1.75 | 68.89 ± 1.75 | 68.89 ± 1.75 | — | — |
| 6 RA + PMI [+TC] | (pooled) | — | — | — | — | — |
| 7 New + NegTC + fsx | (pooled) | 69.56 ± 1.81 | — | — | — | 69.56 ± 1.81 |
| 11 New + PMI [-fsx] | (pooled) | — | — | — | — | — |
| 12 New + NegTC [-fsx] | (pooled) | — | — | — | — | — |
| 13 SFT + CFT | (pooled) | — | — | — | — | — |

## (pooled models) × ifeval (OOD)

Source: [ifeval_ood_val_acc_table_cells.csv](metrics-from-scores/ifeval_ood_val_acc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 59.69 ± 3.47 | — | 59.69 ± 3.47 | — | 59.69 ± 3.47 |
| 1 SFT labelonly 10% | (pooled) | 61.44 ± 3.63 | 61.44 ± 3.63 | 61.44 ± 3.63 | — | — |
| 2 RankAlign | (pooled) | 60.62 ± 3.62 | — | 60.62 ± 3.62 | — | 60.62 ± 3.62 |
| 3 New + fsx [-TC] | (pooled) | 62.69 ± 3.50 | 62.69 ± 3.50 | 62.69 ± 3.50 | — | 62.69 ± 3.50 |
| 4 New + PMI + fsx | (pooled) | 66.25 ± 3.81 | 66.25 ± 3.81 | 66.25 ± 3.81 | — | — |
| 5 RA + PMI + fsx [-NLL] | (pooled) | 60.81 ± 3.57 | 60.81 ± 3.57 | 60.81 ± 3.57 | — | — |
| 6 RA + PMI [+TC] | (pooled) | — | — | — | — | — |
| 7 New + NegTC + fsx | (pooled) | 63.38 ± 3.36 | — | — | — | 63.38 ± 3.36 |
| 11 New + PMI [-fsx] | (pooled) | — | — | — | — | — |
| 12 New + NegTC [-fsx] | (pooled) | — | — | — | — | — |
| 13 SFT + CFT | (pooled) | — | — | — | — | — |

## gemma-4-31B-it × humaneval

Source: [humaneval_v2.1correct-upper_g4-31B-it_val_acc_table_cells.csv](metrics-from-scores/humaneval_v2.1correct-upper_g4-31B-it_val_acc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | — | — | — | — | — |
| 1 SFT labelonly 10% | – | — | — | — | — | — |
| 2 RankAlign | – | — | — | — | — | — |
| 3 New + fsx [-TC] | ✓ (prior run) | 85.67 ± 1.33 | 85.67 ± 1.33 | — | 85.67 ± 1.33 | — |
| 4 New + PMI + fsx | ✓ (prior run) | 85.84 ± 1.43 | 85.84 ± 1.43 | — | — | — |
| 5 RA + PMI + fsx [-NLL] | – | — | — | — | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | ✓ (prior run) | 83.67 ± 1.34 | — | — | 83.67 ± 1.34 | — |
| 11 New + PMI [-fsx] | – | — | — | — | — | — |
| 12 New + NegTC [-fsx] | – | — | — | — | — | — |
| 13 SFT + CFT | ⏳ 42114 R 13:44:22 (≤10:15:38) | — | — | — | — | — |
