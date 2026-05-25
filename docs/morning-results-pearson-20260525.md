# Morning Results — Pearson(gen, val) — 2026-05-25T13:52:33Z

All cells: **Pearson(gen, val) × 100  ±  SE** (mean ± SE across the eval-task split for the corresponding (model × dataset)).

Train column: ✓ ep=N done · ⏳ jobid R elapsed (≤remaining) in-flight · – not started.
Empty cells (—): no eval CSV with that prefix yet.

## gemma-2-2b × membership

Source: [rosch_v7_2b_pearson_table_cells.csv](metrics-from-scores/rosch_v7_2b_pearson_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 46.44 ± 3.91 | — | 54.48 ± 3.19 | — | 40.69 ± 4.83 |
| 1 SFT labelonly 10% | – | — | — | — | — | — |
| 2 RankAlign | – | — | — | — | — | — |
| 3 New + fsx [-TC] | – | — | — | — | — | — |
| 4 New + PMI + fsx | – | — | — | — | — | — |
| 5 RA + PMI + fsx [-NLL] | – | — | — | — | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | – | — | — | — | — | — |
| 12 New + NegTC [-fsx] | – | — | — | — | — | — |
| 13 SFT + CFT | ✓ ep=2 | 53.55 ± 3.59 | 60.42 ± 2.45 | — | 50.44 ± 4.29 | — |

## gemma-2-2b-it × membership

Source: [rosch_v7_2b-it_pearson_table_cells.csv](metrics-from-scores/rosch_v7_2b-it_pearson_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 23.67 ± 4.68 | — | 44.05 ± 4.03 | — | 48.59 ± 4.52 |
| 1 SFT labelonly 10% | ✓ ep=2 | 48.84 ± 3.10 | 43.57 ± 2.77 | 49.49 ± 2.95 | 40.14 ± 3.68 | 40.79 ± 5.93 |
| 2 RankAlign | ✓ ep=2 | 54.86 ± 4.22 | 54.31 ± 3.24 | 73.68 ± 2.11 | 41.17 ± 3.35 | 66.76 ± 3.03 |
| 3 New + fsx [-TC] | ✓ ep=2 | 58.05 ± 5.56 | 58.71 ± 2.02 | 73.75 ± 2.31 | 46.32 ± 3.38 | 63.11 ± 2.75 |
| 4 New + PMI + fsx | ✓ ep=2 | 47.73 ± 5.56 | 58.29 ± 3.56 | 70.62 ± 2.75 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 46.99 ± 3.95 | 67.67 ± 1.82 | 71.16 ± 2.04 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 50.53 ± 4.70 | 68.00 ± 2.36 | 75.36 ± 2.19 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 41.62 ± 4.83 | — | — | 58.04 ± 3.00 | 64.79 ± 3.28 |
| 11 New + PMI [-fsx] | ✓ ep=2 | 50.19 ± 5.33 | 65.89 ± 3.00 | 74.97 ± 2.17 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 48.48 ± 5.13 | — | — | 60.26 ± 2.27 | 69.78 ± 2.27 |
| 13 SFT + CFT | ✓ ep=2 | 56.54 ± 3.82 | 57.16 ± 2.60 | 65.06 ± 2.94 | 49.57 ± 2.40 | 64.03 ± 2.65 |

## gemma-2-9b-it × membership

Source: [rosch_v7_9b-it_pearson_table_cells.csv](metrics-from-scores/rosch_v7_9b-it_pearson_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 31.52 ± 2.57 | — | 50.02 ± 2.66 | — | 48.03 ± 5.39 |
| 1 SFT labelonly 10% | ✓ ep=1 | 56.42 ± 3.76 | 57.62 ± 2.79 | 64.03 ± 2.83 | 50.74 ± 2.88 | 52.77 ± 5.43 |
| 2 RankAlign | ✓ ep=2 | 68.88 ± 3.18 | 68.43 ± 2.47 | 75.16 ± 2.66 | 55.43 ± 3.22 | 62.61 ± 5.52 |
| 3 New + fsx [-TC] | ✓ ep=2 | 67.39 ± 3.57 | 66.05 ± 2.77 | 73.59 ± 2.94 | 54.02 ± 3.45 | 62.43 ± 4.71 |
| 4 New + PMI + fsx | ✓ ep=2 | 68.29 ± 3.19 | 73.76 ± 2.34 | 76.18 ± 2.36 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 62.07 ± 3.13 | 74.42 ± 2.66 | 75.17 ± 2.66 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 61.72 ± 3.20 | 75.25 ± 2.71 | 76.54 ± 2.74 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 65.06 ± 3.15 | — | — | 72.24 ± 3.08 | 61.73 ± 4.66 |
| 11 New + PMI [-fsx] | ✓ ep=2 | 67.86 ± 3.13 | 73.68 ± 2.42 | 76.75 ± 2.52 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 66.50 ± 2.71 | — | — | 70.97 ± 2.98 | 63.04 ± 5.26 |
| 13 SFT + CFT | ✓ ep=2 | 48.94 ± 4.45 | 50.84 ± 3.42 | — | 44.92 ± 3.56 | — |

## gemma-2-2b × persona

Source: [persona_v1_v7_gemma-2-2b_all_pearson_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-2b_all_pearson_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -8.10 ± 9.47 | — | -23.53 ± 6.43 | — | 47.42 ± 8.85 |
| 1 SFT labelonly 10% | – | — | — | — | — | — |
| 2 RankAlign | ✓ ep=1 | — | — | — | — | — |
| 3 New + fsx [-TC] | – | — | — | — | — | — |
| 4 New + PMI + fsx | – | — | — | — | — | — |
| 5 RA + PMI + fsx [-NLL] | – | — | — | — | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=1 | 26.13 ± 15.91 | 37.76 ± 15.70 | 37.41 ± 12.70 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 30.36 ± 12.57 | — | — | 45.03 ± 12.88 | — |
| 13 SFT + CFT | ✓ ep=2 | 26.57 ± 6.58 | 18.83 ± 11.73 | 12.22 ± 8.60 | 27.23 ± 10.50 | 24.42 ± 4.73 |

## gemma-2-2b-it × persona

Source: [persona_v1_v7_gemma-2-2b-it_all_pearson_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-2b-it_all_pearson_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 5.46 ± 7.94 | — | 15.22 ± 5.53 | — | 50.80 ± 5.30 |
| 1 SFT labelonly 10% | ✓ ep=2 | 13.32 ± 10.21 | 16.90 ± 13.38 | 7.13 ± 8.58 | 30.21 ± 10.95 | 28.85 ± 5.64 |
| 2 RankAlign | ✓ ep=2 | 38.15 ± 12.10 | 53.65 ± 7.16 | 29.30 ± 4.27 | 57.01 ± 6.38 | 40.29 ± 5.33 |
| 3 New + fsx [-TC] | ✓ ep=2 | 37.97 ± 15.86 | 44.50 ± 15.34 | 32.49 ± 12.83 | 53.40 ± 12.23 | 31.41 ± 6.79 |
| 4 New + PMI + fsx | ✓ ep=2 | 23.93 ± 15.39 | 29.93 ± 18.14 | 20.95 ± 10.71 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 26.21 ± 18.64 | 41.30 ± 16.04 | 20.48 ± 12.55 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 37.80 ± 16.23 | 60.95 ± 9.63 | 41.99 ± 6.71 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 28.48 ± 13.74 | — | — | 42.44 ± 13.45 | 27.50 ± 3.46 |
| 11 New + PMI [-fsx] | ✓ ep=2 | 24.56 ± 16.20 | 30.69 ± 19.16 | 16.76 ± 11.65 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 28.58 ± 14.29 | — | — | 48.16 ± 11.81 | 13.34 ± 8.20 |
| 13 SFT + CFT | ✓ ep=2 | 17.17 ± 9.65 | 18.76 ± 13.32 | 10.03 ± 7.30 | 31.63 ± 11.28 | 9.79 ± 9.83 |

## gemma-2-9b-it × persona

Source: [persona_v1_v7_gemma-2-9b-it_all_pearson_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-9b-it_all_pearson_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -7.98 ± 9.99 | — | -11.10 ± 8.07 | — | 18.46 ± 2.19 |
| 1 SFT labelonly 10% | ⏳ 42307 R 7:12:45 (≤47:15) | 14.04 ± 13.37 | 23.30 ± 16.11 | 14.68 ± 14.57 | 39.40 ± 10.37 | 49.65 ± 3.46 |
| 2 RankAlign | ✓ ep=2 | 60.78 ± 12.01 | 76.01 ± 5.32 | 65.09 ± 7.21 | 79.05 ± 4.21 | 77.62 ± 3.75 |
| 3 New + fsx [-TC] | ⏳ 42377 R 4:05:22 (≤4:54:38) | 30.68 ± 17.32 | 41.98 ± 17.98 | 33.88 ± 19.04 | 55.93 ± 11.44 | 66.78 ± 5.31 |
| 4 New + PMI + fsx | ✓ ep=1 | 21.13 ± 15.49 | 33.08 ± 17.38 | 24.07 ± 17.46 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 30.00 ± 17.18 | 64.48 ± 10.89 | 44.72 ± 15.05 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 32.42 ± 17.62 | 62.66 ± 11.43 | 34.28 ± 17.02 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 22.58 ± 15.30 | — | — | 47.08 ± 12.45 | 57.07 ± 4.09 |
| 11 New + PMI [-fsx] | ✓ ep=1 | 23.49 ± 16.23 | 39.01 ± 16.91 | 27.71 ± 16.58 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 22.38 ± 15.68 | — | — | 46.53 ± 12.67 | 55.39 ± 4.42 |
| 13 SFT + CFT | ✓ ep=2 | 6.63 ± 12.87 | 13.28 ± 14.38 | 3.91 ± 11.99 | 30.64 ± 9.52 | 35.89 ± 4.19 |

## (pooled models) × ifeval (ID)

Source: [ifeval_id_pearson_table_cells.csv](metrics-from-scores/ifeval_id_pearson_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 18.06 ± 3.52 | — | 46.16 ± 3.00 | — | 25.32 ± 3.96 |
| 1 SFT labelonly 10% | (pooled) | 12.77 ± 3.67 | 19.57 ± 3.74 | 35.57 ± 3.37 | — | — |
| 2 RankAlign | (pooled) | 22.90 ± 3.59 | — | 41.97 ± 3.79 | — | 25.02 ± 4.27 |
| 3 New + fsx [-TC] | (pooled) | 60.58 ± 3.37 | 69.75 ± 2.91 | 72.03 ± 2.63 | — | 59.02 ± 3.61 |
| 4 New + PMI + fsx | (pooled) | 47.71 ± 3.35 | 68.16 ± 2.97 | 71.50 ± 2.70 | — | — |
| 5 RA + PMI + fsx [-NLL] | (pooled) | 33.91 ± 3.28 | 66.80 ± 3.30 | 70.34 ± 3.15 | — | — |
| 6 RA + PMI [+TC] | (pooled) | — | — | — | — | — |
| 7 New + NegTC + fsx | (pooled) | 57.32 ± 3.15 | — | — | — | 19.17 ± 5.12 |
| 11 New + PMI [-fsx] | (pooled) | — | — | — | — | — |
| 12 New + NegTC [-fsx] | (pooled) | — | — | — | — | — |
| 13 SFT + CFT | (pooled) | — | — | — | — | — |

## (pooled models) × ifeval (OOD)

Source: [ifeval_ood_pearson_table_cells.csv](metrics-from-scores/ifeval_ood_pearson_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 9.25 ± 7.54 | — | 35.54 ± 7.47 | — | 25.06 ± 5.76 |
| 1 SFT labelonly 10% | (pooled) | 7.33 ± 7.11 | 13.18 ± 6.97 | 29.53 ± 6.27 | — | — |
| 2 RankAlign | (pooled) | 2.03 ± 8.15 | — | 29.87 ± 7.63 | — | 14.47 ± 6.19 |
| 3 New + fsx [-TC] | (pooled) | 29.80 ± 7.16 | 57.18 ± 5.16 | 56.86 ± 6.48 | — | 54.65 ± 3.93 |
| 4 New + PMI + fsx | (pooled) | 30.29 ± 6.88 | 62.84 ± 4.44 | 61.67 ± 5.21 | — | — |
| 5 RA + PMI + fsx [-NLL] | (pooled) | 15.57 ± 7.42 | 47.62 ± 7.18 | 51.97 ± 6.53 | — | — |
| 6 RA + PMI [+TC] | (pooled) | — | — | — | — | — |
| 7 New + NegTC + fsx | (pooled) | 34.36 ± 6.42 | — | — | — | 14.93 ± 8.92 |
| 11 New + PMI [-fsx] | (pooled) | — | — | — | — | — |
| 12 New + NegTC [-fsx] | (pooled) | — | — | — | — | — |
| 13 SFT + CFT | (pooled) | — | — | — | — | — |

## gemma-4-31B-it × humaneval

Source: [humaneval_v2.1correct-upper_g4-31B-it_pearson_table_cells.csv](metrics-from-scores/humaneval_v2.1correct-upper_g4-31B-it_pearson_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | — | — | — | — | — |
| 1 SFT labelonly 10% | – | — | — | — | — | — |
| 2 RankAlign | – | — | — | — | — | — |
| 3 New + fsx [-TC] | ✓ (prior run) | 65.35 ± 1.72 | 45.50 ± 3.07 | — | 56.43 ± 2.64 | — |
| 4 New + PMI + fsx | ✓ (prior run) | 61.59 ± 1.69 | 66.76 ± 1.75 | — | — | — |
| 5 RA + PMI + fsx [-NLL] | – | — | — | — | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | ✓ (prior run) | 58.53 ± 1.66 | — | — | 76.26 ± 1.31 | — |
| 11 New + PMI [-fsx] | – | — | — | — | — | — |
| 12 New + NegTC [-fsx] | – | — | — | — | — | — |
| 13 SFT + CFT | ⏳ 42114 R 13:44:22 (≤10:15:38) | — | — | — | — | — |
