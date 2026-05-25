# Morning Results — Spearman(gen, val) — 2026-05-25T13:52:33Z

All cells: **Spearman(gen, val) × 100  ±  SE** (mean ± SE across the eval-task split for the corresponding (model × dataset)).

Train column: ✓ ep=N done · ⏳ jobid R elapsed (≤remaining) in-flight · – not started.
Empty cells (—): no eval CSV with that prefix yet.

## gemma-2-2b × membership

Source: [rosch_v7_2b_spearman_table_cells.csv](metrics-from-scores/rosch_v7_2b_spearman_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 47.85 ± 3.97 | — | 55.09 ± 3.21 | — | 43.02 ± 5.42 |
| 1 SFT labelonly 10% | – | — | — | — | — | — |
| 2 RankAlign | – | — | — | — | — | — |
| 3 New + fsx [-TC] | – | — | — | — | — | — |
| 4 New + PMI + fsx | – | — | — | — | — | — |
| 5 RA + PMI + fsx [-NLL] | – | — | — | — | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | – | — | — | — | — | — |
| 12 New + NegTC [-fsx] | – | — | — | — | — | — |
| 13 SFT + CFT | ✓ ep=2 | 59.93 ± 3.94 | 63.62 ± 2.11 | — | 52.51 ± 4.23 | — |

## gemma-2-2b-it × membership

Source: [rosch_v7_2b-it_spearman_table_cells.csv](metrics-from-scores/rosch_v7_2b-it_spearman_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 31.77 ± 4.91 | — | 47.39 ± 4.50 | — | 50.13 ± 4.41 |
| 1 SFT labelonly 10% | ✓ ep=2 | 48.94 ± 3.85 | 42.44 ± 4.03 | 47.61 ± 5.16 | 39.37 ± 5.09 | 37.84 ± 6.67 |
| 2 RankAlign | ✓ ep=2 | 55.02 ± 4.44 | 52.70 ± 3.24 | 74.13 ± 2.35 | 41.29 ± 3.53 | 66.38 ± 3.20 |
| 3 New + fsx [-TC] | ✓ ep=2 | 61.63 ± 5.50 | 57.30 ± 1.95 | 74.08 ± 2.44 | 47.31 ± 3.34 | 62.24 ± 2.97 |
| 4 New + PMI + fsx | ✓ ep=2 | 53.62 ± 5.04 | 58.23 ± 3.38 | 70.47 ± 2.39 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 49.54 ± 4.25 | 67.23 ± 1.77 | 70.31 ± 2.70 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 53.67 ± 4.42 | 67.90 ± 1.51 | 75.54 ± 2.33 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 44.39 ± 3.72 | — | — | 59.93 ± 2.54 | 66.77 ± 3.84 |
| 11 New + PMI [-fsx] | ✓ ep=2 | 56.95 ± 4.93 | 68.83 ± 2.00 | 77.85 ± 1.69 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 54.20 ± 4.18 | — | — | 59.58 ± 2.94 | 69.39 ± 2.89 |
| 13 SFT + CFT | ✓ ep=2 | 60.21 ± 3.80 | 57.03 ± 2.41 | 64.23 ± 2.43 | 51.59 ± 2.69 | 63.39 ± 2.57 |

## gemma-2-9b-it × membership

Source: [rosch_v7_9b-it_spearman_table_cells.csv](metrics-from-scores/rosch_v7_9b-it_spearman_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 38.68 ± 1.82 | — | 55.33 ± 2.74 | — | 52.59 ± 6.19 |
| 1 SFT labelonly 10% | ✓ ep=1 | 63.77 ± 3.16 | 62.23 ± 2.39 | 68.75 ± 2.17 | 54.61 ± 2.66 | 56.13 ± 5.19 |
| 2 RankAlign | ✓ ep=2 | 70.93 ± 2.72 | 68.79 ± 2.39 | 76.91 ± 2.10 | 57.54 ± 3.15 | 66.13 ± 5.00 |
| 3 New + fsx [-TC] | ✓ ep=2 | 69.57 ± 3.03 | 67.08 ± 2.38 | 75.21 ± 2.23 | 56.23 ± 2.93 | 66.07 ± 4.81 |
| 4 New + PMI + fsx | ✓ ep=2 | 70.95 ± 2.98 | 74.30 ± 2.26 | 77.10 ± 2.13 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 66.10 ± 2.22 | 76.77 ± 1.98 | 77.82 ± 1.96 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 64.35 ± 2.51 | 75.92 ± 2.13 | 77.79 ± 2.08 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 68.88 ± 2.80 | — | — | 75.92 ± 3.02 | 66.16 ± 4.64 |
| 11 New + PMI [-fsx] | ✓ ep=2 | 72.56 ± 2.65 | 75.51 ± 1.99 | 79.85 ± 2.08 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 70.27 ± 2.22 | — | — | 74.04 ± 2.51 | 65.89 ± 5.23 |
| 13 SFT + CFT | ✓ ep=2 | 57.48 ± 3.99 | 57.75 ± 2.73 | — | 51.38 ± 3.07 | — |

## gemma-2-2b × persona

Source: [persona_v1_v7_gemma-2-2b_all_spearman_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-2b_all_spearman_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -6.32 ± 9.12 | — | -23.41 ± 7.34 | — | 46.42 ± 9.28 |
| 1 SFT labelonly 10% | – | — | — | — | — | — |
| 2 RankAlign | ✓ ep=1 | — | — | — | — | — |
| 3 New + fsx [-TC] | – | — | — | — | — | — |
| 4 New + PMI + fsx | – | — | — | — | — | — |
| 5 RA + PMI + fsx [-NLL] | – | — | — | — | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=1 | 28.38 ± 17.00 | 37.12 ± 15.86 | 36.75 ± 13.45 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 31.58 ± 13.46 | — | — | 44.12 ± 11.99 | — |
| 13 SFT + CFT | ✓ ep=2 | 30.52 ± 8.63 | 20.97 ± 12.61 | 15.11 ± 10.28 | 29.58 ± 11.59 | 25.86 ± 5.61 |

## gemma-2-2b-it × persona

Source: [persona_v1_v7_gemma-2-2b-it_all_spearman_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-2b-it_all_spearman_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 6.48 ± 7.71 | — | 14.75 ± 5.42 | — | 50.75 ± 4.75 |
| 1 SFT labelonly 10% | ✓ ep=2 | 13.13 ± 10.86 | 16.14 ± 12.92 | 7.57 ± 7.51 | 30.01 ± 10.23 | 29.64 ± 5.49 |
| 2 RankAlign | ✓ ep=2 | 47.65 ± 13.54 | 61.34 ± 7.71 | 24.23 ± 5.85 | 64.72 ± 6.89 | 50.12 ± 6.51 |
| 3 New + fsx [-TC] | ✓ ep=2 | 37.46 ± 16.04 | 43.90 ± 15.15 | 33.03 ± 13.45 | 53.68 ± 11.73 | 33.91 ± 6.60 |
| 4 New + PMI + fsx | ✓ ep=2 | 25.46 ± 16.77 | 29.24 ± 17.91 | 22.86 ± 11.33 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 27.89 ± 19.84 | 41.30 ± 16.64 | 19.70 ± 11.13 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 41.07 ± 15.93 | 59.99 ± 9.27 | 41.76 ± 6.97 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 28.97 ± 14.50 | — | — | 42.12 ± 12.93 | 29.23 ± 4.23 |
| 11 New + PMI [-fsx] | ✓ ep=2 | 25.59 ± 16.26 | 30.91 ± 17.93 | 16.93 ± 12.58 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 29.63 ± 14.69 | — | — | 49.87 ± 11.53 | 16.13 ± 7.80 |
| 13 SFT + CFT | ✓ ep=2 | 20.01 ± 11.18 | 18.54 ± 13.36 | 11.94 ± 7.65 | 31.13 ± 11.34 | 12.69 ± 10.28 |

## gemma-2-9b-it × persona

Source: [persona_v1_v7_gemma-2-9b-it_all_spearman_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-9b-it_all_spearman_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -7.20 ± 9.36 | — | -8.95 ± 7.18 | — | 19.28 ± 1.21 |
| 1 SFT labelonly 10% | ⏳ 42307 R 7:12:45 (≤47:15) | 21.23 ± 15.18 | 25.90 ± 15.71 | 20.14 ± 14.06 | 41.61 ± 9.77 | 53.20 ± 3.16 |
| 2 RankAlign | ✓ ep=2 | 58.53 ± 10.80 | 72.25 ± 3.27 | 65.42 ± 6.83 | 74.89 ± 2.13 | 79.26 ± 2.27 |
| 3 New + fsx [-TC] | ⏳ 42377 R 4:05:22 (≤4:54:38) | 38.49 ± 17.13 | 42.44 ± 16.61 | 37.93 ± 17.93 | 54.84 ± 10.35 | 68.08 ± 4.55 |
| 4 New + PMI + fsx | ✓ ep=1 | 27.74 ± 16.24 | 33.76 ± 16.27 | 26.69 ± 16.80 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 33.16 ± 17.25 | 61.22 ± 9.22 | 45.82 ± 13.72 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 36.48 ± 17.51 | 59.57 ± 9.71 | 35.84 ± 15.95 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 27.83 ± 16.18 | — | — | 46.36 ± 11.37 | 56.18 ± 3.84 |
| 11 New + PMI [-fsx] | ✓ ep=1 | 31.93 ± 16.13 | 38.92 ± 15.57 | 29.66 ± 15.84 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 30.20 ± 15.84 | — | — | 47.91 ± 11.35 | 56.59 ± 3.65 |
| 13 SFT + CFT | ✓ ep=2 | 12.98 ± 14.68 | 15.87 ± 14.71 | 8.51 ± 12.19 | 31.89 ± 9.50 | 37.66 ± 4.42 |

## (pooled models) × ifeval (ID)

Source: [ifeval_id_spearman_table_cells.csv](metrics-from-scores/ifeval_id_spearman_table_cells.csv)

(no data)

## (pooled models) × ifeval (OOD)

Source: [ifeval_ood_spearman_table_cells.csv](metrics-from-scores/ifeval_ood_spearman_table_cells.csv)

(no data)

## gemma-4-31B-it × humaneval

Source: [humaneval_v2.1correct-upper_g4-31B-it_spearman_table_cells.csv](metrics-from-scores/humaneval_v2.1correct-upper_g4-31B-it_spearman_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | — | — | — | — | — |
| 1 SFT labelonly 10% | – | — | — | — | — | — |
| 2 RankAlign | – | — | — | — | — | — |
| 3 New + fsx [-TC] | ✓ (prior run) | 68.87 ± 1.84 | 51.29 ± 2.59 | — | 51.86 ± 2.78 | — |
| 4 New + PMI + fsx | ✓ (prior run) | 67.02 ± 1.78 | 67.85 ± 1.66 | — | — | — |
| 5 RA + PMI + fsx [-NLL] | – | — | — | — | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | ✓ (prior run) | 63.96 ± 1.89 | — | — | 77.05 ± 1.13 | — |
| 11 New + PMI [-fsx] | – | — | — | — | — | — |
| 12 New + NegTC [-fsx] | – | — | — | — | — | — |
| 13 SFT + CFT | ⏳ 42114 R 13:44:22 (≤10:15:38) | — | — | — | — | — |
