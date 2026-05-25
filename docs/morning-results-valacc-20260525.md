# Morning Results — ValAcc — 2026-05-25T15:41:08Z

All cells: **ValAcc × 100 ± SE** (mean ± SE across the eval-task split for that section).

Section header convention: `<model> × <eval-set label>`. The model is
the (LoRA-finetuned base) generator under test; the eval-set label says
which held-out task slice the cells were averaged over. For example,
`gemma-2-9b-it × membership (eval = rosch, all 6 tasks)` means: gemma-2-9b-it
trained on `membership-sans-rosch-v0` and evaluated on the 6 held-out Rosch
cross-categorization tasks. `persona ID` / `persona OOD` are the 3+3 splits
of `persona-v1` (see headers for the per-persona task names).

Train column: ✓ ep=N done · ⏳ jobid R elapsed (≤remaining) in-flight · – not started.
Empty cells (—): no eval CSV with that prefix yet.

## gemma-2-2b × membership (eval = rosch, all 6 tasks)

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

## gemma-2-2b-it × membership (eval = rosch, all 6 tasks)

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

## gemma-2-9b-it × membership (eval = rosch, all 6 tasks)

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

## gemma-2-2b × persona (all 6 personas)

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

## gemma-2-2b-it × persona (all 6 personas)

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

## gemma-2-9b-it × persona (all 6 personas)

Source: [persona_v1_v7_gemma-2-9b-it_all_val_acc_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-9b-it_all_val_acc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 86.47 ± 3.48 | — | 86.47 ± 3.48 | — | 86.47 ± 3.48 |
| 1 SFT labelonly 10% | ✓ ep=1 | 89.67 ± 5.94 | 89.67 ± 5.94 | 89.67 ± 5.94 | 89.67 ± 5.94 | 89.67 ± 5.94 |
| 2 RankAlign | ✓ ep=2 | 97.03 ± 1.83 | 97.03 ± 1.83 | 97.03 ± 1.83 | 97.03 ± 1.83 | 97.03 ± 1.83 |
| 3 New + fsx [-TC] | ⏳ 42377 R 5:53:57 (≤3:06:03) | 91.78 ± 4.57 | 91.78 ± 4.57 | 91.78 ± 4.57 | 91.78 ± 4.57 | 91.78 ± 4.57 |
| 4 New + PMI + fsx | ✓ ep=1 | 93.30 ± 3.90 | 93.30 ± 3.90 | 93.30 ± 3.90 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 95.07 ± 1.87 | 95.07 ± 1.87 | 95.07 ± 1.87 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 96.15 ± 1.77 | 96.15 ± 1.77 | 96.15 ± 1.77 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 94.67 ± 3.74 | — | — | 94.67 ± 3.74 | 94.67 ± 3.74 |
| 11 New + PMI [-fsx] | ✓ ep=1 | 95.57 ± 2.89 | 95.57 ± 2.89 | 94.20 ± 3.29 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 95.42 ± 3.25 | — | — | 95.42 ± 3.25 | 92.87 ± 3.88 |
| 13 SFT + CFT | ✓ ep=2 | 93.93 ± 3.44 | 93.53 ± 3.63 | 93.93 ± 3.44 | 93.53 ± 3.63 | 93.93 ± 3.44 |

## gemma-2-2b × persona ID (3 in-domain: psychopathy, machiavellianism, narcissism)

Source: [persona_v1_v7_gemma-2-2b_id_val_acc_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-2b_id_val_acc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 89.93 ± 7.49 | — | 89.93 ± 7.49 | — | 89.93 ± 7.49 |
| 1 SFT labelonly 10% | – | — | — | — | — | — |
| 2 RankAlign | ✓ ep=1 | — | — | — | — | — |
| 3 New + fsx [-TC] | – | — | — | — | — | — |
| 4 New + PMI + fsx | – | — | — | — | — | — |
| 5 RA + PMI + fsx [-NLL] | – | — | — | — | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=1 | 99.67 ± 0.07 | 99.67 ± 0.07 | 99.67 ± 0.07 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 99.67 ± 0.24 | — | — | 99.67 ± 0.24 | — |
| 13 SFT + CFT | ✓ ep=2 | 97.60 ± 0.70 | 97.53 ± 0.68 | 97.60 ± 0.70 | 97.53 ± 0.68 | 97.60 ± 0.70 |

## gemma-2-2b-it × persona ID (3 in-domain: psychopathy, machiavellianism, narcissism)

Source: [persona_v1_v7_gemma-2-2b-it_id_val_acc_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-2b-it_id_val_acc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 86.80 ± 2.73 | — | 86.80 ± 2.73 | — | 86.80 ± 2.73 |
| 1 SFT labelonly 10% | ✓ ep=2 | 99.60 ± 0.12 | 99.60 ± 0.12 | 99.60 ± 0.12 | 99.60 ± 0.12 | 99.60 ± 0.12 |
| 2 RankAlign | ✓ ep=2 | 75.27 ± 3.26 | 75.27 ± 3.26 | 75.27 ± 3.26 | 75.27 ± 3.26 | 75.27 ± 3.26 |
| 3 New + fsx [-TC] | ✓ ep=2 | 99.67 ± 0.18 | 99.67 ± 0.18 | 99.67 ± 0.18 | 99.67 ± 0.18 | 99.67 ± 0.18 |
| 4 New + PMI + fsx | ✓ ep=2 | 99.73 ± 0.18 | 99.73 ± 0.18 | 99.73 ± 0.18 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 93.73 ± 0.47 | 93.73 ± 0.47 | 93.73 ± 0.47 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 88.80 ± 1.93 | 88.80 ± 1.93 | 88.80 ± 1.93 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 99.67 ± 0.07 | — | — | 99.67 ± 0.07 | 99.67 ± 0.07 |
| 11 New + PMI [-fsx] | ✓ ep=2 | 99.60 ± 0.31 | 99.60 ± 0.31 | 99.33 ± 0.18 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 99.67 ± 0.24 | — | — | 99.67 ± 0.24 | 98.67 ± 0.35 |
| 13 SFT + CFT | ✓ ep=2 | 99.67 ± 0.13 | 99.73 ± 0.07 | 99.67 ± 0.13 | 99.73 ± 0.07 | 99.67 ± 0.13 |

## gemma-2-9b-it × persona ID (3 in-domain: psychopathy, machiavellianism, narcissism)

Source: [persona_v1_v7_gemma-2-9b-it_id_val_acc_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-9b-it_id_val_acc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 80.93 ± 3.14 | — | 80.93 ± 3.14 | — | 80.93 ± 3.14 |
| 1 SFT labelonly 10% | ✓ ep=1 | 99.93 ± 0.07 | 99.93 ± 0.07 | 99.93 ± 0.07 | 99.93 ± 0.07 | 99.93 ± 0.07 |
| 2 RankAlign | ✓ ep=2 | 97.53 ± 2.27 | 97.53 ± 2.27 | 97.53 ± 2.27 | 97.53 ± 2.27 | 97.53 ± 2.27 |
| 3 New + fsx [-TC] | ⏳ 42377 R 5:53:57 (≤3:06:03) | 99.80 ± 0.20 | 99.80 ± 0.20 | 99.80 ± 0.20 | 99.80 ± 0.20 | 99.80 ± 0.20 |
| 4 New + PMI + fsx | ✓ ep=1 | 99.93 ± 0.07 | 99.93 ± 0.07 | 99.93 ± 0.07 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 97.20 ± 1.40 | 97.20 ± 1.40 | 97.20 ± 1.40 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 97.40 ± 1.90 | 97.40 ± 1.90 | 97.40 ± 1.90 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 99.93 ± 0.07 | — | — | 99.93 ± 0.07 | 99.93 ± 0.07 |
| 11 New + PMI [-fsx] | ✓ ep=1 | 99.87 ± 0.07 | 99.87 ± 0.07 | 99.93 ± 0.07 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 99.93 ± 0.07 | — | — | 99.93 ± 0.07 | 99.93 ± 0.07 |
| 13 SFT + CFT | ✓ ep=2 | 99.87 ± 0.07 | 99.87 ± 0.07 | 99.87 ± 0.07 | 99.87 ± 0.07 | 99.87 ± 0.07 |

## gemma-2-2b × persona OOD (3 held-out: desire-to-create-allies, interest-in-music, interest-in-science)

Source: [persona_v1_v7_gemma-2-2b_ood_val_acc_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-2b_ood_val_acc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 97.33 ± 2.47 | — | 97.33 ± 2.47 | — | 97.33 ± 2.47 |
| 1 SFT labelonly 10% | – | — | — | — | — | — |
| 2 RankAlign | ✓ ep=1 | — | — | — | — | — |
| 3 New + fsx [-TC] | – | — | — | — | — | — |
| 4 New + PMI + fsx | – | — | — | — | — | — |
| 5 RA + PMI + fsx [-NLL] | – | — | — | — | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=1 | 77.90 ± 2.46 | 77.90 ± 2.46 | 77.90 ± 2.46 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 62.53 ± 2.43 | — | — | 62.53 ± 2.43 | — |
| 13 SFT + CFT | ✓ ep=2 | 61.37 ± 4.55 | 61.17 ± 4.68 | 61.37 ± 4.55 | 61.17 ± 4.68 | 61.37 ± 4.55 |

## gemma-2-2b-it × persona OOD (3 held-out: desire-to-create-allies, interest-in-music, interest-in-science)

Source: [persona_v1_v7_gemma-2-2b-it_ood_val_acc_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-2b-it_ood_val_acc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 90.53 ± 3.57 | — | 90.53 ± 3.57 | — | 90.53 ± 3.57 |
| 1 SFT labelonly 10% | ✓ ep=2 | 69.13 ± 3.24 | 69.13 ± 3.24 | 69.13 ± 3.24 | 69.13 ± 3.24 | 69.13 ± 3.24 |
| 2 RankAlign | ✓ ep=2 | 62.83 ± 2.05 | 62.83 ± 2.05 | 62.83 ± 2.05 | 62.83 ± 2.05 | 62.83 ± 2.05 |
| 3 New + fsx [-TC] | ✓ ep=2 | 80.10 ± 4.35 | 80.10 ± 4.35 | 80.10 ± 4.35 | 80.10 ± 4.35 | 80.10 ± 4.35 |
| 4 New + PMI + fsx | ✓ ep=2 | 80.53 ± 4.92 | 80.53 ± 4.92 | 80.53 ± 4.92 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 91.30 ± 6.97 | 91.30 ± 6.97 | 91.30 ± 6.97 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 97.13 ± 2.07 | 97.13 ± 2.07 | 97.13 ± 2.07 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 70.73 ± 3.34 | — | — | 70.73 ± 3.34 | 70.73 ± 3.34 |
| 11 New + PMI [-fsx] | ✓ ep=2 | 86.93 ± 9.12 | 86.93 ± 9.12 | 76.53 ± 3.58 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 74.50 ± 1.31 | — | — | 74.50 ± 1.31 | 78.90 ± 5.97 |
| 13 SFT + CFT | ✓ ep=2 | 62.47 ± 6.66 | 61.93 ± 6.68 | 62.47 ± 6.66 | 61.93 ± 6.68 | 62.47 ± 6.66 |

## gemma-2-9b-it × persona OOD (3 held-out: desire-to-create-allies, interest-in-music, interest-in-science)

Source: [persona_v1_v7_gemma-2-9b-it_ood_val_acc_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-9b-it_ood_val_acc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 92.00 ± 4.48 | — | 92.00 ± 4.48 | — | 92.00 ± 4.48 |
| 1 SFT labelonly 10% | ✓ ep=1 | 79.40 ± 8.42 | 93.20 ± 6.05 | 79.40 ± 8.42 | 93.20 ± 6.05 | 79.40 ± 8.42 |
| 2 RankAlign | ✓ ep=2 | 96.53 ± 3.37 | 96.53 ± 3.37 | 96.53 ± 3.37 | 96.53 ± 3.37 | 96.53 ± 3.37 |
| 3 New + fsx [-TC] | ⏳ 42377 R 5:53:57 (≤3:06:03) | 83.77 ± 6.35 | 83.77 ± 6.35 | 83.77 ± 6.35 | 83.77 ± 6.35 | 83.77 ± 6.35 |
| 4 New + PMI + fsx | ✓ ep=1 | 86.67 ± 5.66 | 86.67 ± 5.66 | 86.67 ± 5.66 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 92.93 ± 3.32 | 92.93 ± 3.32 | 92.93 ± 3.32 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 94.90 ± 3.25 | 94.90 ± 3.25 | 94.90 ± 3.25 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 89.40 ± 6.49 | — | — | 89.40 ± 6.49 | 89.40 ± 6.49 |
| 11 New + PMI [-fsx] | ✓ ep=1 | 91.27 ± 4.83 | 91.27 ± 4.83 | 88.47 ± 4.62 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 90.90 ± 5.68 | — | — | 90.90 ± 5.68 | 85.80 ± 5.01 |
| 13 SFT + CFT | ✓ ep=2 | 88.00 ± 4.90 | 87.20 ± 5.06 | 88.00 ± 4.90 | 87.20 ± 5.06 | 88.00 ± 4.90 |

## gemma-2-9b-it × ifeval ID (held-out 50% of completions, prompts seen at train)

Source: [ifeval_id_val_acc_table_cells.csv](metrics-from-scores/ifeval_id_val_acc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 69.02 ± 1.88 | — | 69.02 ± 1.88 | — | 69.02 ± 1.88 |
| 1 SFT labelonly 10% | ✓ (prior run) | 67.37 ± 1.86 | 67.37 ± 1.86 | 67.37 ± 1.86 | — | — |
| 2 RankAlign | ✓ (prior run) | 66.01 ± 1.89 | — | 66.01 ± 1.89 | — | 66.01 ± 1.89 |
| 3 New + fsx [-TC] | ✓ (prior run) | 68.61 ± 1.81 | 68.61 ± 1.81 | 68.61 ± 1.81 | — | 68.61 ± 1.81 |
| 4 New + PMI + fsx | ✓ (prior run) | 68.80 ± 1.83 | 68.80 ± 1.83 | 68.80 ± 1.83 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ (prior run) | 68.89 ± 1.75 | 68.89 ± 1.75 | 68.89 ± 1.75 | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | ✓ (prior run) | 69.56 ± 1.81 | — | — | — | 69.56 ± 1.81 |
| 11 New + PMI [-fsx] | – | — | — | — | — | — |
| 12 New + NegTC [-fsx] | – | — | — | — | — | — |
| 13 SFT + CFT | ⏳ 42343 R 7:39:01 (≤6:20:59) | — | — | — | — | — |

## gemma-2-9b-it × ifeval OOD (20 fully held-out prompts: prompt_1..13, 15..21)

Source: [ifeval_ood_val_acc_table_cells.csv](metrics-from-scores/ifeval_ood_val_acc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 59.69 ± 3.47 | — | 59.69 ± 3.47 | — | 59.69 ± 3.47 |
| 1 SFT labelonly 10% | ✓ (prior run) | 61.44 ± 3.63 | 61.44 ± 3.63 | 61.44 ± 3.63 | — | — |
| 2 RankAlign | ✓ (prior run) | 60.62 ± 3.62 | — | 60.62 ± 3.62 | — | 60.62 ± 3.62 |
| 3 New + fsx [-TC] | ✓ (prior run) | 62.69 ± 3.50 | 62.69 ± 3.50 | 62.69 ± 3.50 | — | 62.69 ± 3.50 |
| 4 New + PMI + fsx | ✓ (prior run) | 66.25 ± 3.81 | 66.25 ± 3.81 | 66.25 ± 3.81 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ (prior run) | 60.81 ± 3.57 | 60.81 ± 3.57 | 60.81 ± 3.57 | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | ✓ (prior run) | 63.38 ± 3.36 | — | — | — | 63.38 ± 3.36 |
| 11 New + PMI [-fsx] | – | — | — | — | — | — |
| 12 New + NegTC [-fsx] | – | — | — | — | — | — |
| 13 SFT + CFT | ⏳ 42343 R 7:39:01 (≤6:20:59) | — | — | — | — | — |

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
| 13 SFT + CFT | ⏳ 42114 R 15:32:57 (≤8:27:03) | — | — | — | — | — |
