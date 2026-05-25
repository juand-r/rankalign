# Morning Results — GenROC — 2026-05-25T15:58:15Z

All cells: **GenROC × 100 ± SE** (mean ± SE across the eval-task split for that section).

Section header convention: `<model> × <eval-set label>`. The model is
the (LoRA-finetuned base) generator under test; the eval-set label says
which held-out task slice the cells were averaged over. For example,
`gemma-2-9b-it × membership (eval = rosch, all 10 tasks)` means: gemma-2-9b-it
trained on `membership-sans-rosch-v0` and evaluated on the 10 held-out Rosch
cross-categorization tasks (bird, carpenters-tool, clothing, fruit, furniture,
sport, toy, vegetable, vehicle, weapon). `persona ID` / `persona OOD` are the
3+3 splits of `persona-v1` (see headers for the per-persona task names).

Train column: ✓ ep=N done · ⏳ jobid R elapsed (≤remaining) in-flight · – not started.
Empty cells (—): no eval CSV with that prefix yet.

> **⚠️ IMPORTANT PROVENANCE CAVEAT — IFEVAL SECTIONS ARE V6, NOT V7.**
>
> All other sections (rosch, persona, humaneval) source from **v7 (fix1)** score
> files. The two ifeval sections below are built by the legacy
> `_build_ifeval_ood_table.py`, which hardcodes a v6 model prefix
> (`v6-google_gemma-2-9b-it-delta0.15-epoch2_ifeval-concat-all`). Their cells
> therefore reflect a **pre-fix1 v6 9b-it run**, not the v7 models we have on
> disk now. The v7 ifeval × s13 trains are still in flight; no v7 ifeval evals
> have been launched for s1–s9 yet. ifeval section headers are tagged
> **`[v6 LEGACY]`** and each ifeval section repeats this warning right above
> the table.

**Method-row coverage.** Tables include s1–s9 and s11–s13 (s10 was never run).
If a row shows all `—`, the eval CSV reports `n=0/missing` for that method;
ifeval and humaneval CSVs do not have an s13 row at all (the v6 builders
predate s13).

## gemma-2-2b × membership (eval = rosch, all 10 tasks)

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
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | – | — | — | — | — | — |
| 12 New + NegTC [-fsx] | – | — | — | — | — | — |
| 13 SFT + CFT | ✓ ep=2 | 80.88 ± 2.08 | 82.52 ± 1.98 | 82.82 ± 1.98 | 81.05 ± 2.29 | 69.48 ± 4.55 |

## gemma-2-2b-it × membership (eval = rosch, all 10 tasks)

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
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=2 | 77.10 ± 3.68 | 84.97 ± 2.63 | 87.97 ± 2.77 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 77.93 ± 3.38 | — | — | 83.55 ± 1.74 | 90.02 ± 1.93 |
| 13 SFT + CFT | ✓ ep=2 | 84.95 ± 1.66 | 85.45 ± 1.29 | 88.07 ± 1.33 | 83.69 ± 1.00 | 87.10 ± 2.12 |

## gemma-2-9b-it × membership (eval = rosch, all 10 tasks)

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
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=2 | 88.94 ± 1.57 | 92.39 ± 1.31 | 93.07 ± 1.46 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 86.70 ± 1.89 | — | — | 92.58 ± 1.48 | 88.64 ± 2.70 |
| 13 SFT + CFT | ✓ ep=2 | 81.94 ± 1.98 | 83.17 ± 1.44 | 85.34 ± 1.43 | 81.88 ± 1.17 | 84.18 ± 3.35 |

## gemma-2-2b × persona (all 6 personas)

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
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=1 | 62.21 ± 11.43 | 70.44 ± 10.37 | 68.27 ± 9.59 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 58.96 ± 11.71 | — | — | 72.11 ± 9.43 | — |
| 13 SFT + CFT | ✓ ep=2 | 51.64 ± 8.85 | 57.33 ± 8.03 | 51.78 ± 7.09 | 64.84 ± 6.20 | 66.60 ± 1.68 |

## gemma-2-2b-it × persona (all 6 personas)

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
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=2 | 60.42 ± 11.93 | 65.31 ± 12.79 | 57.54 ± 8.54 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 61.05 ± 11.36 | — | — | 77.65 ± 8.15 | 61.53 ± 6.05 |
| 13 SFT + CFT | ✓ ep=2 | 53.61 ± 8.72 | 58.08 ± 9.21 | 56.21 ± 4.69 | 67.95 ± 6.99 | 64.40 ± 8.07 |

## gemma-2-9b-it × persona (all 6 personas)

Source: [persona_v1_v7_gemma-2-9b-it_all_gen_roc_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-9b-it_all_gen_roc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 42.96 ± 5.31 | — | 40.88 ± 5.40 | — | 57.19 ± 2.20 |
| 1 SFT labelonly 10% | ✓ ep=1 | 57.17 ± 9.57 | 60.69 ± 9.97 | 60.86 ± 8.22 | 69.17 ± 7.54 | 83.73 ± 2.41 |
| 2 RankAlign | ✓ ep=2 | 85.19 ± 7.54 | 95.43 ± 2.55 | 90.06 ± 4.81 | 96.90 ± 1.78 | 97.24 ± 1.55 |
| 3 New + fsx [-TC] | ⏳ 42377 R 6:11:04 (≤2:48:56) | 63.95 ± 12.22 | 72.47 ± 11.22 | 68.94 ± 11.77 | 81.11 ± 7.48 | 91.52 ± 3.27 |
| 4 New + PMI + fsx | ✓ ep=1 | 60.92 ± 10.71 | 68.71 ± 10.76 | 65.11 ± 10.52 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 67.84 ± 11.06 | 85.91 ± 6.69 | 76.24 ± 9.53 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 69.08 ± 11.58 | 83.72 ± 7.72 | 69.10 ± 11.60 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 62.09 ± 10.51 | — | — | 76.39 ± 8.46 | 85.10 ± 2.30 |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=1 | 62.60 ± 10.87 | 71.83 ± 10.52 | 67.94 ± 9.69 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 62.04 ± 10.60 | — | — | 76.64 ± 8.17 | 85.44 ± 2.21 |
| 13 SFT + CFT | ✓ ep=2 | 51.52 ± 8.66 | 57.53 ± 8.85 | 52.49 ± 7.01 | 67.35 ± 5.94 | 71.69 ± 3.22 |

## gemma-2-2b × persona ID (3 in-domain: psychopathy, machiavellianism, narcissism)

Source: [persona_v1_v7_gemma-2-2b_id_gen_roc_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-2b_id_gen_roc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 51.53 ± 2.77 | — | 44.49 ± 5.57 | — | 57.89 ± 5.00 |
| 1 SFT labelonly 10% | – | — | — | — | — | — |
| 2 RankAlign | ✓ ep=1 | — | — | — | — | — |
| 3 New + fsx [-TC] | – | — | — | — | — | — |
| 4 New + PMI + fsx | – | — | — | — | — | — |
| 5 RA + PMI + fsx [-NLL] | – | — | — | — | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | – | — | — | — | — | — |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=1 | 86.66 ± 3.16 | 93.08 ± 2.96 | 88.95 ± 4.45 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 84.10 ± 2.54 | — | — | 92.17 ± 2.23 | — |
| 13 SFT + CFT | ✓ ep=2 | 70.17 ± 2.87 | 73.98 ± 4.76 | 66.17 ± 4.90 | 77.12 ± 3.54 | 68.51 ± 1.63 |

## gemma-2-2b-it × persona ID (3 in-domain: psychopathy, machiavellianism, narcissism)

Source: [persona_v1_v7_gemma-2-2b-it_id_gen_roc_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-2b-it_id_gen_roc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 57.46 ± 1.43 | — | 59.77 ± 4.99 | — | 70.04 ± 1.97 |
| 1 SFT labelonly 10% | ✓ ep=2 | 73.01 ± 2.29 | 77.90 ± 2.65 | 66.90 ± 1.20 | 80.83 ± 3.43 | 66.73 ± 5.92 |
| 2 RankAlign | ✓ ep=2 | 97.89 ± 0.71 | 98.96 ± 0.37 | 68.23 ± 4.91 | 98.85 ± 0.47 | 80.74 ± 1.72 |
| 3 New + fsx [-TC] | ✓ ep=2 | 92.15 ± 1.00 | 96.91 ± 0.60 | 88.69 ± 1.75 | 97.14 ± 0.69 | 69.35 ± 6.55 |
| 4 New + PMI + fsx | ✓ ep=2 | 83.69 ± 1.45 | 90.95 ± 1.13 | 76.93 ± 2.74 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 95.51 ± 0.72 | 99.19 ± 0.13 | 77.09 ± 2.79 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 96.14 ± 1.38 | 99.45 ± 0.32 | 81.63 ± 2.09 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 84.49 ± 0.45 | — | — | 92.67 ± 0.96 | 68.80 ± 3.86 |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=2 | 86.31 ± 1.22 | 93.52 ± 0.40 | 76.22 ± 2.69 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 85.41 ± 1.34 | — | — | 94.08 ± 1.17 | 50.98 ± 5.19 |
| 13 SFT + CFT | ✓ ep=2 | 71.31 ± 2.94 | 77.23 ± 2.96 | 65.24 ± 3.22 | 80.36 ± 3.86 | 49.45 ± 7.78 |

## gemma-2-9b-it × persona ID (3 in-domain: psychopathy, machiavellianism, narcissism)

Source: [persona_v1_v7_gemma-2-9b-it_id_gen_roc_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-9b-it_id_gen_roc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 53.91 ± 1.77 | — | 49.56 ± 6.02 | — | 58.59 ± 2.86 |
| 1 SFT labelonly 10% | ✓ ep=1 | 76.79 ± 2.27 | 81.42 ± 2.81 | 78.35 ± 0.83 | 83.52 ± 2.94 | 80.91 ± 1.71 |
| 2 RankAlign | ✓ ep=2 | 99.05 ± 0.56 | 99.87 ± 0.06 | 98.83 ± 0.66 | 99.86 ± 0.07 | 99.43 ± 0.28 |
| 3 New + fsx [-TC] | ⏳ 42377 R 6:11:04 (≤2:48:56) | 90.27 ± 0.52 | 97.19 ± 0.41 | 95.04 ± 0.65 | 96.78 ± 0.39 | 96.53 ± 0.62 |
| 4 New + PMI + fsx | ✓ ep=1 | 83.12 ± 2.42 | 91.42 ± 1.86 | 87.94 ± 1.97 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 91.32 ± 1.85 | 99.82 ± 0.05 | 97.48 ± 0.45 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 93.74 ± 1.63 | 99.81 ± 0.06 | 95.02 ± 0.57 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 83.80 ± 1.56 | — | — | 92.52 ± 0.99 | 86.78 ± 1.54 |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=1 | 85.08 ± 2.23 | 93.97 ± 1.90 | 88.87 ± 2.03 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 83.94 ± 2.62 | — | — | 92.63 ± 1.99 | 86.40 ± 0.87 |
| 13 SFT + CFT | ✓ ep=2 | 69.28 ± 2.33 | 75.94 ± 2.51 | 66.56 ± 2.63 | 79.05 ± 2.76 | 66.31 ± 3.29 |

## gemma-2-2b × persona OOD (3 held-out: desire-to-create-allies, interest-in-music, interest-in-science)

Source: [persona_v1_v7_gemma-2-2b_ood_gen_roc_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-2b_ood_gen_roc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 28.49 ± 4.77 | — | 31.16 ± 0.24 | — | 89.89 ± 1.16 |
| 1 SFT labelonly 10% | – | — | — | — | — | — |
| 2 RankAlign | ✓ ep=1 | — | — | — | — | — |
| 3 New + fsx [-TC] | – | — | — | — | — | — |
| 4 New + PMI + fsx | – | — | — | — | — | — |
| 5 RA + PMI + fsx [-NLL] | – | — | — | — | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | – | — | — | — | — | — |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=1 | 37.75 ± 6.69 | 47.81 ± 4.05 | 47.58 ± 3.47 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 33.82 ± 6.89 | — | — | 52.05 ± 6.11 | — |
| 13 SFT + CFT | ✓ ep=2 | 33.10 ± 6.33 | 40.68 ± 4.72 | 37.38 ± 4.49 | 52.56 ± 5.36 | 64.69 ± 2.81 |

## gemma-2-2b-it × persona OOD (3 held-out: desire-to-create-allies, interest-in-music, interest-in-science)

Source: [persona_v1_v7_gemma-2-2b-it_ood_gen_roc_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-2b-it_ood_gen_roc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 39.65 ± 7.25 | — | 55.03 ± 3.84 | — | 81.61 ± 4.06 |
| 1 SFT labelonly 10% | ✓ ep=2 | 36.67 ± 7.41 | 40.91 ± 5.85 | 46.08 ± 4.71 | 56.85 ± 7.98 | 78.32 ± 6.82 |
| 2 RankAlign | ✓ ep=2 | 58.71 ± 11.91 | 73.90 ± 9.10 | 50.70 ± 3.00 | 80.65 ± 8.75 | 87.12 ± 4.57 |
| 3 New + fsx [-TC] | ✓ ep=2 | 41.58 ± 7.76 | 49.42 ± 5.11 | 48.84 ± 1.43 | 63.54 ± 6.70 | 75.85 ± 6.31 |
| 4 New + PMI + fsx | ✓ ep=2 | 37.34 ± 7.01 | 41.60 ± 5.24 | 47.69 ± 0.82 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 37.30 ± 5.69 | 45.95 ± 1.76 | 41.32 ± 6.50 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 51.90 ± 8.91 | 71.84 ± 6.46 | 66.46 ± 5.79 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 33.45 ± 6.39 | — | — | 48.57 ± 6.72 | 70.38 ± 4.23 |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=2 | 34.53 ± 6.32 | 37.09 ± 4.69 | 38.86 ± 2.89 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 36.68 ± 7.05 | — | — | 61.22 ± 7.82 | 72.09 ± 6.70 |
| 13 SFT + CFT | ✓ ep=2 | 35.90 ± 7.62 | 38.93 ± 7.00 | 47.18 ± 4.25 | 55.54 ± 8.68 | 79.35 ± 6.44 |

## gemma-2-9b-it × persona OOD (3 held-out: desire-to-create-allies, interest-in-music, interest-in-science)

Source: [persona_v1_v7_gemma-2-9b-it_ood_gen_roc_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-9b-it_ood_gen_roc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 32.01 ± 4.25 | — | 32.20 ± 5.85 | — | 55.79 ± 3.74 |
| 1 SFT labelonly 10% | ✓ ep=1 | 37.54 ± 8.20 | 39.96 ± 7.71 | 43.38 ± 5.62 | 54.81 ± 8.33 | 86.55 ± 4.27 |
| 2 RankAlign | ✓ ep=2 | 71.34 ± 9.58 | 90.99 ± 3.59 | 81.29 ± 6.18 | 93.95 ± 2.66 | 95.05 ± 2.67 |
| 3 New + fsx [-TC] | ⏳ 42377 R 6:11:04 (≤2:48:56) | 37.63 ± 7.34 | 47.75 ± 4.27 | 42.85 ± 3.40 | 65.44 ± 5.85 | 86.50 ± 5.28 |
| 4 New + PMI + fsx | ✓ ep=1 | 38.71 ± 8.65 | 46.00 ± 7.72 | 42.28 ± 5.29 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 44.36 ± 7.52 | 72.00 ± 5.52 | 55.00 ± 1.57 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 44.42 ± 7.72 | 67.63 ± 6.24 | 43.18 ± 0.47 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 40.38 ± 8.88 | — | — | 60.26 ± 9.82 | 83.41 ± 4.60 |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=1 | 40.12 ± 8.99 | 49.70 ± 7.70 | 47.00 ± 5.20 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 40.13 ± 8.67 | — | — | 60.65 ± 8.58 | 84.48 ± 4.77 |
| 13 SFT + CFT | ✓ ep=2 | 33.76 ± 7.33 | 39.12 ± 6.78 | 38.43 ± 6.38 | 55.65 ± 5.64 | 77.07 ± 3.48 |

## gemma-2-9b-it × ifeval ID  **[v6 LEGACY]** (n=79 prompts ≥ 22; held-out 50% of completions for each)

> **⚠️ V6 LEGACY DATA — NOT v7.** All cells in this section come from
> a pre-fix1 v6 training run (`v6-google_gemma-2-9b-it-delta0.15-epoch2_ifeval-concat-all`). No v7 ifeval evaluations have been run yet (v7 × s13 trains in flight; v7 × s1–s9 not started). Treat these numbers as legacy reference, not as v7 results.

Source: [ifeval_id_table_cells.csv](metrics-from-scores/ifeval_id_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 58.72 ± 2.13 | — | 81.64 ± 1.43 | — | 65.86 ± 2.11 |
| 1 SFT labelonly 10% | ✓ (prior run) | 56.98 ± 2.06 | 62.99 ± 2.00 | 75.39 ± 1.75 | — | — |
| 2 RankAlign | ✓ (prior run) | 59.54 ± 2.06 | — | 74.56 ± 1.83 | — | 65.21 ± 2.06 |
| 3 New + fsx [-TC] | ✓ (prior run) | 71.97 ± 2.22 | 78.29 ± 2.16 | 83.67 ± 1.78 | — | 71.01 ± 2.09 |
| 4 New + PMI + fsx | ✓ (prior run) | 69.46 ± 2.15 | 76.86 ± 2.07 | 82.98 ± 1.88 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ (prior run) | 62.39 ± 2.13 | 79.36 ± 1.81 | 83.71 ± 1.52 | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | ✓ (prior run) | 72.25 ± 2.08 | — | — | — | 56.59 ± 2.33 |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | – | — | — | — | — | — |
| 12 New + NegTC [-fsx] | – | — | — | — | — | — |
| 13 SFT + CFT | ⏳ 42343 PD | — | — | — | — | — |

## gemma-2-9b-it × ifeval OOD **[v6 LEGACY]** (n=20 fully held-out prompts: prompt_1..13, 15..21)

> **⚠️ V6 LEGACY DATA — NOT v7.** All cells in this section come from
> a pre-fix1 v6 training run (`v6-google_gemma-2-9b-it-delta0.15-epoch2_ifeval-concat-all`). No v7 ifeval evaluations have been run yet (v7 × s13 trains in flight; v7 × s1–s9 not started). Treat these numbers as legacy reference, not as v7 results.

Source: [ifeval_ood_table_cells.csv](metrics-from-scores/ifeval_ood_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 52.81 ± 4.39 | — | 78.30 ± 2.22 | — | 59.27 ± 2.66 |
| 1 SFT labelonly 10% | ✓ (prior run) | 51.18 ± 4.25 | 57.64 ± 4.03 | 74.53 ± 2.60 | — | — |
| 2 RankAlign | ✓ (prior run) | 51.87 ± 4.28 | — | 74.88 ± 3.58 | — | 59.42 ± 2.68 |
| 3 New + fsx [-TC] | ✓ (prior run) | 60.16 ± 4.19 | 82.02 ± 2.41 | 83.53 ± 2.47 | — | 67.27 ± 3.59 |
| 4 New + PMI + fsx | ✓ (prior run) | 59.65 ± 4.39 | 82.03 ± 2.58 | 85.06 ± 2.71 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ (prior run) | 54.01 ± 4.44 | 78.40 ± 3.24 | 83.39 ± 2.60 | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | ✓ (prior run) | 60.77 ± 4.26 | — | — | — | 56.96 ± 3.79 |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | – | — | — | — | — | — |
| 12 New + NegTC [-fsx] | – | — | — | — | — | — |
| 13 SFT + CFT | ⏳ 42343 PD | — | — | — | — | — |

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
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | – | — | — | — | — | — |
| 12 New + NegTC [-fsx] | – | — | — | — | — | — |
| 13 SFT + CFT | ⏳ 42114 R 15:50:04 (≤8:09:56) | — | — | — | — | — |
