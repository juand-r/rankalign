# Morning Results — Pearson(gen, val) — 2026-05-25T15:58:15Z

All cells: **Pearson(gen, val) × 100 ± SE** (mean ± SE across the eval-task split for that section).

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
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | – | — | — | — | — | — |
| 12 New + NegTC [-fsx] | – | — | — | — | — | — |
| 13 SFT + CFT | ✓ ep=2 | 53.55 ± 3.59 | 60.42 ± 2.45 | 60.75 ± 2.57 | 50.44 ± 4.29 | 36.03 ± 5.82 |

## gemma-2-2b-it × membership (eval = rosch, all 10 tasks)

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
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=2 | 50.19 ± 5.33 | 65.89 ± 3.00 | 74.97 ± 2.17 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 48.48 ± 5.13 | — | — | 60.26 ± 2.27 | 69.78 ± 2.27 |
| 13 SFT + CFT | ✓ ep=2 | 56.54 ± 3.82 | 57.16 ± 2.60 | 65.06 ± 2.94 | 49.57 ± 2.40 | 64.03 ± 2.65 |

## gemma-2-9b-it × membership (eval = rosch, all 10 tasks)

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
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=2 | 67.86 ± 3.13 | 73.68 ± 2.42 | 76.75 ± 2.52 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 66.50 ± 2.71 | — | — | 70.97 ± 2.98 | 63.04 ± 5.26 |
| 13 SFT + CFT | ✓ ep=2 | 48.94 ± 4.45 | 50.84 ± 3.42 | 55.37 ± 3.46 | 44.92 ± 3.56 | 54.13 ± 4.91 |

## gemma-2-2b × persona (all 6 personas)

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
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=1 | 26.13 ± 15.91 | 37.76 ± 15.70 | 37.41 ± 12.70 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 30.36 ± 12.57 | — | — | 45.03 ± 12.88 | — |
| 13 SFT + CFT | ✓ ep=2 | 26.57 ± 6.58 | 18.83 ± 11.73 | 12.22 ± 8.60 | 27.23 ± 10.50 | 24.42 ± 4.73 |

## gemma-2-2b-it × persona (all 6 personas)

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
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=2 | 24.56 ± 16.20 | 30.69 ± 19.16 | 16.76 ± 11.65 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 28.58 ± 14.29 | — | — | 48.16 ± 11.81 | 13.34 ± 8.20 |
| 13 SFT + CFT | ✓ ep=2 | 17.17 ± 9.65 | 18.76 ± 13.32 | 10.03 ± 7.30 | 31.63 ± 11.28 | 9.79 ± 9.83 |

## gemma-2-9b-it × persona (all 6 personas)

Source: [persona_v1_v7_gemma-2-9b-it_all_pearson_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-9b-it_all_pearson_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -7.98 ± 9.99 | — | -11.10 ± 8.07 | — | 18.46 ± 2.19 |
| 1 SFT labelonly 10% | ✓ ep=1 | 14.04 ± 13.37 | 19.94 ± 15.53 | 14.68 ± 14.57 | 33.71 ± 11.60 | 49.65 ± 3.46 |
| 2 RankAlign | ✓ ep=2 | 60.78 ± 12.01 | 76.01 ± 5.32 | 65.09 ± 7.21 | 79.05 ± 4.21 | 77.62 ± 3.75 |
| 3 New + fsx [-TC] | ⏳ 42377 R 6:11:04 (≤2:48:56) | 30.68 ± 17.32 | 41.98 ± 17.98 | 33.88 ± 19.04 | 55.93 ± 11.44 | 66.78 ± 5.31 |
| 4 New + PMI + fsx | ✓ ep=1 | 21.13 ± 15.49 | 33.08 ± 17.38 | 24.07 ± 17.46 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 30.00 ± 17.18 | 64.48 ± 10.89 | 44.72 ± 15.05 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 32.42 ± 17.62 | 62.66 ± 11.43 | 34.28 ± 17.02 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 22.58 ± 15.30 | — | — | 47.08 ± 12.45 | 57.07 ± 4.09 |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=1 | 23.49 ± 16.23 | 39.01 ± 16.91 | 27.71 ± 16.58 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 22.38 ± 15.68 | — | — | 46.53 ± 12.67 | 55.39 ± 4.42 |
| 13 SFT + CFT | ✓ ep=2 | 5.26 ± 12.54 | 13.28 ± 14.38 | 2.39 ± 11.82 | 30.64 ± 9.52 | 33.37 ± 4.78 |

## gemma-2-2b × persona ID (3 in-domain: psychopathy, machiavellianism, narcissism)

Source: [persona_v1_v7_gemma-2-2b_id_pearson_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-2b_id_pearson_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 11.69 ± 3.02 | — | -14.51 ± 11.18 | — | 28.55 ± 5.16 |
| 1 SFT labelonly 10% | – | — | — | — | — | — |
| 2 RankAlign | ✓ ep=1 | — | — | — | — | — |
| 3 New + fsx [-TC] | – | — | — | — | — | — |
| 4 New + PMI + fsx | – | — | — | — | — | — |
| 5 RA + PMI + fsx [-NLL] | – | — | — | — | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | – | — | — | — | — | — |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=1 | 60.69 ± 5.18 | 72.27 ± 5.13 | 64.76 ± 6.77 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 57.10 ± 3.14 | — | — | 72.50 ± 3.37 | — |
| 13 SFT + CFT | ✓ ep=2 | 40.42 ± 4.72 | 43.83 ± 7.74 | 29.65 ± 7.87 | 49.56 ± 5.49 | 31.84 ± 2.90 |

## gemma-2-2b-it × persona ID (3 in-domain: psychopathy, machiavellianism, narcissism)

Source: [persona_v1_v7_gemma-2-2b-it_id_pearson_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-2b-it_id_pearson_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 20.97 ± 1.96 | — | 21.61 ± 7.39 | — | 44.57 ± 3.15 |
| 1 SFT labelonly 10% | ✓ ep=2 | 35.21 ± 4.49 | 45.61 ± 5.46 | 26.03 ± 2.30 | 50.78 ± 6.53 | 27.14 ± 8.88 |
| 2 RankAlign | ✓ ep=2 | 62.47 ± 4.01 | 67.44 ± 3.73 | 36.67 ± 3.10 | 66.98 ± 4.26 | 44.62 ± 1.15 |
| 3 New + fsx [-TC] | ✓ ep=2 | 72.34 ± 1.36 | 78.28 ± 1.79 | 61.00 ± 2.04 | 79.23 ± 1.14 | 30.50 ± 10.40 |
| 4 New + PMI + fsx | ✓ ep=2 | 58.00 ± 1.99 | 70.33 ± 2.28 | 44.20 ± 4.96 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 67.57 ± 2.59 | 76.56 ± 2.71 | 47.53 ± 4.36 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 71.44 ± 1.04 | 80.70 ± 0.75 | 53.67 ± 3.92 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 59.06 ± 0.94 | — | — | 71.96 ± 2.29 | 30.97 ± 3.96 |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=2 | 60.57 ± 0.91 | 73.08 ± 0.94 | 41.95 ± 3.47 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 60.20 ± 1.45 | — | — | 72.74 ± 1.50 | 1.38 ± 8.24 |
| 13 SFT + CFT | ✓ ep=2 | 36.25 ± 5.78 | 46.69 ± 6.07 | 23.62 ± 7.94 | 52.18 ± 7.01 | -5.23 ± 11.05 |

## gemma-2-9b-it × persona ID (3 in-domain: psychopathy, machiavellianism, narcissism)

Source: [persona_v1_v7_gemma-2-9b-it_id_pearson_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-9b-it_id_pearson_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 13.50 ± 2.69 | — | 5.16 ± 2.67 | — | 20.48 ± 0.68 |
| 1 SFT labelonly 10% | ✓ ep=1 | 43.23 ± 3.49 | 53.37 ± 4.95 | 47.14 ± 0.93 | 56.96 ± 5.25 | 52.86 ± 3.20 |
| 2 RankAlign | ✓ ep=2 | 83.83 ± 3.21 | 85.94 ± 2.84 | 78.07 ± 3.26 | 86.33 ± 2.76 | 81.32 ± 2.32 |
| 3 New + fsx [-TC] | ⏳ 42377 R 6:11:04 (≤2:48:56) | 69.19 ± 0.59 | 82.16 ± 1.13 | 76.18 ± 1.74 | 80.76 ± 1.28 | 77.52 ± 0.99 |
| 4 New + PMI + fsx | ✓ ep=1 | 55.00 ± 2.83 | 71.35 ± 2.90 | 62.67 ± 2.65 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 67.47 ± 2.91 | 88.07 ± 1.11 | 78.27 ± 1.47 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 70.64 ± 3.25 | 87.03 ± 2.56 | 72.22 ± 2.52 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 56.03 ± 1.13 | — | — | 72.99 ± 1.58 | 60.88 ± 3.06 |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=1 | 58.72 ± 2.47 | 75.64 ± 3.37 | 64.14 ± 3.10 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 56.41 ± 3.04 | — | — | 72.84 ± 3.24 | 61.38 ± 1.06 |
| 13 SFT + CFT | ✓ ep=2 | 32.29 ± 4.50 | 44.31 ± 3.84 | 27.34 ± 3.71 | 49.93 ± 4.44 | 27.41 ± 4.92 |

## gemma-2-2b × persona OOD (3 held-out: desire-to-create-allies, interest-in-music, interest-in-science)

Source: [persona_v1_v7_gemma-2-2b_ood_pearson_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-2b_ood_pearson_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -27.88 ± 6.91 | — | -32.55 ± 0.50 | — | 66.28 ± 2.99 |
| 1 SFT labelonly 10% | – | — | — | — | — | — |
| 2 RankAlign | ✓ ep=1 | — | — | — | — | — |
| 3 New + fsx [-TC] | – | — | — | — | — | — |
| 4 New + PMI + fsx | – | — | — | — | — | — |
| 5 RA + PMI + fsx [-NLL] | – | — | — | — | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | – | — | — | — | — | — |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=1 | -8.43 ± 6.72 | 3.26 ± 4.00 | 10.06 ± 3.48 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 3.62 ± 8.05 | — | — | 17.56 ± 7.99 | — |
| 13 SFT + CFT | ✓ ep=2 | 12.72 ± 1.58 | -6.17 ± 1.77 | -5.21 ± 1.93 | 4.90 ± 4.67 | 16.99 ± 6.95 |

## gemma-2-2b-it × persona OOD (3 held-out: desire-to-create-allies, interest-in-music, interest-in-science)

Source: [persona_v1_v7_gemma-2-2b-it_ood_pearson_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-2b-it_ood_pearson_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -10.05 ± 8.39 | — | 8.83 ± 7.57 | — | 57.03 ± 9.57 |
| 1 SFT labelonly 10% | ✓ ep=2 | -8.58 ± 4.66 | -11.81 ± 6.39 | -11.77 ± 2.38 | 9.63 ± 11.58 | 30.55 ± 8.79 |
| 2 RankAlign | ✓ ep=2 | 13.82 ± 11.15 | 39.87 ± 7.25 | 21.92 ± 5.22 | 47.05 ± 9.28 | 35.96 ± 11.05 |
| 3 New + fsx [-TC] | ✓ ep=2 | 3.60 ± 8.65 | 10.72 ± 5.71 | 3.97 ± 2.37 | 27.58 ± 8.92 | 32.32 ± 11.04 |
| 4 New + PMI + fsx | ✓ ep=2 | -10.14 ± 4.36 | -10.48 ± 2.73 | -2.30 ± 2.93 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | -15.15 ± 4.42 | 6.04 ± 6.02 | -6.58 ± 6.06 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 4.17 ± 13.58 | 41.20 ± 8.56 | 30.31 ± 8.56 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | -2.09 ± 2.93 | — | — | 12.93 ± 5.25 | 24.03 ± 5.66 |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=2 | -11.44 ± 3.78 | -11.70 ± 6.22 | -8.44 ± 5.61 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | -3.05 ± 4.32 | — | — | 23.57 ± 9.54 | 25.30 ± 11.19 |
| 13 SFT + CFT | ✓ ep=2 | -1.90 ± 8.28 | -9.18 ± 8.37 | -3.57 ± 4.29 | 11.07 ± 12.81 | 24.82 ± 11.62 |

## gemma-2-9b-it × persona OOD (3 held-out: desire-to-create-allies, interest-in-music, interest-in-science)

Source: [persona_v1_v7_gemma-2-9b-it_ood_pearson_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-9b-it_ood_pearson_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -29.47 ± 5.49 | — | -27.37 ± 7.36 | — | 16.44 ± 4.42 |
| 1 SFT labelonly 10% | ✓ ep=1 | -15.15 ± 5.45 | -13.49 ± 7.99 | -17.77 ± 2.66 | 10.45 ± 10.24 | 46.44 ± 6.28 |
| 2 RankAlign | ✓ ep=2 | 37.74 ± 13.39 | 66.08 ± 5.90 | 52.11 ± 9.00 | 71.76 ± 5.30 | 73.93 ± 7.17 |
| 3 New + fsx [-TC] | ⏳ 42377 R 6:11:04 (≤2:48:56) | -7.84 ± 3.99 | 1.80 ± 1.12 | -8.42 ± 4.59 | 31.10 ± 6.05 | 56.05 ± 4.97 |
| 4 New + PMI + fsx | ✓ ep=1 | -12.74 ± 6.67 | -5.18 ± 6.16 | -14.53 ± 5.23 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | -7.47 ± 8.02 | 40.89 ± 5.92 | 11.18 ± 2.35 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | -5.81 ± 9.01 | 38.29 ± 7.23 | -3.66 ± 1.52 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | -10.87 ± 7.03 | — | — | 21.18 ± 10.08 | 53.26 ± 7.73 |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=1 | -11.74 ± 8.38 | 2.38 ± 8.70 | -8.72 ± 6.20 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | -11.65 ± 7.89 | — | — | 20.22 ± 9.99 | 49.40 ± 7.79 |
| 13 SFT + CFT | ✓ ep=2 | -21.76 ± 6.02 | -17.75 ± 7.49 | -22.55 ± 7.92 | 11.36 ± 7.87 | 39.32 ± 7.39 |

## gemma-2-9b-it × ifeval ID  **[v6 LEGACY]** (n=79 prompts ≥ 22; held-out 50% of completions for each)

> **⚠️ V6 LEGACY DATA — NOT v7.** All cells in this section come from
> a pre-fix1 v6 training run (`v6-google_gemma-2-9b-it-delta0.15-epoch2_ifeval-concat-all`). No v7 ifeval evaluations have been run yet (v7 × s13 trains in flight; v7 × s1–s9 not started). Treat these numbers as legacy reference, not as v7 results.

Source: [ifeval_id_pearson_table_cells.csv](metrics-from-scores/ifeval_id_pearson_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 18.06 ± 3.52 | — | 46.16 ± 3.00 | — | 25.32 ± 3.96 |
| 1 SFT labelonly 10% | ✓ (prior run) | 12.77 ± 3.67 | 19.57 ± 3.74 | 35.57 ± 3.37 | — | — |
| 2 RankAlign | ✓ (prior run) | 22.90 ± 3.59 | — | 41.97 ± 3.79 | — | 25.02 ± 4.27 |
| 3 New + fsx [-TC] | ✓ (prior run) | 60.58 ± 3.37 | 69.75 ± 2.91 | 72.03 ± 2.63 | — | 59.02 ± 3.61 |
| 4 New + PMI + fsx | ✓ (prior run) | 47.71 ± 3.35 | 68.16 ± 2.97 | 71.50 ± 2.70 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ (prior run) | 33.91 ± 3.28 | 66.80 ± 3.30 | 70.34 ± 3.15 | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | ✓ (prior run) | 57.32 ± 3.15 | — | — | — | 19.17 ± 5.12 |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | – | — | — | — | — | — |
| 12 New + NegTC [-fsx] | – | — | — | — | — | — |
| 13 SFT + CFT | ⏳ 42343 PD | — | — | — | — | — |

## gemma-2-9b-it × ifeval OOD **[v6 LEGACY]** (n=20 fully held-out prompts: prompt_1..13, 15..21)

> **⚠️ V6 LEGACY DATA — NOT v7.** All cells in this section come from
> a pre-fix1 v6 training run (`v6-google_gemma-2-9b-it-delta0.15-epoch2_ifeval-concat-all`). No v7 ifeval evaluations have been run yet (v7 × s13 trains in flight; v7 × s1–s9 not started). Treat these numbers as legacy reference, not as v7 results.

Source: [ifeval_ood_pearson_table_cells.csv](metrics-from-scores/ifeval_ood_pearson_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 9.25 ± 7.54 | — | 35.54 ± 7.47 | — | 25.06 ± 5.76 |
| 1 SFT labelonly 10% | ✓ (prior run) | 7.33 ± 7.11 | 13.18 ± 6.97 | 29.53 ± 6.27 | — | — |
| 2 RankAlign | ✓ (prior run) | 2.03 ± 8.15 | — | 29.87 ± 7.63 | — | 14.47 ± 6.19 |
| 3 New + fsx [-TC] | ✓ (prior run) | 29.80 ± 7.16 | 57.18 ± 5.16 | 56.86 ± 6.48 | — | 54.65 ± 3.93 |
| 4 New + PMI + fsx | ✓ (prior run) | 30.29 ± 6.88 | 62.84 ± 4.44 | 61.67 ± 5.21 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ (prior run) | 15.57 ± 7.42 | 47.62 ± 7.18 | 51.97 ± 6.53 | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | ✓ (prior run) | 34.36 ± 6.42 | — | — | — | 14.93 ± 8.92 |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | – | — | — | — | — | — |
| 12 New + NegTC [-fsx] | – | — | — | — | — | — |
| 13 SFT + CFT | ⏳ 42343 PD | — | — | — | — | — |

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
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | – | — | — | — | — | — |
| 12 New + NegTC [-fsx] | – | — | — | — | — | — |
| 13 SFT + CFT | ⏳ 42114 R 15:50:04 (≤8:09:56) | — | — | — | — | — |
