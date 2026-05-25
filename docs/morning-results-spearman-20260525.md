# Morning Results — Spearman(gen, val) — 2026-05-25T15:58:15Z

All cells: **Spearman(gen, val) × 100 ± SE** (mean ± SE across the eval-task split for that section).

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
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | – | — | — | — | — | — |
| 12 New + NegTC [-fsx] | – | — | — | — | — | — |
| 13 SFT + CFT | ✓ ep=2 | 59.93 ± 3.94 | 63.62 ± 2.11 | — | 52.51 ± 4.23 | — |

## gemma-2-2b-it × membership (eval = rosch, all 10 tasks)

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
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=2 | 56.95 ± 4.93 | 68.83 ± 2.00 | 77.85 ± 1.69 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 54.20 ± 4.18 | — | — | 59.58 ± 2.94 | 69.39 ± 2.89 |
| 13 SFT + CFT | ✓ ep=2 | 60.21 ± 3.80 | 57.03 ± 2.41 | 64.23 ± 2.43 | 51.59 ± 2.69 | 63.39 ± 2.57 |

## gemma-2-9b-it × membership (eval = rosch, all 10 tasks)

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
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=2 | 72.56 ± 2.65 | 75.51 ± 1.99 | 79.85 ± 2.08 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 70.27 ± 2.22 | — | — | 74.04 ± 2.51 | 65.89 ± 5.23 |
| 13 SFT + CFT | ✓ ep=2 | 57.48 ± 3.99 | 57.75 ± 2.73 | — | 51.38 ± 3.07 | — |

## gemma-2-2b × persona (all 6 personas)

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
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=1 | 28.38 ± 17.00 | 37.12 ± 15.86 | 36.75 ± 13.45 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 31.58 ± 13.46 | — | — | 44.12 ± 11.99 | — |
| 13 SFT + CFT | ✓ ep=2 | 30.52 ± 8.63 | 20.97 ± 12.61 | 15.11 ± 10.28 | 29.58 ± 11.59 | 25.86 ± 5.61 |

## gemma-2-2b-it × persona (all 6 personas)

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
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=2 | 25.59 ± 16.26 | 30.91 ± 17.93 | 16.93 ± 12.58 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 29.63 ± 14.69 | — | — | 49.87 ± 11.53 | 16.13 ± 7.80 |
| 13 SFT + CFT | ✓ ep=2 | 20.01 ± 11.18 | 18.54 ± 13.36 | 11.94 ± 7.65 | 31.13 ± 11.34 | 12.69 ± 10.28 |

## gemma-2-9b-it × persona (all 6 personas)

Source: [persona_v1_v7_gemma-2-9b-it_all_spearman_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-9b-it_all_spearman_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -7.20 ± 9.36 | — | -8.95 ± 7.18 | — | 19.28 ± 1.21 |
| 1 SFT labelonly 10% | ✓ ep=1 | 21.23 ± 15.18 | 21.90 ± 16.28 | 20.14 ± 14.06 | 35.45 ± 11.38 | 53.20 ± 3.16 |
| 2 RankAlign | ✓ ep=2 | 58.53 ± 10.80 | 72.25 ± 3.27 | 65.42 ± 6.83 | 74.89 ± 2.13 | 79.26 ± 2.27 |
| 3 New + fsx [-TC] | ⏳ 42377 R 6:11:04 (≤2:48:56) | 38.49 ± 17.13 | 42.44 ± 16.61 | 37.93 ± 17.93 | 54.84 ± 10.35 | 68.08 ± 4.55 |
| 4 New + PMI + fsx | ✓ ep=1 | 27.74 ± 16.24 | 33.76 ± 16.27 | 26.69 ± 16.80 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 33.16 ± 17.25 | 61.22 ± 9.22 | 45.82 ± 13.72 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 36.48 ± 17.51 | 59.57 ± 9.71 | 35.84 ± 15.95 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 27.83 ± 16.18 | — | — | 46.36 ± 11.37 | 56.18 ± 3.84 |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=1 | 31.93 ± 16.13 | 38.92 ± 15.57 | 29.66 ± 15.84 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 30.20 ± 15.84 | — | — | 47.91 ± 11.35 | 56.59 ± 3.65 |
| 13 SFT + CFT | ✓ ep=2 | 12.69 ± 14.74 | 15.87 ± 14.71 | 7.16 ± 12.08 | 31.89 ± 9.50 | 34.96 ± 5.24 |

## gemma-2-2b × persona ID (3 in-domain: psychopathy, machiavellianism, narcissism)

Source: [persona_v1_v7_gemma-2-2b_id_spearman_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-2b_id_spearman_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 10.83 ± 2.83 | — | -12.68 ± 12.36 | — | 26.27 ± 4.57 |
| 1 SFT labelonly 10% | – | — | — | — | — | — |
| 2 RankAlign | ✓ ep=1 | — | — | — | — | — |
| 3 New + fsx [-TC] | – | — | — | — | — | — |
| 4 New + PMI + fsx | – | — | — | — | — | — |
| 5 RA + PMI + fsx [-NLL] | – | — | — | — | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | – | — | — | — | — | — |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=1 | 65.46 ± 3.90 | 72.12 ± 4.79 | 65.87 ± 7.11 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 59.99 ± 2.32 | — | — | 69.26 ± 2.20 | — |
| 13 SFT + CFT | ✓ ep=2 | 48.83 ± 5.77 | 47.28 ± 9.63 | 35.31 ± 10.56 | 53.67 ± 7.02 | 35.56 ± 3.63 |

## gemma-2-2b-it × persona ID (3 in-domain: psychopathy, machiavellianism, narcissism)

Source: [persona_v1_v7_gemma-2-2b-it_id_spearman_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-2b-it_id_spearman_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 20.20 ± 2.62 | — | 21.15 ± 7.29 | — | 46.06 ± 2.66 |
| 1 SFT labelonly 10% | ✓ ep=2 | 36.76 ± 2.24 | 43.82 ± 5.59 | 24.08 ± 0.98 | 49.48 ± 5.67 | 28.07 ± 7.46 |
| 2 RankAlign | ✓ ep=2 | 74.61 ± 4.56 | 76.20 ± 3.07 | 35.69 ± 5.41 | 75.74 ± 3.68 | 55.72 ± 4.38 |
| 3 New + fsx [-TC] | ✓ ep=2 | 72.16 ± 0.81 | 77.35 ± 1.60 | 63.00 ± 2.07 | 78.25 ± 1.20 | 32.72 ± 8.74 |
| 4 New + PMI + fsx | ✓ ep=2 | 62.54 ± 1.28 | 69.17 ± 1.48 | 47.23 ± 6.16 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 71.92 ± 1.92 | 78.17 ± 1.54 | 42.85 ± 5.31 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 74.19 ± 3.04 | 78.84 ± 0.35 | 55.39 ± 4.26 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 60.81 ± 2.42 | — | — | 70.19 ± 2.73 | 33.61 ± 2.89 |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=2 | 61.56 ± 1.86 | 70.28 ± 1.96 | 43.99 ± 4.55 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 61.89 ± 1.01 | — | — | 73.07 ± 1.28 | 5.38 ± 6.02 |
| 13 SFT + CFT | ✓ ep=2 | 43.59 ± 4.31 | 46.49 ± 6.46 | 26.75 ± 6.95 | 52.22 ± 6.59 | -2.78 ± 9.31 |

## gemma-2-9b-it × persona ID (3 in-domain: psychopathy, machiavellianism, narcissism)

Source: [persona_v1_v7_gemma-2-9b-it_id_spearman_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-9b-it_id_spearman_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 12.97 ± 1.22 | — | 4.65 ± 4.67 | — | 19.34 ± 0.45 |
| 1 SFT labelonly 10% | ✓ ep=1 | 53.46 ± 2.53 | 57.02 ± 5.34 | 50.29 ± 1.06 | 58.29 ± 5.66 | 51.56 ± 4.39 |
| 2 RankAlign | ✓ ep=2 | 79.49 ± 1.19 | 78.67 ± 2.28 | 78.85 ± 1.66 | 78.91 ± 2.24 | 83.99 ± 1.08 |
| 3 New + fsx [-TC] | ⏳ 42377 R 6:11:04 (≤2:48:56) | 76.57 ± 0.74 | 79.55 ± 0.85 | 77.68 ± 0.91 | 77.19 ± 0.33 | 76.82 ± 2.68 |
| 4 New + PMI + fsx | ✓ ep=1 | 63.65 ± 1.48 | 69.76 ± 1.34 | 63.92 ± 1.56 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 70.61 ± 2.09 | 80.56 ± 0.36 | 76.35 ± 1.46 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 74.19 ± 2.18 | 79.50 ± 0.29 | 71.33 ± 1.89 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 63.43 ± 1.10 | — | — | 70.51 ± 1.06 | 58.00 ± 5.32 |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=1 | 66.77 ± 2.77 | 72.76 ± 2.18 | 64.58 ± 3.02 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 65.02 ± 1.37 | — | — | 71.81 ± 1.71 | 59.49 ± 2.70 |
| 13 SFT + CFT | ✓ ep=2 | 44.99 ± 2.50 | 47.49 ± 5.50 | 32.88 ± 4.13 | 51.01 ± 6.48 | 26.54 ± 5.51 |

## gemma-2-2b × persona OOD (3 held-out: desire-to-create-allies, interest-in-music, interest-in-science)

Source: [persona_v1_v7_gemma-2-2b_ood_spearman_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-2b_ood_spearman_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -23.47 ± 10.67 | — | -34.14 ± 1.28 | — | 66.56 ± 1.92 |
| 1 SFT labelonly 10% | – | — | — | — | — | — |
| 2 RankAlign | ✓ ep=1 | — | — | — | — | — |
| 3 New + fsx [-TC] | – | — | — | — | — | — |
| 4 New + PMI + fsx | – | — | — | — | — | — |
| 5 RA + PMI + fsx [-NLL] | – | — | — | — | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | – | — | — | — | — | — |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=1 | -8.70 ± 7.40 | 2.11 ± 2.96 | 7.63 ± 2.47 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 3.18 ± 9.69 | — | — | 18.98 ± 9.06 | — |
| 13 SFT + CFT | ✓ ep=2 | 12.22 ± 2.00 | -5.34 ± 3.18 | -5.10 ± 3.03 | 5.49 ± 6.52 | 16.17 ± 7.08 |

## gemma-2-2b-it × persona OOD (3 held-out: desire-to-create-allies, interest-in-music, interest-in-science)

Source: [persona_v1_v7_gemma-2-2b-it_ood_spearman_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-2b-it_ood_spearman_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -7.24 ± 10.09 | — | 8.35 ± 7.27 | — | 55.44 ± 9.16 |
| 1 SFT labelonly 10% | ✓ ep=2 | -10.49 ± 5.10 | -11.54 ± 6.11 | -8.94 ± 2.97 | 10.53 ± 10.58 | 31.21 ± 9.63 |
| 2 RankAlign | ✓ ep=2 | 20.69 ± 13.00 | 46.49 ± 8.21 | 12.77 ± 3.23 | 53.70 ± 10.13 | 44.52 ± 12.70 |
| 3 New + fsx [-TC] | ✓ ep=2 | 2.75 ± 9.02 | 10.45 ± 5.05 | 3.07 ± 1.36 | 29.10 ± 9.07 | 35.10 ± 11.83 |
| 4 New + PMI + fsx | ✓ ep=2 | -11.63 ± 5.45 | -10.68 ± 2.93 | -1.51 ± 3.12 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | -16.14 ± 5.15 | 4.44 ± 4.75 | -3.44 ± 7.44 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 7.95 ± 12.75 | 41.13 ± 8.60 | 28.13 ± 6.21 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | -2.87 ± 5.65 | — | — | 14.05 ± 6.33 | 24.85 ± 7.88 |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=2 | -10.37 ± 5.08 | -8.47 ± 7.30 | -10.14 ± 6.20 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | -2.64 ± 6.12 | — | — | 26.66 ± 11.15 | 26.88 ± 12.35 |
| 13 SFT + CFT | ✓ ep=2 | -3.56 ± 7.12 | -9.41 ± 8.31 | -2.88 ± 4.96 | 10.03 ± 12.44 | 28.15 ± 14.22 |

## gemma-2-9b-it × persona OOD (3 held-out: desire-to-create-allies, interest-in-music, interest-in-science)

Source: [persona_v1_v7_gemma-2-9b-it_ood_spearman_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-9b-it_ood_spearman_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -27.36 ± 5.49 | — | -22.55 ± 7.12 | — | 19.22 ± 2.66 |
| 1 SFT labelonly 10% | ✓ ep=1 | -10.99 ± 10.40 | -13.22 ± 7.94 | -10.02 ± 8.80 | 12.62 ± 9.70 | 54.84 ± 5.29 |
| 2 RankAlign | ✓ ep=2 | 37.56 ± 11.91 | 65.83 ± 2.63 | 51.99 ± 7.08 | 70.86 ± 1.19 | 74.54 ± 1.50 |
| 3 New + fsx [-TC] | ⏳ 42377 R 6:11:04 (≤2:48:56) | 0.41 ± 3.97 | 5.32 ± 0.75 | -1.82 ± 5.16 | 32.49 ± 6.03 | 59.33 ± 4.47 |
| 4 New + PMI + fsx | ✓ ep=1 | -8.18 ± 5.11 | -2.23 ± 5.18 | -10.53 ± 4.78 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | -4.30 ± 8.94 | 41.89 ± 7.15 | 15.28 ± 2.63 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | -1.23 ± 10.27 | 39.64 ± 8.61 | 0.34 ± 2.83 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | -7.78 ± 6.34 | — | — | 22.20 ± 7.85 | 54.36 ± 6.48 |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=1 | -2.91 ± 8.92 | 5.07 ± 7.89 | -5.27 ± 5.12 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | -4.63 ± 6.26 | — | — | 24.02 ± 8.34 | 53.69 ± 7.15 |
| 13 SFT + CFT | ✓ ep=2 | -19.61 ± 6.04 | -15.74 ± 7.27 | -18.56 ± 7.17 | 12.76 ± 6.62 | 43.37 ± 5.99 |

## gemma-2-9b-it × ifeval ID  **[v6 LEGACY]** (n=79 prompts ≥ 22; held-out 50% of completions for each)

> **⚠️ V6 LEGACY DATA — NOT v7.** All cells in this section come from
> a pre-fix1 v6 training run (`v6-google_gemma-2-9b-it-delta0.15-epoch2_ifeval-concat-all`). No v7 ifeval evaluations have been run yet (v7 × s13 trains in flight; v7 × s1–s9 not started). Treat these numbers as legacy reference, not as v7 results.

Source: [ifeval_id_spearman_table_cells.csv](metrics-from-scores/ifeval_id_spearman_table_cells.csv)

(no data)

## gemma-2-9b-it × ifeval OOD **[v6 LEGACY]** (n=20 fully held-out prompts: prompt_1..13, 15..21)

> **⚠️ V6 LEGACY DATA — NOT v7.** All cells in this section come from
> a pre-fix1 v6 training run (`v6-google_gemma-2-9b-it-delta0.15-epoch2_ifeval-concat-all`). No v7 ifeval evaluations have been run yet (v7 × s13 trains in flight; v7 × s1–s9 not started). Treat these numbers as legacy reference, not as v7 results.

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
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | – | — | — | — | — | — |
| 12 New + NegTC [-fsx] | – | — | — | — | — | — |
| 13 SFT + CFT | ⏳ 42114 R 15:50:04 (≤8:09:56) | — | — | — | — | — |
