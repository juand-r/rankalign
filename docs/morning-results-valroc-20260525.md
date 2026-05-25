# Morning Results — ValROC — 2026-05-25T15:53:30Z

All cells: **ValROC × 100 ± SE** (mean ± SE across the eval-task split for that section).

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

**Provenance caveat — ifeval sections are v6, not v7.** All other sections
(rosch, persona, humaneval) source from v7 (fix1) score files. The two ifeval
sections below are built by the legacy `_build_ifeval_ood_table.py` which
hardcodes a v6 model prefix; their cells reflect a pre-fix1 9b-it run trained
on ifeval-concat-all (delta0.15, epoch2). The v7 ifeval × s13 trains are still
in flight; no v7 ifeval evals have been launched for s1–s9 yet. Sections for
ifeval are labeled `[v6 legacy]` to make this explicit.

**Method-row coverage.** Tables include s1–s9 and s11–s13 (s10 was never run).
If a row shows all `—`, the eval CSV reports `n=0/missing` for that method;
ifeval and humaneval CSVs do not have an s13 row at all (the v6 builders
predate s13).

## gemma-2-2b × membership (eval = rosch, all 10 tasks)

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
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | – | — | — | — | — | — |
| 12 New + NegTC [-fsx] | – | — | — | — | — | — |
| 13 SFT + CFT | ✓ ep=2 | 88.26 ± 3.13 | 88.26 ± 3.13 | — | 88.26 ± 3.13 | — |

## gemma-2-2b-it × membership (eval = rosch, all 10 tasks)

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
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=2 | 89.35 ± 2.75 | 89.35 ± 2.75 | 89.35 ± 2.75 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 87.15 ± 2.62 | — | — | 87.15 ± 2.62 | 87.15 ± 2.62 |
| 13 SFT + CFT | ✓ ep=2 | 89.73 ± 3.05 | 89.73 ± 3.05 | 89.73 ± 3.05 | 89.73 ± 3.05 | 89.73 ± 3.05 |

## gemma-2-9b-it × membership (eval = rosch, all 10 tasks)

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
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=2 | 94.76 ± 1.92 | 94.76 ± 1.92 | 94.76 ± 1.92 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 94.69 ± 1.91 | — | — | 94.69 ± 1.91 | 94.69 ± 1.91 |
| 13 SFT + CFT | ✓ ep=2 | 94.72 ± 1.85 | 94.72 ± 1.85 | — | 94.72 ± 1.85 | — |

## gemma-2-2b × persona (all 6 personas)

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
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=1 | 96.17 ± 2.86 | 96.17 ± 2.86 | 96.17 ± 2.86 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 90.29 ± 5.98 | — | — | 90.29 ± 5.98 | — |
| 13 SFT + CFT | ✓ ep=2 | 86.20 ± 7.66 | 86.09 ± 7.70 | 86.20 ± 7.66 | 86.09 ± 7.70 | 86.20 ± 7.66 |

## gemma-2-2b-it × persona (all 6 personas)

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
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=2 | 95.72 ± 3.64 | 95.72 ± 3.64 | 94.81 ± 2.78 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 95.45 ± 3.57 | — | — | 95.45 ± 3.57 | 94.68 ± 2.62 |
| 13 SFT + CFT | ✓ ep=2 | 91.30 ± 4.64 | 90.95 ± 4.81 | 91.30 ± 4.64 | 90.95 ± 4.81 | 91.30 ± 4.64 |

## gemma-2-9b-it × persona (all 6 personas)

Source: [persona_v1_v7_gemma-2-9b-it_all_val_roc_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-9b-it_all_val_roc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 95.59 ± 1.49 | — | 95.59 ± 1.49 | — | 95.59 ± 1.49 |
| 1 SFT labelonly 10% | ✓ ep=1 | 98.07 ± 1.85 | 98.92 ± 1.08 | 98.07 ± 1.85 | 98.92 ± 1.08 | 98.07 ± 1.85 |
| 2 RankAlign | ✓ ep=2 | 99.35 ± 0.43 | 99.35 ± 0.43 | 99.35 ± 0.43 | 99.35 ± 0.43 | 99.35 ± 0.43 |
| 3 New + fsx [-TC] | ⏳ 42377 R 6:06:19 (≤2:53:41) | 96.92 ± 2.97 | 96.92 ± 2.97 | 96.92 ± 2.97 | 96.92 ± 2.97 | 96.92 ± 2.97 |
| 4 New + PMI + fsx | ✓ ep=1 | 96.74 ± 3.21 | 96.74 ± 3.21 | 96.74 ± 3.21 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 99.41 ± 0.33 | 99.41 ± 0.33 | 99.41 ± 0.33 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 99.40 ± 0.36 | 99.40 ± 0.36 | 99.40 ± 0.36 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 97.76 ± 2.22 | — | — | 97.76 ± 2.22 | 97.76 ± 2.22 |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=1 | 98.03 ± 1.96 | 98.03 ± 1.96 | 97.66 ± 2.32 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 98.16 ± 1.84 | — | — | 98.16 ± 1.84 | 97.85 ± 2.10 |
| 13 SFT + CFT | ✓ ep=2 | 98.88 ± 1.05 | 98.94 ± 1.00 | 98.88 ± 1.05 | 98.94 ± 1.00 | 98.88 ± 1.05 |

## gemma-2-2b × persona ID (3 in-domain: psychopathy, machiavellianism, narcissism)

Source: [persona_v1_v7_gemma-2-2b_id_val_roc_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-2b_id_val_roc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 93.87 ± 5.81 | — | 93.87 ± 5.81 | — | 93.87 ± 5.81 |
| 1 SFT labelonly 10% | – | — | — | — | — | — |
| 2 RankAlign | ✓ ep=1 | — | — | — | — | — |
| 3 New + fsx [-TC] | – | — | — | — | — | — |
| 4 New + PMI + fsx | – | — | — | — | — | — |
| 5 RA + PMI + fsx [-NLL] | – | — | — | — | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | – | — | — | — | — | — |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=1 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 99.99 ± 0.01 | — | — | 99.99 ± 0.01 | — |
| 13 SFT + CFT | ✓ ep=2 | 99.85 ± 0.03 | 99.86 ± 0.03 | 99.85 ± 0.03 | 99.86 ± 0.03 | 99.85 ± 0.03 |

## gemma-2-2b-it × persona ID (3 in-domain: psychopathy, machiavellianism, narcissism)

Source: [persona_v1_v7_gemma-2-2b-it_id_val_roc_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-2b-it_id_val_roc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 94.75 ± 2.01 | — | 94.75 ± 2.01 | — | 94.75 ± 2.01 |
| 1 SFT labelonly 10% | ✓ ep=2 | 99.99 ± 0.01 | 99.99 ± 0.01 | 99.99 ± 0.01 | 99.99 ± 0.01 | 99.99 ± 0.01 |
| 2 RankAlign | ✓ ep=2 | 96.50 ± 0.73 | 96.50 ± 0.73 | 96.50 ± 0.73 | 96.50 ± 0.73 | 96.50 ± 0.73 |
| 3 New + fsx [-TC] | ✓ ep=2 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| 4 New + PMI + fsx | ✓ ep=2 | 99.95 ± 0.05 | 99.95 ± 0.05 | 99.95 ± 0.05 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 98.61 ± 0.19 | 98.61 ± 0.19 | 98.61 ± 0.19 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 97.83 ± 1.66 | 97.83 ± 1.66 | 97.83 ± 1.66 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 100.00 ± 0.00 | — | — | 100.00 ± 0.00 | 100.00 ± 0.00 |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=2 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 99.99 ± 0.01 | — | — | 99.99 ± 0.01 | 99.97 ± 0.01 |
| 13 SFT + CFT | ✓ ep=2 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |

## gemma-2-9b-it × persona ID (3 in-domain: psychopathy, machiavellianism, narcissism)

Source: [persona_v1_v7_gemma-2-9b-it_id_val_roc_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-9b-it_id_val_roc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 93.75 ± 1.64 | — | 93.75 ± 1.64 | — | 93.75 ± 1.64 |
| 1 SFT labelonly 10% | ✓ ep=1 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| 2 RankAlign | ✓ ep=2 | 99.20 ± 0.80 | 99.20 ± 0.80 | 99.20 ± 0.80 | 99.20 ± 0.80 | 99.20 ± 0.80 |
| 3 New + fsx [-TC] | ⏳ 42377 R 6:06:19 (≤2:53:41) | 99.98 ± 0.02 | 99.98 ± 0.02 | 99.98 ± 0.02 | 99.98 ± 0.02 | 99.98 ± 0.02 |
| 4 New + PMI + fsx | ✓ ep=1 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 99.45 ± 0.40 | 99.45 ± 0.40 | 99.45 ± 0.40 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 99.41 ± 0.55 | 99.41 ± 0.55 | 99.41 ± 0.55 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 100.00 ± 0.00 | — | — | 100.00 ± 0.00 | 100.00 ± 0.00 |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=1 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 100.00 ± 0.00 | — | — | 100.00 ± 0.00 | 100.00 ± 0.00 |
| 13 SFT + CFT | ✓ ep=2 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |

## gemma-2-2b × persona OOD (3 held-out: desire-to-create-allies, interest-in-music, interest-in-science)

Source: [persona_v1_v7_gemma-2-2b_ood_val_roc_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-2b_ood_val_roc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 99.55 ± 0.44 | — | 99.55 ± 0.44 | — | 99.55 ± 0.44 |
| 1 SFT labelonly 10% | – | — | — | — | — | — |
| 2 RankAlign | ✓ ep=1 | — | — | — | — | — |
| 3 New + fsx [-TC] | – | — | — | — | — | — |
| 4 New + PMI + fsx | – | — | — | — | — | — |
| 5 RA + PMI + fsx [-NLL] | – | — | — | — | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | – | — | — | — | — | — |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=1 | 92.34 ± 5.13 | 92.34 ± 5.13 | 92.34 ± 5.13 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 80.59 ± 9.21 | — | — | 80.59 ± 9.21 | — |
| 13 SFT + CFT | ✓ ep=2 | 72.54 ± 10.36 | 72.33 ± 10.35 | 72.54 ± 10.36 | 72.33 ± 10.35 | 72.54 ± 10.36 |

## gemma-2-2b-it × persona OOD (3 held-out: desire-to-create-allies, interest-in-music, interest-in-science)

Source: [persona_v1_v7_gemma-2-2b-it_ood_val_roc_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-2b-it_ood_val_roc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 99.15 ± 0.43 | — | 99.15 ± 0.43 | — | 99.15 ± 0.43 |
| 1 SFT labelonly 10% | ✓ ep=2 | 83.47 ± 3.56 | 83.47 ± 3.56 | 83.47 ± 3.56 | 83.47 ± 3.56 | 83.47 ± 3.56 |
| 2 RankAlign | ✓ ep=2 | 87.65 ± 10.38 | 87.65 ± 10.38 | 87.65 ± 10.38 | 87.65 ± 10.38 | 87.65 ± 10.38 |
| 3 New + fsx [-TC] | ✓ ep=2 | 89.28 ± 4.35 | 89.28 ± 4.35 | 89.28 ± 4.35 | 89.28 ± 4.35 | 89.28 ± 4.35 |
| 4 New + PMI + fsx | ✓ ep=2 | 94.61 ± 3.18 | 94.61 ± 3.18 | 94.61 ± 3.18 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 97.15 ± 2.58 | 97.15 ± 2.58 | 97.15 ± 2.58 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 99.35 ± 0.62 | 99.35 ± 0.62 | 99.35 ± 0.62 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 83.84 ± 3.58 | — | — | 83.84 ± 3.58 | 83.84 ± 3.58 |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=2 | 91.45 ± 6.91 | 91.45 ± 6.91 | 89.61 ± 3.41 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=2 | 90.91 ± 6.57 | — | — | 90.91 ± 6.57 | 89.40 ± 2.50 |
| 13 SFT + CFT | ✓ ep=2 | 82.61 ± 5.67 | 81.91 ± 5.83 | 82.61 ± 5.67 | 81.91 ± 5.83 | 82.61 ± 5.67 |

## gemma-2-9b-it × persona OOD (3 held-out: desire-to-create-allies, interest-in-music, interest-in-science)

Source: [persona_v1_v7_gemma-2-9b-it_ood_val_roc_table_cells.csv](metrics-from-scores/persona_v1_v7_gemma-2-9b-it_ood_val_roc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 97.44 ± 2.24 | — | 97.44 ± 2.24 | — | 97.44 ± 2.24 |
| 1 SFT labelonly 10% | ✓ ep=1 | 96.14 ± 3.66 | 97.84 ± 2.15 | 96.14 ± 3.66 | 97.84 ± 2.15 | 96.14 ± 3.66 |
| 2 RankAlign | ✓ ep=2 | 99.49 ± 0.51 | 99.49 ± 0.51 | 99.49 ± 0.51 | 99.49 ± 0.51 | 99.49 ± 0.51 |
| 3 New + fsx [-TC] | ⏳ 42377 R 6:06:19 (≤2:53:41) | 93.86 ± 5.90 | 93.86 ± 5.90 | 93.86 ± 5.90 | 93.86 ± 5.90 | 93.86 ± 5.90 |
| 4 New + PMI + fsx | ✓ ep=1 | 93.48 ± 6.38 | 93.48 ± 6.38 | 93.48 ± 6.38 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ ep=2 | 99.38 ± 0.61 | 99.38 ± 0.61 | 99.38 ± 0.61 | — | — |
| 6 RA + PMI [+TC] | ✓ ep=2 | 99.39 ± 0.59 | 99.39 ± 0.59 | 99.39 ± 0.59 | — | — |
| 7 New + NegTC + fsx | ✓ ep=2 | 95.52 ± 4.44 | — | — | 95.52 ± 4.44 | 95.52 ± 4.44 |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | ✓ ep=1 | 96.07 ± 3.92 | 96.07 ± 3.92 | 95.33 ± 4.63 | — | — |
| 12 New + NegTC [-fsx] | ✓ ep=1 | 96.32 ± 3.68 | — | — | 96.32 ± 3.68 | 95.69 ± 4.16 |
| 13 SFT + CFT | ✓ ep=2 | 97.77 ± 2.06 | 97.89 ± 1.96 | 97.77 ± 2.06 | 97.89 ± 1.96 | 97.77 ± 2.06 |

## gemma-2-9b-it × ifeval ID  [v6 legacy] (n=79 prompts ≥ 22; held-out 50% of completions for each)

Source: [ifeval_id_val_roc_table_cells.csv](metrics-from-scores/ifeval_id_val_roc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 84.04 ± 1.59 | — | 84.04 ± 1.59 | — | 84.04 ± 1.59 |
| 1 SFT labelonly 10% | ✓ (prior run) | 82.78 ± 1.65 | 82.78 ± 1.65 | 82.78 ± 1.65 | — | — |
| 2 RankAlign | ✓ (prior run) | 81.53 ± 1.75 | — | 81.53 ± 1.75 | — | 81.53 ± 1.75 |
| 3 New + fsx [-TC] | ✓ (prior run) | 84.51 ± 1.48 | 84.51 ± 1.48 | 84.51 ± 1.48 | — | 84.51 ± 1.48 |
| 4 New + PMI + fsx | ✓ (prior run) | 82.53 ± 1.57 | 82.53 ± 1.57 | 82.53 ± 1.57 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ (prior run) | 81.95 ± 1.60 | 81.95 ± 1.60 | 81.95 ± 1.60 | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | ✓ (prior run) | 84.65 ± 1.54 | — | — | — | 84.65 ± 1.54 |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | – | — | — | — | — | — |
| 12 New + NegTC [-fsx] | – | — | — | — | — | — |
| 13 SFT + CFT | ⏳ 42343 R 7:51:23 (≤6:08:37) | — | — | — | — | — |

## gemma-2-9b-it × ifeval OOD [v6 legacy] (n=20 fully held-out prompts: prompt_1..13, 15..21)

Source: [ifeval_ood_val_roc_table_cells.csv](metrics-from-scores/ifeval_ood_val_roc_table_cells.csv)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | 78.41 ± 3.47 | — | 78.41 ± 3.47 | — | 78.41 ± 3.47 |
| 1 SFT labelonly 10% | ✓ (prior run) | 77.25 ± 3.40 | 77.25 ± 3.40 | 77.25 ± 3.40 | — | — |
| 2 RankAlign | ✓ (prior run) | 74.50 ± 3.79 | — | 74.50 ± 3.79 | — | 74.50 ± 3.79 |
| 3 New + fsx [-TC] | ✓ (prior run) | 79.80 ± 3.42 | 79.80 ± 3.42 | 79.80 ± 3.42 | — | 79.80 ± 3.42 |
| 4 New + PMI + fsx | ✓ (prior run) | 79.58 ± 3.24 | 79.58 ± 3.24 | 79.58 ± 3.24 | — | — |
| 5 RA + PMI + fsx [-NLL] | ✓ (prior run) | 76.34 ± 3.55 | 76.34 ± 3.55 | 76.34 ± 3.55 | — | — |
| 6 RA + PMI [+TC] | – | — | — | — | — | — |
| 7 New + NegTC + fsx | ✓ (prior run) | 79.21 ± 3.07 | — | — | — | 79.21 ± 3.07 |
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | – | — | — | — | — | — |
| 12 New + NegTC [-fsx] | – | — | — | — | — | — |
| 13 SFT + CFT | ⏳ 42343 R 7:51:23 (≤6:08:37) | — | — | — | — | — |

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
| 8 RA + NegTC + fsx [-NLL] | – | — | — | — | — | — |
| 9 RA + NegTC [+TC] | – | — | — | — | — | — |
| 11 New + PMI [-fsx] | – | — | — | — | — | — |
| 12 New + NegTC [-fsx] | – | — | — | — | — | — |
| 13 SFT + CFT | ⏳ 42114 R 15:45:19 (≤8:14:41) | — | — | — | — | — |
