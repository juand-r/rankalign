# humaneval-v2.1correct-multi × gemma-4-31B-it — v7 epoch2 tables

Snapshot: **2026-05-25 00:55 CT**

Model: gemma-4-31B-it, trained on humaneval-v2.1correct-multi-all, epoch 2.
Columns = scoring method at eval time (Raw / PMI base / Neg base etc.).
Rows = training method. Cells with `(n=N)` are partial — N < 82 problems scored.
`—` = no CSVs on disk yet. `---` = N/A per eval policy.

To refresh: `python scripts/_save_cm_v7_tables_md.py`

## GenROC × 100 (mean ± SE across problems)

*Humaneval-v2.1correct-multi GenROC × 100 — mean ± SE across 82 problems*

| Method | Raw | PMI self | PMI base | Neg self | Neg base |
| --- | --- | --- | --- | --- | --- |
| 0 Base | 42.79 ± 1.48 | 68.33 ± 1.32 | — | 65.24 ± 1.77 | — |
| 1 SFT labelonly 10% | 71.21 ± 1.32 | — | 89.95 ± 0.81 | — | 95.87 ± 0.50 |
| 2 RankAlign | 88.46 ± 0.95 | — | 90.57 ± 0.84 | — | 93.26 ± 0.83 |
| 3 New + fsx [-TC] | — | — | — | — | — |
| 4 New + PMI + fsx | 70.87 ± 1.31 | — | 86.90 ± 0.97 | --- | --- |
| 5 RA + PMI + fsx [-NLL] | — | — | — | --- | --- |
| 6 RA + PMI [+TC] | — | — | — | --- | --- |
| 11 New + PMI [-fsx] | — | — | — | --- | --- |
| 7 New + NegTC + fsx | 65.80 ± 1.95 (n=46) | --- | --- | — | 90.61 ± 1.08 (n=46) |
| 8 RA + NegTC + fsx [-NLL] | — | --- | --- | — | — |
| 9 RA + NegTC [+TC] | — | --- | --- | — | — |
| 12 New + NegTC [-fsx] | — | --- | --- | — | — |

## Pearson(gen, val) × 100

*Humaneval-v2.1correct-multi Pearson(gen, val) × 100 — mean ± SE across 82 problems*

| Method | Raw | PMI self | PMI base | Neg self | Neg base |
| --- | --- | --- | --- | --- | --- |
| 0 Base | 3.21 ± 2.34 | 24.11 ± 2.63 | — | 26.49 ± 2.64 | — |
| 1 SFT labelonly 10% | 31.38 ± 1.76 | — | 48.07 ± 2.19 | — | 59.05 ± 1.68 |
| 2 RankAlign | 68.42 ± 1.55 | — | 65.75 ± 1.81 | — | 71.37 ± 1.49 |
| 3 New + fsx [-TC] | — | — | — | — | — |
| 4 New + PMI + fsx | 47.12 ± 1.72 | — | 66.32 ± 1.74 | --- | --- |
| 5 RA + PMI + fsx [-NLL] | — | — | — | --- | --- |
| 6 RA + PMI [+TC] | — | — | — | --- | --- |
| 11 New + PMI [-fsx] | — | — | — | --- | --- |
| 7 New + NegTC + fsx | 41.03 ± 2.57 (n=46) | --- | --- | — | 75.01 ± 1.26 (n=46) |
| 8 RA + NegTC + fsx [-NLL] | — | --- | --- | — | — |
| 9 RA + NegTC [+TC] | — | --- | --- | — | — |
| 12 New + NegTC [-fsx] | — | --- | --- | — | — |

## Spearman(gen, val) × 100

*Humaneval-v2.1correct-multi Spearman(gen, val) × 100 — mean ± SE across 82 problems*

| Method | Raw | PMI self | PMI base | Neg self | Neg base |
| --- | --- | --- | --- | --- | --- |
| 0 Base | 0.53 ± 2.66 | 33.58 ± 2.43 | — | 26.20 ± 2.90 | — |
| 1 SFT labelonly 10% | 35.30 ± 2.23 | — | 57.08 ± 1.81 | — | 61.69 ± 1.74 |
| 2 RankAlign | 75.64 ± 1.48 | — | 67.16 ± 1.70 | — | 70.48 ± 1.56 |
| 3 New + fsx [-TC] | — | — | — | — | — |
| 4 New + PMI + fsx | 50.82 ± 2.14 | — | 69.57 ± 1.70 | --- | --- |
| 5 RA + PMI + fsx [-NLL] | — | — | — | --- | --- |
| 6 RA + PMI [+TC] | — | — | — | --- | --- |
| 11 New + PMI [-fsx] | — | — | — | --- | --- |
| 7 New + NegTC + fsx | 42.34 ± 3.19 (n=46) | --- | --- | — | 78.57 ± 1.37 (n=46) |
| 8 RA + NegTC + fsx [-NLL] | — | --- | --- | — | — |
| 9 RA + NegTC [+TC] | — | --- | --- | — | — |
| 12 New + NegTC [-fsx] | — | --- | --- | — | — |

## ValROC × 100

*Humaneval-v2.1correct-multi ValROC × 100 — mean ± SE across 82 problems*

| Method | Raw | PMI self | PMI base | Neg self | Neg base |
| --- | --- | --- | --- | --- | --- |
| 0 Base | 86.23 ± 1.15 | 86.23 ± 1.15 | — | 86.23 ± 1.15 | — |
| 1 SFT labelonly 10% | 88.52 ± 1.05 | — | 88.52 ± 1.05 | — | 88.52 ± 1.05 |
| 2 RankAlign | 91.20 ± 0.93 | — | 91.20 ± 0.93 | — | 91.20 ± 0.93 |
| 3 New + fsx [-TC] | — | — | — | — | — |
| 4 New + PMI + fsx | 92.50 ± 0.93 | — | 92.50 ± 0.93 | --- | --- |
| 5 RA + PMI + fsx [-NLL] | — | — | — | --- | --- |
| 6 RA + PMI [+TC] | — | — | — | --- | --- |
| 11 New + PMI [-fsx] | — | — | — | --- | --- |
| 7 New + NegTC + fsx | 93.74 ± 0.98 (n=46) | --- | --- | — | 93.74 ± 0.98 (n=46) |
| 8 RA + NegTC + fsx [-NLL] | — | --- | --- | — | — |
| 9 RA + NegTC [+TC] | — | --- | --- | — | — |
| 12 New + NegTC [-fsx] | — | --- | --- | — | — |

## ValAcc × 100

*Humaneval-v2.1correct-multi ValAcc × 100 — mean ± SE across 82 problems*

| Method | Raw | PMI self | PMI base | Neg self | Neg base |
| --- | --- | --- | --- | --- | --- |
| 0 Base | 72.29 ± 1.25 | 72.29 ± 1.25 | — | 72.29 ± 1.25 | — |
| 1 SFT labelonly 10% | 70.34 ± 1.20 | — | 70.34 ± 1.20 | — | 70.34 ± 1.20 |
| 2 RankAlign | 77.12 ± 1.34 | — | 77.12 ± 1.34 | — | 77.12 ± 1.34 |
| 3 New + fsx [-TC] | — | — | — | — | — |
| 4 New + PMI + fsx | 78.04 ± 1.24 | — | 78.04 ± 1.24 | --- | --- |
| 5 RA + PMI + fsx [-NLL] | — | — | — | --- | --- |
| 6 RA + PMI [+TC] | — | — | — | --- | --- |
| 11 New + PMI [-fsx] | — | — | — | --- | --- |
| 7 New + NegTC + fsx | 77.16 ± 1.59 (n=46) | --- | --- | — | 77.16 ± 1.59 (n=46) |
| 8 RA + NegTC + fsx [-NLL] | — | --- | --- | — | — |
| 9 RA + NegTC [+TC] | — | --- | --- | — | — |
| 12 New + NegTC [-fsx] | — | --- | --- | — | — |

