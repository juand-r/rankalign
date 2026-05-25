# humaneval-v2.1correct-upper × gemma-4-31B-it — v7 epoch2 tables

Snapshot: **2026-05-24 23:53 CT**

Model: gemma-4-31B-it, trained on humaneval-v2.1correct-upper-all, epoch 2.
Columns = scoring method at eval time (Raw / PMI base / Neg base etc.).
Rows = training method. Cells with `(n=N)` are partial — N < 82 problems scored.
`—` = no CSVs on disk yet. `---` = N/A per eval policy.

To refresh: `python scripts/_save_cu_v7_tables_md.py`

## GenROC × 100 (mean ± SE across problems)

*Humaneval-v2.1correct-upper GenROC × 100 — mean ± SE across 82 problems*

| Method | Raw | PMI self | PMI base | Neg self | Neg base |
| --- | --- | --- | --- | --- | --- |
| 0 Base | — | — | — | — | — |
| 1 SFT labelonly 10% | — | — | — | — | — |
| 2 RankAlign | — | — | — | — | — |
| 3 New + fsx [-TC] | 91.11 ± 0.96 | — | 86.09 ± 1.18 | — | 90.43 ± 2.31 (n=12) |
| 4 New + PMI + fsx | 88.30 ± 0.99 (n=81) | — | 92.16 ± 0.77 (n=81) | --- | --- |
| 5 RA + PMI + fsx [-NLL] | — | — | — | --- | --- |
| 6 RA + PMI [+TC] | — | — | — | --- | --- |
| 11 New + PMI [-fsx] | — | — | — | --- | --- |
| 7 New + NegTC + fsx | 83.14 ± 1.09 (n=81) | --- | --- | — | 94.16 ± 0.91 (n=81) |
| 8 RA + NegTC + fsx [-NLL] | — | --- | --- | — | — |
| 9 RA + NegTC [+TC] | — | --- | --- | — | — |
| 12 New + NegTC [-fsx] | — | --- | --- | — | — |

## Pearson(gen, val) × 100

*Humaneval-v2.1correct-upper Pearson(gen, val) × 100 — mean ± SE across 82 problems*

| Method | Raw | PMI self | PMI base | Neg self | Neg base |
| --- | --- | --- | --- | --- | --- |
| 0 Base | — | — | — | — | — |
| 1 SFT labelonly 10% | — | — | — | — | — |
| 2 RankAlign | — | — | — | — | — |
| 3 New + fsx [-TC] | 65.35 ± 1.72 | — | 45.50 ± 3.07 | — | 61.79 ± 5.17 (n=12) |
| 4 New + PMI + fsx | 61.59 ± 1.69 (n=81) | — | 66.76 ± 1.75 (n=81) | --- | --- |
| 5 RA + PMI + fsx [-NLL] | — | — | — | --- | --- |
| 6 RA + PMI [+TC] | — | — | — | --- | --- |
| 11 New + PMI [-fsx] | — | — | — | --- | --- |
| 7 New + NegTC + fsx | 58.53 ± 1.66 (n=81) | --- | --- | — | 76.26 ± 1.31 (n=81) |
| 8 RA + NegTC + fsx [-NLL] | — | --- | --- | — | — |
| 9 RA + NegTC [+TC] | — | --- | --- | — | — |
| 12 New + NegTC [-fsx] | — | --- | --- | — | — |

## Spearman(gen, val) × 100

*Humaneval-v2.1correct-upper Spearman(gen, val) × 100 — mean ± SE across 82 problems*

| Method | Raw | PMI self | PMI base | Neg self | Neg base |
| --- | --- | --- | --- | --- | --- |
| 0 Base | — | — | — | — | — |
| 1 SFT labelonly 10% | — | — | — | — | — |
| 2 RankAlign | — | — | — | — | — |
| 3 New + fsx [-TC] | 68.87 ± 1.84 | — | 51.29 ± 2.59 | — | 57.05 ± 4.86 (n=12) |
| 4 New + PMI + fsx | 67.02 ± 1.78 (n=81) | — | 67.85 ± 1.66 (n=81) | --- | --- |
| 5 RA + PMI + fsx [-NLL] | — | — | — | --- | --- |
| 6 RA + PMI [+TC] | — | — | — | --- | --- |
| 11 New + PMI [-fsx] | — | — | — | --- | --- |
| 7 New + NegTC + fsx | 63.96 ± 1.89 (n=81) | --- | --- | — | 77.05 ± 1.13 (n=81) |
| 8 RA + NegTC + fsx [-NLL] | — | --- | --- | — | — |
| 9 RA + NegTC [+TC] | — | --- | --- | — | — |
| 12 New + NegTC [-fsx] | — | --- | --- | — | — |

## ValROC × 100

*Humaneval-v2.1correct-upper ValROC × 100 — mean ± SE across 82 problems*

| Method | Raw | PMI self | PMI base | Neg self | Neg base |
| --- | --- | --- | --- | --- | --- |
| 0 Base | — | — | — | — | — |
| 1 SFT labelonly 10% | — | — | — | — | — |
| 2 RankAlign | — | — | — | — | — |
| 3 New + fsx [-TC] | 93.73 ± 0.86 | — | 93.73 ± 0.86 | — | 95.41 ± 1.59 (n=12) |
| 4 New + PMI + fsx | 93.34 ± 0.90 (n=81) | — | 93.34 ± 0.90 (n=81) | --- | --- |
| 5 RA + PMI + fsx [-NLL] | — | — | — | --- | --- |
| 6 RA + PMI [+TC] | — | — | — | --- | --- |
| 11 New + PMI [-fsx] | — | — | — | --- | --- |
| 7 New + NegTC + fsx | 93.22 ± 0.93 (n=81) | --- | --- | — | 93.22 ± 0.93 (n=81) |
| 8 RA + NegTC + fsx [-NLL] | — | --- | --- | — | — |
| 9 RA + NegTC [+TC] | — | --- | --- | — | — |
| 12 New + NegTC [-fsx] | — | --- | --- | — | — |

## ValAcc × 100

*Humaneval-v2.1correct-upper ValAcc × 100 — mean ± SE across 82 problems*

| Method | Raw | PMI self | PMI base | Neg self | Neg base |
| --- | --- | --- | --- | --- | --- |
| 0 Base | — | — | — | — | — |
| 1 SFT labelonly 10% | — | — | — | — | — |
| 2 RankAlign | — | — | — | — | — |
| 3 New + fsx [-TC] | 85.67 ± 1.33 | — | 85.67 ± 1.33 | — | 89.37 ± 3.39 (n=12) |
| 4 New + PMI + fsx | 85.84 ± 1.43 (n=81) | — | 85.84 ± 1.43 (n=81) | --- | --- |
| 5 RA + PMI + fsx [-NLL] | — | — | — | --- | --- |
| 6 RA + PMI [+TC] | — | — | — | --- | --- |
| 11 New + PMI [-fsx] | — | — | — | --- | --- |
| 7 New + NegTC + fsx | 83.67 ± 1.34 (n=81) | --- | --- | — | 83.67 ± 1.34 (n=81) |
| 8 RA + NegTC + fsx [-NLL] | — | --- | --- | — | — |
| 9 RA + NegTC [+TC] | — | --- | --- | — | — |
| 12 New + NegTC [-fsx] | — | --- | --- | — | — |

