# membership->rosch stratified by item-overlap (gemma-2-2b)

**Train task:** membership-sans-rosch-v0-all. **Validator:** log-odds. **Generator column:** `tc`. **Values × 100.**

Buckets defined by what fraction of each rosch task's positive items appear in the membership training pool (pos+neg). High-overlap buckets are closer to a memorization probe; low-overlap buckets test genuine OOD generalization.

Item-overlap fractions: rosch-bird 89%, rosch-carpenters-tool 61%, rosch-fruit 60%, rosch-vehicle 56%, rosch-furniture 45%, rosch-vegetable 44%, rosch-toy 42%, rosch-clothing 36%, rosch-weapon 36%, rosch-sport 9%.

Tables follow the canonical 4-table layout (see [docs/results_table_format.md](../../docs/results_table_format.md)): T1/T3 = baselines (Base HF + SFT) for the self/neg eval refs; T2/T4 = the 3×2 TC × pairs grids for self/neg. Cells in T2/T4 use the *best* eval ref per row (offline TC → `basetyp[neg]`; everything else → `self`/`neg`). The (offline TC, online pairs) cell is always blank because that variant is not in the launcher.

Long-form metrics: [quickiter_metrics_long_membership_to_rosch.csv](quickiter_metrics_long_membership_to_rosch.csv)

## High overlap (>=60%)  (n_tasks=3)

### Generator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 72.72 (3.49) |
| SFT+fsx (NLL all) | 81.48 (7.86) |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 84.13 (9.71) `[basetyp]` | — |
| online TC | 81.62 (9.81) | 80.28 (11.03) |
| no TC | 76.33 (11.10) (RankAlign+fsx) | 74.01 (13.25) |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 72.97 (12.76) |
| SFT+fsx (NLL all) | 77.47 (10.90) |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 71.57 (8.82) `[basetypneg]` | — |
| online TC | 84.52 (11.34) | 76.01 (13.62) |
| no TC | 81.52 (12.54) (RankAlign+fsx) | 76.89 (11.58) |

### Validator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 87.24 (14.22) |
| SFT+fsx (NLL all) | 87.87 (12.01) |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 86.75 (15.28) `[basetyp]` | — |
| online TC | 86.32 (14.21) | 86.81 (15.31) |
| no TC | 86.88 (15.76) (RankAlign+fsx) | 85.76 (17.12) |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 87.24 (14.22) |
| SFT+fsx (NLL all) | 87.87 (12.01) |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 85.24 (18.21) `[basetypneg]` | — |
| online TC | 87.98 (14.32) | 87.56 (14.86) |
| no TC | 86.88 (15.76) (RankAlign+fsx) | 85.76 (17.12) |

### Validator accuracy (thr 0) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 78.33 (19.01) |
| SFT+fsx (NLL all) | 52.99 (3.34) |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 81.74 (14.03) `[basetyp]` | — |
| online TC | 62.93 (14.57) | 79.46 (18.08) |
| no TC | 78.68 (16.54) (RankAlign+fsx) | 77.06 (18.41) |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 78.33 (19.01) |
| SFT+fsx (NLL all) | 52.99 (3.34) |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 76.83 (18.22) `[basetypneg]` | — |
| online TC | 77.54 (18.75) | 81.06 (17.03) |
| no TC | 78.68 (16.54) (RankAlign+fsx) | 77.06 (18.41) |

### Pearson(gen, validator) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 58.51 (12.33) |
| SFT+fsx (NLL all) | 61.59 (11.98) |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 71.30 (5.12) `[basetyp]` | — |
| online TC | 69.88 (7.05) | 68.19 (7.89) |
| no TC | 66.72 (9.74) (RankAlign+fsx) | 63.76 (11.49) |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 35.08 (16.50) |
| SFT+fsx (NLL all) | 43.69 (19.12) |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 43.14 (9.85) `[basetypneg]` | — |
| online TC | 77.71 (7.43) | 67.77 (15.12) |
| no TC | 51.58 (12.47) (RankAlign+fsx) | 48.68 (7.97) |

## Mid overlap (35-55%)  (n_tasks=6)

### Generator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 79.10 (5.67) |
| SFT+fsx (NLL all) | 83.84 (4.90) |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 86.79 (6.81) `[basetyp]` | — |
| online TC | 83.95 (8.14) | 82.35 (6.85) |
| no TC | 83.57 (6.70) (RankAlign+fsx) | 83.91 (6.64) |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 79.92 (14.33) |
| SFT+fsx (NLL all) | 73.76 (15.40) |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 77.78 (12.05) `[basetypneg]` | — |
| online TC | 84.34 (7.20) | 79.79 (8.80) |
| no TC | 84.78 (8.52) (RankAlign+fsx) | 75.52 (16.04) |

### Validator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 91.46 (5.44) |
| SFT+fsx (NLL all) | 90.33 (6.54) |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 91.33 (6.41) `[basetyp]` | — |
| online TC | 88.93 (6.10) | 91.09 (7.03) |
| no TC | 88.39 (8.64) (RankAlign+fsx) | 89.55 (6.86) |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 91.46 (5.44) |
| SFT+fsx (NLL all) | 90.33 (6.54) |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 89.46 (6.28) `[basetypneg]` | — |
| online TC | 90.13 (7.07) | 89.76 (7.35) |
| no TC | 88.39 (8.64) (RankAlign+fsx) | 89.55 (6.86) |

### Validator accuracy (thr 0) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 79.27 (7.81) |
| SFT+fsx (NLL all) | 51.76 (2.75) |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 81.41 (8.40) `[basetyp]` | — |
| online TC | 57.64 (6.11) | 82.63 (9.73) |
| no TC | 77.63 (9.07) (RankAlign+fsx) | 73.57 (6.61) |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 79.27 (7.81) |
| SFT+fsx (NLL all) | 51.76 (2.75) |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 75.02 (8.49) `[basetypneg]` | — |
| online TC | 72.90 (9.36) | 79.35 (6.68) |
| no TC | 77.63 (9.07) (RankAlign+fsx) | 73.57 (6.61) |

### Pearson(gen, validator) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 52.39 (10.37) |
| SFT+fsx (NLL all) | 60.30 (9.40) |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 70.78 (6.38) `[basetyp]` | — |
| online TC | 64.77 (11.48) | 67.72 (10.46) |
| no TC | 67.29 (7.59) (RankAlign+fsx) | 71.18 (6.26) |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 43.85 (16.73) |
| SFT+fsx (NLL all) | 48.09 (15.66) |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 62.09 (15.82) `[basetypneg]` | — |
| online TC | 78.20 (4.70) | 75.38 (5.45) |
| no TC | 57.04 (15.46) (RankAlign+fsx) | 40.79 (18.03) |

## Clean (<10%)  (n_tasks=1)

### Generator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 80.41 |
| SFT+fsx (NLL all) | 82.07 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 90.57 `[basetyp]` | — |
| online TC | 89.15 | 84.85 |
| no TC | 85.92 (RankAlign+fsx) | 83.93 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 78.69 |
| SFT+fsx (NLL all) | 73.68 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 81.21 `[basetypneg]` | — |
| online TC | 89.70 | 88.49 |
| no TC | 76.39 (RankAlign+fsx) | 72.19 |

### Validator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 86.29 |
| SFT+fsx (NLL all) | 88.26 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 89.74 `[basetyp]` | — |
| online TC | 90.17 | 90.26 |
| no TC | 77.17 (RankAlign+fsx) | 83.84 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 86.29 |
| SFT+fsx (NLL all) | 88.26 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 82.09 `[basetypneg]` | — |
| online TC | 90.64 | 83.60 |
| no TC | 77.17 (RankAlign+fsx) | 83.84 |

### Validator accuracy (thr 0) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 77.17 |
| SFT+fsx (NLL all) | 52.17 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 69.57 `[basetyp]` | — |
| online TC | 68.48 | 81.52 |
| no TC | 65.22 (RankAlign+fsx) | 55.43 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 77.17 |
| SFT+fsx (NLL all) | 52.17 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 70.65 `[basetypneg]` | — |
| online TC | 85.87 | 73.91 |
| no TC | 65.22 (RankAlign+fsx) | 55.43 |

### Pearson(gen, validator) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 54.98 |
| SFT+fsx (NLL all) | 60.16 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 73.95 `[basetyp]` | — |
| online TC | 73.20 | 73.81 |
| no TC | 59.97 (RankAlign+fsx) | 56.03 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 38.56 |
| SFT+fsx (NLL all) | 54.75 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 49.80 `[basetypneg]` | — |
| online TC | 83.00 | 78.68 |
| no TC | 50.78 (RankAlign+fsx) | 26.82 |
