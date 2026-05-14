# membership->rosch mean across 10 rosch tasks (gemma-2-2b)

**Train task:** membership-sans-rosch-v0-all (165 categories, 2k items, 5110 pair samples).

**Eval tasks:** all 10 rosch categories (rosch-bird, rosch-carpenters-tool, rosch-clothing, rosch-fruit, rosch-furniture, rosch-sport, rosch-toy, rosch-vegetable, rosch-vehicle, rosch-weapon).

**Validator:** log-odds (`--validator-log-odds`). **Generator column variant:** `tc` (TC-corrected gen score where applicable).

**All numeric cells are raw values × 100** (percentage points for ROC/accuracy; 0–100 units for Pearson). Reported as **mean (std)** across the 10 rosch tasks.

Tables follow the canonical 4-table layout (see [docs/results_table_format.md](../../docs/results_table_format.md)): T1/T3 = baselines (Base HF + SFT) for the self/neg eval refs; T2/T4 = the 3×2 TC × pairs grids for self/neg. Cells in T2/T4 use the *best* eval ref per row (offline TC → `basetyp[neg]`; everything else → `self`/`neg`). The (offline TC, online pairs) cell is always blank because that variant is not in the launcher.

Long-form metrics: [quickiter_metrics_long_membership_to_rosch.csv](quickiter_metrics_long_membership_to_rosch.csv)

## All 10 rosch tasks  (n_tasks=10)

### Generator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 77.32 (5.55) |
| SFT (NLL all) | 82.96 (5.33) |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 86.37 (7.10) `[basetyp]` | — |
| online TC | 83.77 (7.94) | 81.98 (7.42) |
| no TC | 81.63 (8.14) (RankAlign) | 80.94 (9.30) |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 77.71 (12.69) |
| SFT (NLL all) | 74.86 (12.71) |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 76.26 (10.47) `[basetypneg]` | — |
| online TC | 84.93 (7.76) | 79.52 (9.87) |
| no TC | 82.96 (9.11) (RankAlign) | 75.60 (13.21) |

### Validator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 89.68 (8.17) |
| SFT (NLL all) | 89.39 (7.57) |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 89.80 (8.91) `[basetyp]` | — |
| online TC | 88.27 (8.22) | 89.72 (9.15) |
| no TC | 86.81 (10.42) (RankAlign) | 87.84 (9.82) |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 89.68 (8.17) |
| SFT (NLL all) | 89.39 (7.57) |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 87.46 (10.15) `[basetypneg]` | — |
| online TC | 89.53 (8.63) | 88.49 (9.12) |
| no TC | 86.81 (10.42) (RankAlign) | 87.84 (9.82) |

### Validator accuracy (thr 0) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 78.78 (10.71) |
| SFT (NLL all) | 52.17 (2.65) |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 80.33 (9.86) `[basetyp]` | — |
| online TC | 60.31 (9.07) | 81.57 (11.29) |
| no TC | 76.70 (11.09) (RankAlign) | 72.80 (11.81) |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 78.78 (10.71) |
| SFT (NLL all) | 52.17 (2.65) |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 75.12 (10.82) `[basetypneg]` | — |
| online TC | 75.59 (12.02) | 79.32 (9.67) |
| no TC | 76.70 (11.09) (RankAlign) | 72.80 (11.81) |

### Pearson(gen, validator) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 54.48 (10.09) |
| SFT (NLL all) | 60.68 (9.02) |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 71.26 (5.42) `[basetyp]` | — |
| online TC | 67.15 (9.72) | 68.47 (8.85) |
| no TC | 66.39 (7.63) (RankAlign) | 67.44 (8.91) |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 40.69 (15.28) |
| SFT (NLL all) | 47.44 (15.11) |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 55.18 (15.62) `[basetypneg]` | — |
| online TC | 78.53 (5.20) | 73.43 (9.14) |
| no TC | 54.78 (13.27) (RankAlign) | 41.76 (15.36) |
