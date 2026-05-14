# Cross-task quick-iter (gemma-2-2b) — trained on membership-sans-rosch-v0, evaluated on **rosch-furniture**

**Train task:** membership-sans-rosch-v0 (model checkpoints). **Eval task:** rosch-furniture (80 items, 40/40 yes/no). **Validator:** log-odds (`--validator-log-odds`). **Metrics from `summarize_scores_file.py`, generator column variant `tc`** (TC-corrected gen score where applicable).

Models were trained on a *different* rosch category set (furniture+bird) and are evaluated here as out-of-distribution. This IS a generalization read.

**All numeric cells are raw values × 100** (i.e. ROC-AUC and accuracy are in percentage points; Pearson is in 0–100 units).

Tables follow the canonical 4-table layout (see [docs/results_table_format.md](../docs/results_table_format.md)): T1/T3 list base + SFT for the self/neg eval refs; T2/T4 are the 3×2 TC × pairs grids. Cells in T2/T4 use the *best* eval ref per row (offline TC → `basetyp[neg]`; everything else → `self`/`neg`). The (offline TC, online pairs) cell is always blank because that variant is not in the launcher.

Long-form metrics: [quickiter_metrics_long_membership_to_rosch.csv](quickiter_metrics_long_membership_to_rosch.csv)

### Generator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 80.19 |
| SFT (NLL all) | 89.34 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 93.25 `[basetyp]` | — |
| online TC | 91.94 | 88.62 |
| no TC | 87.63 (RankAlign) | 90.34 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 93.00 |
| SFT (NLL all) | 94.03 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 83.44 `[basetypneg]` | — |
| online TC | 88.59 | 82.03 |
| no TC | 97.66 (RankAlign) | 94.62 |

### Validator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 91.06 |
| SFT (NLL all) | 93.50 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 92.62 `[basetyp]` | — |
| online TC | 88.63 | 90.62 |
| no TC | 92.31 (RankAlign) | 90.38 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 91.06 |
| SFT (NLL all) | 93.50 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 95.12 `[basetypneg]` | — |
| online TC | 89.94 | 89.31 |
| no TC | 92.31 (RankAlign) | 90.38 |

### Validator accuracy (thr 0) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 78.75 |
| SFT (NLL all) | 50.00 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 82.50 `[basetyp]` | — |
| online TC | 68.75 | 83.75 |
| no TC | 76.25 (RankAlign) | 80.00 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 78.75 |
| SFT (NLL all) | 50.00 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 85.00 `[basetypneg]` | — |
| online TC | 72.50 | 80.00 |
| no TC | 76.25 (RankAlign) | 80.00 |

### Pearson(gen, validator) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 62.72 |
| SFT (NLL all) | 66.66 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 75.88 `[basetyp]` | — |
| online TC | 80.34 | 80.11 |
| no TC | 70.67 (RankAlign) | 76.14 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 60.60 |
| SFT (NLL all) | 70.86 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 64.58 `[basetypneg]` | — |
| online TC | 84.54 | 79.20 |
| no TC | 74.69 (RankAlign) | 58.55 |
