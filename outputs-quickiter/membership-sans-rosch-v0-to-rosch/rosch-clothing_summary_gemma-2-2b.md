# Cross-task quick-iter (gemma-2-2b) — trained on membership-sans-rosch-v0, evaluated on **rosch-clothing**

**Train task:** membership-sans-rosch-v0 (model checkpoints). **Eval task:** rosch-clothing (88 items, 44/44 yes/no). **Validator:** log-odds (`--validator-log-odds`). **Metrics from `summarize_scores_file.py`, generator column variant `tc`** (TC-corrected gen score where applicable).

Models were trained on a *different* rosch category set (furniture+bird) and are evaluated here as out-of-distribution. This IS a generalization read.

**All numeric cells are raw values × 100** (i.e. ROC-AUC and accuracy are in percentage points; Pearson is in 0–100 units).

Tables follow the canonical 4-table layout (see [docs/results_table_format.md](../docs/results_table_format.md)): T1/T3 list base + SFT for the self/neg eval refs; T2/T4 are the 3×2 TC × pairs grids. Cells in T2/T4 use the *best* eval ref per row (offline TC → `basetyp[neg]`; everything else → `self`/`neg`). The (offline TC, online pairs) cell is always blank because that variant is not in the launcher.

Long-form metrics: [quickiter_metrics_long_membership_to_rosch.csv](quickiter_metrics_long_membership_to_rosch.csv)

### Generator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 69.06 |
| SFT (NLL all) | 88.71 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 90.83 `[basetyp]` | — |
| online TC | 90.88 | 89.95 |
| no TC | 88.15 (RankAlign) | 85.49 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 54.83 |
| SFT (NLL all) | 66.71 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 89.57 `[basetypneg]` | — |
| online TC | 92.79 | 84.58 |
| no TC | 74.66 (RankAlign) | 58.63 |

### Validator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 94.37 |
| SFT (NLL all) | 92.69 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 94.99 `[basetyp]` | — |
| online TC | 91.68 | 95.76 |
| no TC | 92.92 (RankAlign) | 93.85 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 94.37 |
| SFT (NLL all) | 92.69 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 94.73 `[basetypneg]` | — |
| online TC | 95.87 | 94.47 |
| no TC | 92.92 (RankAlign) | 93.85 |

### Validator accuracy (thr 0) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 81.82 |
| SFT (NLL all) | 50.00 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 78.41 `[basetyp]` | — |
| online TC | 55.68 | 90.91 |
| no TC | 79.55 (RankAlign) | 76.14 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 81.82 |
| SFT (NLL all) | 50.00 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 73.86 `[basetypneg]` | — |
| online TC | 78.41 | 79.55 |
| no TC | 79.55 (RankAlign) | 76.14 |

### Pearson(gen, validator) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 40.81 |
| SFT (NLL all) | 68.12 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 76.07 `[basetyp]` | — |
| online TC | 70.79 | 78.10 |
| no TC | 71.16 (RankAlign) | 75.72 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 17.94 |
| SFT (NLL all) | 49.19 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 73.12 `[basetypneg]` | — |
| online TC | 82.60 | 81.06 |
| no TC | 51.84 (RankAlign) | 29.28 |
