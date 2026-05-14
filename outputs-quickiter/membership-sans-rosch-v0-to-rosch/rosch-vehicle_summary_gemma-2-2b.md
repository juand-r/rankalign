# Cross-task quick-iter (gemma-2-2b) — trained on membership-sans-rosch-v0, evaluated on **rosch-vehicle**

**Train task:** membership-sans-rosch-v0 (model checkpoints). **Eval task:** rosch-vehicle (82 items, 41/41 yes/no). **Validator:** log-odds (`--validator-log-odds`). **Metrics from `summarize_scores_file.py`, generator column variant `tc`** (TC-corrected gen score where applicable).

Models were trained on a *different* rosch category set (furniture+bird) and are evaluated here as out-of-distribution. This IS a generalization read.

**All numeric cells are raw values × 100** (i.e. ROC-AUC and accuracy are in percentage points; Pearson is in 0–100 units).

Tables follow the canonical 4-table layout (see [docs/results_table_format.md](../docs/results_table_format.md)): T1/T3 list base + SFT for the self/neg eval refs; T2/T4 are the 3×2 TC × pairs grids. Cells in T2/T4 use the *best* eval ref per row (offline TC → `basetyp[neg]`; everything else → `self`/`neg`). The (offline TC, online pairs) cell is always blank because that variant is not in the launcher.

Long-form metrics: [quickiter_metrics_long_membership_to_rosch.csv](quickiter_metrics_long_membership_to_rosch.csv)

### Generator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 85.69 |
| SFT (NLL all) | 86.62 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 93.07 `[basetyp]` | — |
| online TC | 90.30 | 86.05 |
| no TC | 92.30 (RankAlign) | 91.73 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 77.04 |
| SFT (NLL all) | 47.89 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 87.98 `[basetypneg]` | — |
| online TC | 87.33 | 87.66 |
| no TC | 77.93 (RankAlign) | 58.48 |

### Validator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 96.61 |
| SFT (NLL all) | 93.96 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 94.77 `[basetyp]` | — |
| online TC | 93.04 | 95.48 |
| no TC | 93.04 (RankAlign) | 93.99 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 96.61 |
| SFT (NLL all) | 93.96 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 87.69 `[basetypneg]` | — |
| online TC | 90.72 | 95.12 |
| no TC | 93.04 (RankAlign) | 93.99 |

### Validator accuracy (thr 0) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 82.93 |
| SFT (NLL all) | 50.00 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 91.46 `[basetyp]` | — |
| online TC | 57.32 | 89.02 |
| no TC | 86.59 (RankAlign) | 70.73 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 82.93 |
| SFT (NLL all) | 50.00 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 68.29 `[basetypneg]` | — |
| online TC | 68.29 | 80.49 |
| no TC | 86.59 (RankAlign) | 70.73 |

### Pearson(gen, validator) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 55.03 |
| SFT (NLL all) | 60.40 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 73.57 `[basetyp]` | — |
| online TC | 62.11 | 62.31 |
| no TC | 66.83 (RankAlign) | 68.33 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 32.30 |
| SFT (NLL all) | 29.20 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 75.05 `[basetypneg]` | — |
| online TC | 79.27 | 71.44 |
| no TC | 56.65 (RankAlign) | 25.77 |
