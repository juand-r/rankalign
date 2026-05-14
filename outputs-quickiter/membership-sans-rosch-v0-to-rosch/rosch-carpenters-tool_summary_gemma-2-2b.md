# Cross-task quick-iter (gemma-2-2b) — trained on membership-sans-rosch-v0, evaluated on **rosch-carpenters-tool**

**Train task:** membership-sans-rosch-v0 (model checkpoints). **Eval task:** rosch-carpenters-tool (82 items, 41/41 yes/no). **Validator:** log-odds (`--validator-log-odds`). **Metrics from `summarize_scores_file.py`, generator column variant `tc`** (TC-corrected gen score where applicable).

Models were trained on a *different* rosch category set (furniture+bird) and are evaluated here as out-of-distribution. This IS a generalization read.

**All numeric cells are raw values × 100** (i.e. ROC-AUC and accuracy are in percentage points; Pearson is in 0–100 units).

Tables follow the canonical 4-table layout (see [docs/results_table_format.md](../docs/results_table_format.md)): T1/T3 list base + SFT for the self/neg eval refs; T2/T4 are the 3×2 TC × pairs grids. Cells in T2/T4 use the *best* eval ref per row (offline TC → `basetyp[neg]`; everything else → `self`/`neg`). The (offline TC, online pairs) cell is always blank because that variant is not in the launcher.

Long-form metrics: [quickiter_metrics_long_membership_to_rosch.csv](quickiter_metrics_long_membership_to_rosch.csv)

### Generator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 71.09 |
| SFT (NLL all) | 73.80 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 73.17 `[basetyp]` | — |
| online TC | 73.35 | 68.62 |
| no TC | 63.98 (RankAlign) | 59.82 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 74.36 |
| SFT (NLL all) | 64.96 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 64.43 `[basetypneg]` | — |
| online TC | 74.39 | 64.13 |
| no TC | 77.96 (RankAlign) | 63.62 |

### Validator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 71.33 |
| SFT (NLL all) | 74.66 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 69.54 `[basetyp]` | — |
| online TC | 70.67 | 69.54 |
| no TC | 68.95 (RankAlign) | 66.15 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 71.33 |
| SFT (NLL all) | 74.66 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 64.37 `[basetypneg]` | — |
| online TC | 71.68 | 70.67 |
| no TC | 68.95 (RankAlign) | 66.15 |

### Validator accuracy (thr 0) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 57.32 |
| SFT (NLL all) | 50.00 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 65.85 `[basetyp]` | — |
| online TC | 51.22 | 59.76 |
| no TC | 62.20 (RankAlign) | 56.10 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 57.32 |
| SFT (NLL all) | 50.00 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 57.32 `[basetypneg]` | — |
| online TC | 57.32 | 62.20 |
| no TC | 62.20 (RankAlign) | 56.10 |

### Pearson(gen, validator) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 61.59 |
| SFT (NLL all) | 47.85 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 66.33 `[basetyp]` | — |
| online TC | 68.53 | 67.82 |
| no TC | 56.73 (RankAlign) | 63.94 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 29.19 |
| SFT (NLL all) | 32.89 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 36.91 `[basetypneg]` | — |
| online TC | 79.40 | 68.14 |
| no TC | 40.56 (RankAlign) | 39.49 |
