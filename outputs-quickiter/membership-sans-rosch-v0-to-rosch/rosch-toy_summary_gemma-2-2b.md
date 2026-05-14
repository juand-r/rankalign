# Cross-task quick-iter (gemma-2-2b) — trained on membership-sans-rosch-v0, evaluated on **rosch-toy**

**Train task:** membership-sans-rosch-v0 (model checkpoints). **Eval task:** rosch-toy (72 items, 36/36 yes/no). **Validator:** log-odds (`--validator-log-odds`). **Metrics from `summarize_scores_file.py`, generator column variant `tc`** (TC-corrected gen score where applicable).

Models were trained on a *different* rosch category set (furniture+bird) and are evaluated here as out-of-distribution. This IS a generalization read.

**All numeric cells are raw values × 100** (i.e. ROC-AUC and accuracy are in percentage points; Pearson is in 0–100 units).

Tables follow the canonical 4-table layout (see [docs/results_table_format.md](../docs/results_table_format.md)): T1/T3 list base + SFT for the self/neg eval refs; T2/T4 are the 3×2 TC × pairs grids. Cells in T2/T4 use the *best* eval ref per row (offline TC → `basetyp[neg]`; everything else → `self`/`neg`). The (offline TC, online pairs) cell is always blank because that variant is not in the launcher.

Long-form metrics: [quickiter_metrics_long_membership_to_rosch.csv](quickiter_metrics_long_membership_to_rosch.csv)

### Generator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 79.17 |
| SFT (NLL all) | 79.48 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 78.67 `[basetyp]` | — |
| online TC | 75.50 | 76.93 |
| no TC | 77.16 (RankAlign) | 76.35 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 75.54 |
| SFT (NLL all) | 77.16 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 57.14 `[basetypneg]` | — |
| online TC | 74.38 | 64.12 |
| no TC | 81.02 (RankAlign) | 80.59 |

### Validator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 81.40 |
| SFT (NLL all) | 77.28 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 78.47 `[basetyp]` | — |
| online TC | 76.85 | 77.78 |
| no TC | 71.14 (RankAlign) | 76.54 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 81.40 |
| SFT (NLL all) | 77.28 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 78.09 `[basetypneg]` | — |
| online TC | 76.70 | 75.85 |
| no TC | 71.14 (RankAlign) | 76.54 |

### Validator accuracy (thr 0) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 63.89 |
| SFT (NLL all) | 50.00 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 66.67 `[basetyp]` | — |
| online TC | 50.00 | 63.89 |
| no TC | 61.11 (RankAlign) | 65.28 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 63.89 |
| SFT (NLL all) | 50.00 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 66.67 `[basetypneg]` | — |
| online TC | 58.33 | 66.67 |
| no TC | 61.11 (RankAlign) | 65.28 |

### Pearson(gen, validator) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 44.72 |
| SFT (NLL all) | 45.23 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 60.76 `[basetyp]` | — |
| online TC | 49.95 | 56.93 |
| no TC | 52.88 (RankAlign) | 63.39 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 41.90 |
| SFT (NLL all) | 37.73 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 32.00 `[basetypneg]` | — |
| online TC | 72.86 | 68.25 |
| no TC | 29.95 (RankAlign) | 47.61 |
