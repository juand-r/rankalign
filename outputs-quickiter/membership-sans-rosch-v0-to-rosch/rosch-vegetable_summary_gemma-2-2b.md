# Cross-task quick-iter (gemma-2-2b) — trained on membership-sans-rosch-v0, evaluated on **rosch-vegetable**

**Train task:** membership-sans-rosch-v0 (model checkpoints). **Eval task:** rosch-vegetable (86 items, 43/43 yes/no). **Validator:** log-odds (`--validator-log-odds`). **Metrics from `summarize_scores_file.py`, generator column variant `tc`** (TC-corrected gen score where applicable).

Models were trained on a *different* rosch category set (furniture+bird) and are evaluated here as out-of-distribution. This IS a generalization read.

**All numeric cells are raw values × 100** (i.e. ROC-AUC and accuracy are in percentage points; Pearson is in 0–100 units).

Tables follow the canonical 4-table layout (see [docs/results_table_format.md](../docs/results_table_format.md)): T1/T3 list base + SFT for the self/neg eval refs; T2/T4 are the 3×2 TC × pairs grids. Cells in T2/T4 use the *best* eval ref per row (offline TC → `basetyp[neg]`; everything else → `self`/`neg`). The (offline TC, online pairs) cell is always blank because that variant is not in the launcher.

Long-form metrics: [quickiter_metrics_long_membership_to_rosch.csv](quickiter_metrics_long_membership_to_rosch.csv)

### Generator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 82.77 |
| SFT (NLL all) | 80.07 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 86.40 `[basetyp]` | — |
| online TC | 81.10 | 79.45 |
| no TC | 80.31 (RankAlign) | 83.26 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 92.10 |
| SFT (NLL all) | 77.39 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 74.74 `[basetypneg]` | — |
| online TC | 86.18 | 85.26 |
| no TC | 89.99 (RankAlign) | 92.05 |

### Validator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 94.75 |
| SFT (NLL all) | 94.13 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 94.86 `[basetyp]` | — |
| online TC | 91.13 | 96.43 |
| no TC | 92.70 (RankAlign) | 94.54 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 94.75 |
| SFT (NLL all) | 94.13 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 91.67 `[basetypneg]` | — |
| online TC | 96.00 | 94.75 |
| no TC | 92.70 (RankAlign) | 94.54 |

### Validator accuracy (thr 0) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 84.88 |
| SFT (NLL all) | 55.81 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 83.72 `[basetyp]` | — |
| online TC | 56.98 | 86.05 |
| no TC | 84.88 (RankAlign) | 81.40 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 84.88 |
| SFT (NLL all) | 55.81 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 86.05 `[basetypneg]` | — |
| online TC | 86.05 | 86.05 |
| no TC | 84.88 (RankAlign) | 81.40 |

### Pearson(gen, validator) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 65.82 |
| SFT (NLL all) | 68.21 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 73.47 `[basetyp]` | — |
| online TC | 71.24 | 72.14 |
| no TC | 74.61 (RankAlign) | 78.12 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 60.35 |
| SFT (NLL all) | 40.13 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 59.22 `[basetypneg]` | — |
| online TC | 74.75 | 80.31 |
| no TC | 66.77 (RankAlign) | 63.02 |
