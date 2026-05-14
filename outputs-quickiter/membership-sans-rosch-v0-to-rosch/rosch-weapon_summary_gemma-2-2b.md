# Cross-task quick-iter (gemma-2-2b) — trained on membership-sans-rosch-v0, evaluated on **rosch-weapon**

**Train task:** membership-sans-rosch-v0 (model checkpoints). **Eval task:** rosch-weapon (84 items, 42/42 yes/no). **Validator:** log-odds (`--validator-log-odds`). **Metrics from `summarize_scores_file.py`, generator column variant `tc`** (TC-corrected gen score where applicable).

Models were trained on a *different* rosch category set (furniture+bird) and are evaluated here as out-of-distribution. This IS a generalization read.

**All numeric cells are raw values × 100** (i.e. ROC-AUC and accuracy are in percentage points; Pearson is in 0–100 units).

Tables follow the canonical 4-table layout (see [docs/results_table_format.md](../docs/results_table_format.md)): T1/T3 list base + SFT for the self/neg eval refs; T2/T4 are the 3×2 TC × pairs grids. Cells in T2/T4 use the *best* eval ref per row (offline TC → `basetyp[neg]`; everything else → `self`/`neg`). The (offline TC, online pairs) cell is always blank because that variant is not in the launcher.

Long-form metrics: [quickiter_metrics_long_membership_to_rosch.csv](quickiter_metrics_long_membership_to_rosch.csv)

### Generator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 77.72 |
| SFT (NLL all) | 78.83 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 78.54 `[basetyp]` | — |
| online TC | 73.98 | 73.07 |
| no TC | 75.88 (RankAlign) | 76.30 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 86.99 |
| SFT (NLL all) | 79.37 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 73.84 `[basetypneg]` | — |
| online TC | 76.73 | 75.09 |
| no TC | 87.39 (RankAlign) | 68.74 |

### Validator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 90.59 |
| SFT (NLL all) | 90.45 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 92.29 `[basetyp]` | — |
| online TC | 92.23 | 90.48 |
| no TC | 88.21 (RankAlign) | 87.98 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 90.59 |
| SFT (NLL all) | 90.45 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 89.46 `[basetypneg]` | — |
| online TC | 91.55 | 89.06 |
| no TC | 88.21 (RankAlign) | 87.98 |

### Validator accuracy (thr 0) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 83.33 |
| SFT (NLL all) | 54.76 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 85.71 `[basetyp]` | — |
| online TC | 57.14 | 82.14 |
| no TC | 77.38 (RankAlign) | 67.86 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 83.33 |
| SFT (NLL all) | 54.76 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 70.24 `[basetypneg]` | — |
| online TC | 73.81 | 83.33 |
| no TC | 77.38 (RankAlign) | 67.86 |

### Pearson(gen, validator) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 45.22 |
| SFT (NLL all) | 53.20 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 64.97 `[basetyp]` | — |
| online TC | 54.21 | 56.72 |
| no TC | 67.64 (RankAlign) | 65.37 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 50.01 |
| SFT (NLL all) | 61.46 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 68.59 `[basetypneg]` | — |
| online TC | 75.20 | 72.05 |
| no TC | 62.34 (RankAlign) | 20.53 |
