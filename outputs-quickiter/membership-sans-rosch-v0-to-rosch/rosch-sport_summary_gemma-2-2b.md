# Cross-task quick-iter (gemma-2-2b) — trained on membership-sans-rosch-v0, evaluated on **rosch-sport**

**Train task:** membership-sans-rosch-v0 (model checkpoints). **Eval task:** rosch-sport (92 items, 46/46 yes/no). **Validator:** log-odds (`--validator-log-odds`). **Metrics from `summarize_scores_file.py`, generator column variant `tc`** (TC-corrected gen score where applicable).

Models were trained on a *different* rosch category set (furniture+bird) and are evaluated here as out-of-distribution. This IS a generalization read.

**All numeric cells are raw values × 100** (i.e. ROC-AUC and accuracy are in percentage points; Pearson is in 0–100 units).

Tables follow the canonical 4-table layout (see [docs/results_table_format.md](../docs/results_table_format.md)): T1/T3 list base + SFT for the self/neg eval refs; T2/T4 are the 3×2 TC × pairs grids. Cells in T2/T4 use the *best* eval ref per row (offline TC → `basetyp[neg]`; everything else → `self`/`neg`). The (offline TC, online pairs) cell is always blank because that variant is not in the launcher.

Long-form metrics: [quickiter_metrics_long_membership_to_rosch.csv](quickiter_metrics_long_membership_to_rosch.csv)

### Generator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 80.41 |
| SFT (NLL all) | 82.07 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 90.57 `[basetyp]` | — |
| online TC | 89.15 | 84.85 |
| no TC | 85.92 (RankAlign) | 83.93 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 78.69 |
| SFT (NLL all) | 73.68 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 81.21 `[basetypneg]` | — |
| online TC | 89.70 | 88.49 |
| no TC | 76.39 (RankAlign) | 72.19 |

### Validator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 86.29 |
| SFT (NLL all) | 88.26 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 89.74 `[basetyp]` | — |
| online TC | 90.17 | 90.26 |
| no TC | 77.17 (RankAlign) | 83.84 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 86.29 |
| SFT (NLL all) | 88.26 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 82.09 `[basetypneg]` | — |
| online TC | 90.64 | 83.60 |
| no TC | 77.17 (RankAlign) | 83.84 |

### Validator accuracy (thr 0) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 77.17 |
| SFT (NLL all) | 52.17 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 69.57 `[basetyp]` | — |
| online TC | 68.48 | 81.52 |
| no TC | 65.22 (RankAlign) | 55.43 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 77.17 |
| SFT (NLL all) | 52.17 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 70.65 `[basetypneg]` | — |
| online TC | 85.87 | 73.91 |
| no TC | 65.22 (RankAlign) | 55.43 |

### Pearson(gen, validator) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 54.98 |
| SFT (NLL all) | 60.16 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 73.95 `[basetyp]` | — |
| online TC | 73.20 | 73.81 |
| no TC | 59.97 (RankAlign) | 56.03 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 38.56 |
| SFT (NLL all) | 54.75 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 49.80 `[basetypneg]` | — |
| online TC | 83.00 | 78.68 |
| no TC | 50.78 (RankAlign) | 26.82 |
