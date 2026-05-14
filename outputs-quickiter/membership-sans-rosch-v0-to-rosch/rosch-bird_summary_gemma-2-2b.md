# Cross-task quick-iter (gemma-2-2b) — trained on membership-sans-rosch-v0, evaluated on **rosch-bird**

**Train task:** membership-sans-rosch-v0 (model checkpoints). **Eval task:** rosch-bird (106 items, 53/53 yes/no). **Validator:** log-odds (`--validator-log-odds`). **Metrics from `summarize_scores_file.py`, generator column variant `tc`** (TC-corrected gen score where applicable).

Models were trained on a *different* rosch category set (furniture+bird) and are evaluated here as out-of-distribution. This IS a generalization read.

**All numeric cells are raw values × 100** (i.e. ROC-AUC and accuracy are in percentage points; Pearson is in 0–100 units).

Tables follow the canonical 4-table layout (see [docs/results_table_format.md](../docs/results_table_format.md)): T1/T3 list base + SFT for the self/neg eval refs; T2/T4 are the 3×2 TC × pairs grids. Cells in T2/T4 use the *best* eval ref per row (offline TC → `basetyp[neg]`; everything else → `self`/`neg`). The (offline TC, online pairs) cell is always blank because that variant is not in the launcher.

Long-form metrics: [quickiter_metrics_long_membership_to_rosch.csv](quickiter_metrics_long_membership_to_rosch.csv)

### Generator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 70.35 |
| SFT (NLL all) | 89.50 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 87.56 `[basetyp]` | — |
| online TC | 79.05 | 81.65 |
| no TC | 79.57 (RankAlign) | 76.17 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 59.58 |
| SFT (NLL all) | 84.92 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 68.85 `[basetypneg]` | — |
| online TC | 82.41 | 73.02 |
| no TC | 71.15 (RankAlign) | 82.04 |

### Validator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 98.68 |
| SFT (NLL all) | 98.11 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 98.75 `[basetyp]` | — |
| online TC | 98.43 | 98.75 |
| no TC | 98.50 (RankAlign) | 97.76 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 98.68 |
| SFT (NLL all) | 98.11 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 97.86 `[basetypneg]` | — |
| online TC | 98.54 | 98.65 |
| no TC | 98.50 (RankAlign) | 97.76 |

### Validator accuracy (thr 0) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 94.34 |
| SFT (NLL all) | 56.60 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 92.45 `[basetyp]` | — |
| online TC | 79.25 | 95.28 |
| no TC | 95.28 (RankAlign) | 90.57 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 94.34 |
| SFT (NLL all) | 56.60 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 93.40 `[basetypneg]` | — |
| online TC | 94.34 | 95.28 |
| no TC | 95.28 (RankAlign) | 90.57 |

### Pearson(gen, validator) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 44.93 |
| SFT (NLL all) | 67.03 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 71.02 `[basetyp]` | — |
| online TC | 63.59 | 60.49 |
| no TC | 67.22 (RankAlign) | 52.19 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 22.34 |
| SFT (NLL all) | 65.76 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 38.03 `[basetypneg]` | — |
| online TC | 69.58 | 52.46 |
| no TC | 49.05 (RankAlign) | 52.86 |
