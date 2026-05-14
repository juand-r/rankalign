# Cross-task quick-iter (gemma-2-2b) — trained on membership-sans-rosch-v0, evaluated on **rosch-fruit**

**Train task:** membership-sans-rosch-v0 (model checkpoints). **Eval task:** rosch-fruit (84 items, 42/42 yes/no). **Validator:** log-odds (`--validator-log-odds`). **Metrics from `summarize_scores_file.py`, generator column variant `tc`** (TC-corrected gen score where applicable).

Models were trained on a *different* rosch category set (furniture+bird) and are evaluated here as out-of-distribution. This IS a generalization read.

**All numeric cells are raw values × 100** (i.e. ROC-AUC and accuracy are in percentage points; Pearson is in 0–100 units).

Tables follow the canonical 4-table layout (see [docs/results_table_format.md](../docs/results_table_format.md)): T1/T3 list base + SFT for the self/neg eval refs; T2/T4 are the 3×2 TC × pairs grids. Cells in T2/T4 use the *best* eval ref per row (offline TC → `basetyp[neg]`; everything else → `self`/`neg`). The (offline TC, online pairs) cell is always blank because that variant is not in the launcher.

Long-form metrics: [quickiter_metrics_long_membership_to_rosch.csv](quickiter_metrics_long_membership_to_rosch.csv)

### Generator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 76.73 |
| SFT (NLL all) | 81.15 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 91.67 `[basetyp]` | — |
| online TC | 92.46 | 90.56 |
| no TC | 85.46 (RankAlign) | 86.05 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 84.98 |
| SFT (NLL all) | 82.54 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 81.43 `[basetypneg]` | — |
| online TC | 96.77 | 90.87 |
| no TC | 95.46 (RankAlign) | 85.01 |

### Validator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 91.72 |
| SFT (NLL all) | 90.84 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 91.95 `[basetyp]` | — |
| online TC | 89.85 | 92.12 |
| no TC | 93.20 (RankAlign) | 93.37 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 91.72 |
| SFT (NLL all) | 90.84 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 93.48 `[basetypneg]` | — |
| online TC | 93.71 | 93.37 |
| no TC | 93.20 (RankAlign) | 93.37 |

### Validator accuracy (thr 0) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 83.33 |
| SFT (NLL all) | 52.38 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 86.90 `[basetyp]` | — |
| online TC | 58.33 | 83.33 |
| no TC | 78.57 (RankAlign) | 84.52 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 83.33 |
| SFT (NLL all) | 52.38 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 79.76 `[basetypneg]` | — |
| online TC | 80.95 | 85.71 |
| no TC | 78.57 (RankAlign) | 84.52 |

### Pearson(gen, validator) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 69.00 |
| SFT (NLL all) | 69.89 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 76.56 `[basetyp]` | — |
| online TC | 77.51 | 76.26 |
| no TC | 76.20 (RankAlign) | 75.15 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 53.72 |
| SFT (NLL all) | 32.41 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 54.50 `[basetypneg]` | — |
| online TC | 84.14 | 82.70 |
| no TC | 65.12 (RankAlign) | 53.68 |
