# Rosch quick-iter (gemma-2-2b) — rosch-furniture-and-bird

**Task:** combined furniture + bird (186 items, 93/93 yes/no). **Validator:** log-odds (`--validator-log-odds`). **Metrics from `summarize_scores_file.py`, generator column variant `tc`** (TC-corrected gen score where applicable).

**All numeric cells are raw values × 100** (i.e. ROC-AUC and accuracy are in percentage points; Pearson is in 0–100 units).

Tables follow the canonical 4-table layout (see [docs/results_table_format.md](../docs/results_table_format.md)): T1/T3 = baselines (Base HF + SFT) for the self/neg eval refs; T2/T4 = the 3×2 TC × pairs grids for self/neg. Cells in T2/T4 use the *best* eval ref per row (offline TC → `basetyp[neg]`; everything else → `self`/`neg`). The (offline TC, online pairs) cell is always blank because that variant is not in the launcher.

Long-form metrics: [quickiter_metrics_long.csv](quickiter_metrics_long.csv)

### Generator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 74.66 |
| SFT (NLL all) | 99.53 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 95.10 `[basetyp]` | — |
| online TC | 95.75 | 92.02 |
| no TC | 84.75 (RankAlign) | 85.44 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 73.75 |
| SFT (NLL all) | 80.99 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 92.83 `[basetypneg]` | — |
| online TC | 88.73 | 90.76 |
| no TC | 86.44 (RankAlign) | 77.89 |

### Validator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 96.05 |
| SFT (NLL all) | 93.22 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 82.07 `[basetyp]` | — |
| online TC | 93.72 | 89.35 |
| no TC | 91.09 (RankAlign) | 95.42 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 96.05 |
| SFT (NLL all) | 93.22 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 83.80 `[basetypneg]` | — |
| online TC | 86.68 | 94.48 |
| no TC | 91.09 (RankAlign) | 95.42 |

### Validator accuracy (thr 0) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 87.63 |
| SFT (NLL all) | 87.63 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 60.75 `[basetyp]` | — |
| online TC | 87.63 | 73.12 |
| no TC | 76.34 (RankAlign) | 88.71 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 87.63 |
| SFT (NLL all) | 87.63 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 66.13 `[basetypneg]` | — |
| online TC | 58.06 | 71.51 |
| no TC | 76.34 (RankAlign) | 88.71 |

### Pearson(gen, validator) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 50.10 |
| SFT (NLL all) | 61.03 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 59.21 `[basetyp]` | — |
| online TC | 88.18 | 69.75 |
| no TC | 64.13 (RankAlign) | 73.40 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 34.64 |
| SFT (NLL all) | 33.88 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 46.55 `[basetypneg]` | — |
| online TC | 50.88 | 88.66 |
| no TC | 59.19 (RankAlign) | 42.61 |
