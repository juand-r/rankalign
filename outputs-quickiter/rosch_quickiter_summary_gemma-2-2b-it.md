# Rosch quick-iter (gemma-2-2b-it) — rosch-furniture-and-bird

**Task:** rosch-furniture-and-bird (186 items, 93/93 yes/no). **Validator:** log-odds (`--validator-log-odds`). **Metrics from `summarize_scores_file.py`, generator column variant `tc`** (TC-corrected gen score where applicable).

Train ≡ test by construction — these tables are a memorization probe; do NOT read them as cross-task generalization.

**All numeric cells are raw values × 100** (i.e. ROC-AUC and accuracy are in percentage points; Pearson is in 0–100 units).

Tables follow the canonical 4-table layout (see [docs/results_table_format.md](../docs/results_table_format.md)): T1/T3 list base + SFT for the self/neg eval refs; T2/T4 are the 3×2 TC × pairs grids. Cells in T2/T4 use the *best* eval ref per row (offline TC → `basetyp[neg]`; everything else → `self`/`neg`). The (offline TC, online pairs) cell is always blank because that variant is not in the launcher.

Long-form metrics: [quickiter_metrics_long.csv](quickiter_metrics_long.csv)

### Generator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 87.37 |
| SFT (NLL all) | 98.18 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 93.63 `[basetyp]` | — |
| online TC | 89.82 | 84.15 |
| no TC | 85.16 (RankAlign) | 78.10 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 93.44 |
| SFT (NLL all) | 86.71 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 91.50 `[basetypneg]` | — |
| online TC | 89.98 | 80.84 |
| no TC | 74.99 (RankAlign) | 79.06 |

### Validator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 96.16 |
| SFT (NLL all) | 96.90 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 96.15 `[basetyp]` | — |
| online TC | 92.57 | 93.99 |
| no TC | 96.05 (RankAlign) | 85.71 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 96.16 |
| SFT (NLL all) | 96.90 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 88.37 `[basetypneg]` | — |
| online TC | 93.87 | 83.82 |
| no TC | 96.05 (RankAlign) | 85.71 |

### Validator accuracy (thr 0) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 88.17 |
| SFT (NLL all) | 50.00 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 90.32 `[basetyp]` | — |
| online TC | 83.33 | 58.06 |
| no TC | 88.71 (RankAlign) | 68.28 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 88.17 |
| SFT (NLL all) | 50.00 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 76.34 `[basetypneg]` | — |
| online TC | 87.10 | 50.00 |
| no TC | 88.71 (RankAlign) | 68.28 |

### Pearson(gen, validator) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 56.48 |
| SFT (NLL all) | 69.67 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 78.64 `[basetyp]` | — |
| online TC | 80.74 | 77.62 |
| no TC | 66.31 (RankAlign) | 68.15 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 67.11 |
| SFT (NLL all) | 66.40 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 70.05 `[basetypneg]` | — |
| online TC | 62.61 | 70.65 |
| no TC | 41.72 (RankAlign) | 59.96 |
