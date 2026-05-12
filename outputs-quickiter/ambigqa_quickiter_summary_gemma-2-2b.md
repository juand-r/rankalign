# Ambigqa quick-iter (gemma-2-2b) — ambigqa-train-as-test

**Task:** ambigqa-train-as-test (7,992 items, 1,998/5,994 yes/no). **Validator:** log-odds (`--validator-log-odds`). **Metrics from `summarize_scores_file.py`, generator column variant `tc`** (TC-corrected gen score where applicable).

Train ≡ test by construction — these tables are a memorization probe; do NOT read them as cross-task generalization.

**All numeric cells are raw values × 100** (i.e. ROC-AUC and accuracy are in percentage points; Pearson is in 0–100 units).

Tables follow the canonical 4-table layout (see [docs/results_table_format.md](../docs/results_table_format.md)): T1/T3 list base + SFT for the self/neg eval refs; T2/T4 are the 3×2 TC × pairs grids. Cells in T2/T4 use the *best* eval ref per row (offline TC → `basetyp[neg]`; everything else → `self`/`neg`). The (offline TC, online pairs) cell is always blank because that variant is not in the launcher.

Long-form metrics: [quickiter_metrics_long_ambigqa.csv](quickiter_metrics_long_ambigqa.csv)

### Generator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 51.47 |
| SFT (NLL all) | 90.23 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 57.66 `[basetyp]` | — |
| online TC | 55.95 | 49.15 |
| no TC | 60.91 (RankAlign) | 53.16 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 64.25 |
| SFT (NLL all) | 89.22 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 58.85 `[basetypneg]` | — |
| online TC | 57.31 | 41.02 |
| no TC | 61.19 (RankAlign) | 59.10 |

### Validator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 56.09 |
| SFT (NLL all) | 72.61 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 53.18 `[basetyp]` | — |
| online TC | 57.84 | 51.41 |
| no TC | 58.29 (RankAlign) | 47.84 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 56.09 |
| SFT (NLL all) | 72.61 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 55.54 `[basetypneg]` | — |
| online TC | 56.99 | 48.72 |
| no TC | 58.29 (RankAlign) | 47.84 |

### Validator accuracy (thr 0) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 38.84 |
| SFT (NLL all) | 75.00 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 52.29 `[basetyp]` | — |
| online TC | 38.94 | 28.68 |
| no TC | 33.00 (RankAlign) | 25.79 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 38.84 |
| SFT (NLL all) | 75.00 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 25.46 `[basetypneg]` | — |
| online TC | 36.65 | 25.05 |
| no TC | 33.00 (RankAlign) | 25.79 |

### Pearson(gen, validator) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 42.50 |
| SFT (NLL all) | 47.42 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 47.86 `[basetyp]` | — |
| online TC | 43.93 | 57.25 |
| no TC | 42.46 (RankAlign) | 31.72 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 38.92 |
| SFT (NLL all) | 26.21 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 32.87 `[basetypneg]` | — |
| online TC | 58.23 | -6.66 |
| no TC | 18.85 (RankAlign) | 23.84 |
