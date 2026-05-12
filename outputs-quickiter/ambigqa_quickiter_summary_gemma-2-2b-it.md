# Ambigqa quick-iter (gemma-2-2b-it) — ambigqa-train-as-test

**Task:** ambigqa-train-as-test (7,992 items, 1,998/5,994 yes/no). **Validator:** log-odds (`--validator-log-odds`). **Metrics from `summarize_scores_file.py`, generator column variant `tc`** (TC-corrected gen score where applicable).

Train ≡ test by construction — these tables are a memorization probe; do NOT read them as cross-task generalization.

**All numeric cells are raw values × 100** (i.e. ROC-AUC and accuracy are in percentage points; Pearson is in 0–100 units).

Tables follow the canonical 4-table layout (see [docs/results_table_format.md](../docs/results_table_format.md)): T1/T3 list base + SFT for the self/neg eval refs; T2/T4 are the 3×2 TC × pairs grids. Cells in T2/T4 use the *best* eval ref per row (offline TC → `basetyp[neg]`; everything else → `self`/`neg`). The (offline TC, online pairs) cell is always blank because that variant is not in the launcher.

Long-form metrics: [quickiter_metrics_long_ambigqa.csv](quickiter_metrics_long_ambigqa.csv)

### Generator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 52.14 |
| SFT (NLL all) | 88.11 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 63.60 `[basetyp]` | — |
| online TC | 67.23 | 53.83 |
| no TC | 63.80 (RankAlign) | 64.00 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 66.40 |
| SFT (NLL all) | 75.10 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 64.34 `[basetypneg]` | — |
| online TC | 65.32 | 63.16 |
| no TC | 57.09 (RankAlign) | 58.22 |

### Validator ROC-AUC — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 58.84 |
| SFT (NLL all) | 57.68 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 57.09 `[basetyp]` | — |
| online TC | 52.71 | 47.42 |
| no TC | 62.15 (RankAlign) | 63.74 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 58.84 |
| SFT (NLL all) | 57.68 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 54.98 `[basetypneg]` | — |
| online TC | 54.54 | 48.63 |
| no TC | 62.15 (RankAlign) | 63.74 |

### Validator accuracy (thr 0) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 61.11 |
| SFT (NLL all) | 58.88 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 57.53 `[basetyp]` | — |
| online TC | 75.00 | 75.00 |
| no TC | 74.87 (RankAlign) | 55.07 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 61.11 |
| SFT (NLL all) | 58.88 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 51.14 `[basetypneg]` | — |
| online TC | 65.47 | 25.00 |
| no TC | 74.87 (RankAlign) | 55.07 |

### Pearson(gen, validator) — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 38.76 |
| SFT (NLL all) | 7.28 |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 22.70 `[basetyp]` | — |
| online TC | 27.25 | 23.75 |
| no TC | 42.34 (RankAlign) | 30.80 |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 27.97 |
| SFT (NLL all) | 19.74 |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 34.35 `[basetypneg]` | — |
| online TC | 32.95 | 18.13 |
| no TC | 7.62 (RankAlign) | 29.35 |
