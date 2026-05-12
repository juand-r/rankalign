# Cross-task mean across 8 OOD rosch tasks (gemma-2-2b)

**Train task:** rosch-furniture-and-bird-all (models trained here, 8 OOD eval tasks below).

**OOD tasks:** rosch-carpenters-tool, rosch-clothing, rosch-fruit, rosch-sport, rosch-toy, rosch-vegetable, rosch-vehicle, rosch-weapon.

**Validator:** log-odds (`--validator-log-odds`). **Generator column variant:** `tc` (TC-corrected gen score where applicable).

**All numeric cells are raw values × 100** (percentage points for ROC/accuracy; 0–100 units for Pearson). Reported as **mean (std)** across the 8 OOD tasks.

Tables follow the canonical 4-table layout (see [docs/results_table_format.md](../../docs/results_table_format.md)): T1/T3 = baselines (Base HF + SFT) for the self/neg eval refs; T2/T4 = the 3×2 TC × pairs grids for self/neg. Cells in T2/T4 use the *best* eval ref per row (offline TC → `basetyp[neg]`; everything else → `self`/`neg`). The (offline TC, online pairs) cell is always blank because that variant is not in the launcher.

Long-form metrics: [quickiter_metrics_long_crosstask.csv](quickiter_metrics_long_crosstask.csv)

### Generator ROC-AUC — mean (std) across 8 OOD tasks — × 100 — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 77.83 (5.58) |
| SFT (NLL all) | 69.34 (7.77) |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 74.26 (9.00) `[basetyp]` | — |
| online TC | 76.72 (8.84) | 65.33 (8.82) |
| no TC | 76.42 (8.67) (RankAlign) | 77.88 (8.37) |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 78.07 (11.25) |
| SFT (NLL all) | 60.72 (20.48) |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 55.63 (12.17) `[basetypneg]` | — |
| online TC | 63.54 (7.89) | 79.71 (9.70) |
| no TC | 71.01 (14.62) (RankAlign) | 67.44 (13.78) |

### Validator ROC-AUC — mean (std) across 8 OOD tasks — × 100 — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 88.38 (8.49) |
| SFT (NLL all) | 88.56 (9.16) |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 80.49 (8.98) `[basetyp]` | — |
| online TC | 86.86 (8.47) | 86.83 (9.84) |
| no TC | 86.67 (10.52) (RankAlign) | 86.44 (9.22) |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 88.38 (8.49) |
| SFT (NLL all) | 88.56 (9.16) |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 84.45 (9.19) `[basetypneg]` | — |
| online TC | 83.54 (10.68) | 85.96 (12.09) |
| no TC | 86.67 (10.52) (RankAlign) | 86.44 (9.22) |

### Validator accuracy (thr 0) — mean (std) across 8 OOD tasks — × 100 — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 76.83 (10.42) |
| SFT (NLL all) | 79.65 (11.39) |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 66.40 (10.43) `[basetyp]` | — |
| online TC | 75.61 (8.66) | 67.77 (10.82) |
| no TC | 69.40 (9.50) (RankAlign) | 78.18 (10.60) |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 76.83 (10.42) |
| SFT (NLL all) | 79.65 (11.39) |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 59.94 (10.87) `[basetypneg]` | — |
| online TC | 54.92 (5.51) | 69.17 (7.23) |
| no TC | 69.40 (9.50) (RankAlign) | 78.18 (10.60) |

### Pearson(gen, validator) — mean (std) across 8 OOD tasks — × 100 — × 100

#### Table 1 — baselines, self eval

| variant | value |
| --- | --- |
| Base HF | 54.65 (10.41) |
| SFT (NLL all) | 34.61 (15.75) |

#### Table 2 — self eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 50.30 (7.72) `[basetyp]` | — |
| online TC | 57.72 (13.41) | 40.38 (12.17) |
| no TC | 60.40 (9.32) (RankAlign) | 58.52 (10.15) |

#### Table 3 — baselines, neg eval

| variant | value |
| --- | --- |
| Base HF | 40.50 (13.98) |
| SFT (NLL all) | 14.47 (21.79) |

#### Table 4 — neg eval, TC × pairs

|  | offline pairs | online pairs |
| --- | --- | --- |
| offline TC | 10.00 (25.33) `[basetypneg]` | — |
| online TC | 36.16 (14.65) | 73.22 (8.56) |
| no TC | 44.81 (17.96) (RankAlign) | 21.00 (24.20) |
