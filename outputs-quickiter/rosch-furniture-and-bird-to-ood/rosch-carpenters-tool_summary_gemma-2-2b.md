# Cross-task quick-iter (gemma-2-2b) — trained on rosch-furniture-and-bird, evaluated on **rosch-carpenters-tool**

**Train task:** rosch-furniture-and-bird (model checkpoints). **Eval task:** rosch-carpenters-tool (82 items, 41/41 yes/no). **Validator:** log-odds (`--validator-log-odds`). **Metrics from `summarize_scores_file.py`, generator column variant `tc`** (TC-corrected gen score where applicable).

Models were trained on a *different* rosch category set (furniture+bird) and are evaluated here as out-of-distribution. This IS a generalization read.

**All numeric cells are raw values × 100** (i.e. ROC-AUC and accuracy are in percentage points; Pearson is in 0–100 units).

Long-form metrics: [quickiter_metrics_long_crosstask.csv](quickiter_metrics_long_crosstask.csv)

### Generator ROC-AUC (`tc` column) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 71.09 | 74.36 | — | — |
| 1 | RankAlign baseline | 58.51 | 77.84 | 58.00 | 46.52 |
| 2 | + offline self-TC | 58.83 | — | 59.58 | — |
| 3 | + online self-TC | 61.69 | — | 59.73 | — |
| 4 | + online pairs | 60.77 | 78.41 | 60.08 | 41.94 |
| 5 | + both online (self) | 49.82 | — | 45.69 | — |
| 6 | SFT (NLL all) | 53.33 | 62.58 | 53.03 | 35.87 |
| 7 | + offline neg-TC | — | 53.36 | — | 40.54 |
| 8 | + online neg-TC | — | 55.23 | — | 42.68 |
| 9 | + both online (neg) | — | 64.84 | — | 34.62 |

### Validator ROC-AUC (same across gen variants; shown for reference) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 71.33 | 71.33 | — | — |
| 1 | RankAlign baseline | 65.85 | 65.85 | 65.85 | 65.85 |
| 2 | + offline self-TC | 73.71 | — | 73.71 | — |
| 3 | + online self-TC | 72.16 | — | 72.16 | — |
| 4 | + online pairs | 68.71 | 68.71 | 68.71 | 68.71 |
| 5 | + both online (self) | 66.39 | — | 66.39 | — |
| 6 | SFT (NLL all) | 69.04 | 69.04 | 69.04 | 69.04 |
| 7 | + offline neg-TC | — | 68.00 | — | 68.00 |
| 8 | + online neg-TC | — | 58.60 | — | 58.60 |
| 9 | + both online (neg) | — | 61.93 | — | 61.93 |

### Validator accuracy (threshold 0) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 57.32 | 57.32 | — | — |
| 1 | RankAlign baseline | 59.76 | 59.76 | 59.76 | 59.76 |
| 2 | + offline self-TC | 63.41 | — | 63.41 | — |
| 3 | + online self-TC | 63.41 | — | 63.41 | — |
| 4 | + online pairs | 54.88 | 54.88 | 54.88 | 54.88 |
| 5 | + both online (self) | 62.20 | — | 62.20 | — |
| 6 | SFT (NLL all) | 60.98 | 60.98 | 60.98 | 60.98 |
| 7 | + offline neg-TC | — | 59.76 | — | 59.76 |
| 8 | + online neg-TC | — | 51.22 | — | 51.22 |
| 9 | + both online (neg) | — | 56.10 | — | 56.10 |

### Pearson(gen, validator) — `tc` gen vs val_score — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 61.59 | 29.19 | — | — |
| 1 | RankAlign baseline | 51.77 | 55.87 | 57.53 | 16.10 |
| 2 | + offline self-TC | 59.87 | — | 52.93 | — |
| 3 | + online self-TC | 40.83 | — | 43.02 | — |
| 4 | + online pairs | 55.03 | 42.23 | 53.05 | -1.58 |
| 5 | + both online (self) | 27.20 | — | 20.58 | — |
| 6 | SFT (NLL all) | 34.52 | 7.24 | 34.06 | -18.37 |
| 7 | + offline neg-TC | — | 23.53 | — | -20.32 |
| 8 | + online neg-TC | — | 42.95 | — | -33.22 |
| 9 | + both online (neg) | — | 59.58 | — | -23.77 |
