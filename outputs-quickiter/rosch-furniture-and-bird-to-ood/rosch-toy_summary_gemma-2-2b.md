# Cross-task quick-iter (gemma-2-2b) — trained on rosch-furniture-and-bird, evaluated on **rosch-toy**

**Train task:** rosch-furniture-and-bird (model checkpoints). **Eval task:** rosch-toy (72 items, 36/36 yes/no). **Validator:** log-odds (`--validator-log-odds`). **Metrics from `summarize_scores_file.py`, generator column variant `tc`** (TC-corrected gen score where applicable).

Models were trained on a *different* rosch category set (furniture+bird) and are evaluated here as out-of-distribution. This IS a generalization read.

**All numeric cells are raw values × 100** (i.e. ROC-AUC and accuracy are in percentage points; Pearson is in 0–100 units).

Long-form metrics: [quickiter_metrics_long_crosstask.csv](quickiter_metrics_long_crosstask.csv)

### Generator ROC-AUC (`tc` column) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 79.17 | 75.54 | — | — |
| 1 | RankAlign baseline | 72.65 | 76.77 | 71.99 | 54.28 |
| 2 | + offline self-TC | 67.86 | — | 65.24 | — |
| 3 | + online self-TC | 66.98 | — | 57.56 | — |
| 4 | + online pairs | 74.69 | 63.54 | 74.85 | 52.89 |
| 5 | + both online (self) | 63.93 | — | 55.86 | — |
| 6 | SFT (NLL all) | 68.25 | 63.23 | 68.71 | 41.78 |
| 7 | + offline neg-TC | — | 34.41 | — | 41.98 |
| 8 | + online neg-TC | — | 56.44 | — | 22.45 |
| 9 | + both online (neg) | — | 66.63 | — | 29.86 |

### Validator ROC-AUC (same across gen variants; shown for reference) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 81.40 | 81.40 | — | — |
| 1 | RankAlign baseline | 76.00 | 76.00 | 76.00 | 76.00 |
| 2 | + offline self-TC | 68.75 | — | 68.75 | — |
| 3 | + online self-TC | 75.39 | — | 75.39 | — |
| 4 | + online pairs | 77.39 | 77.39 | 77.39 | 77.39 |
| 5 | + both online (self) | 80.29 | — | 80.29 | — |
| 6 | SFT (NLL all) | 84.72 | 84.72 | 84.72 | 84.72 |
| 7 | + offline neg-TC | — | 77.85 | — | 77.85 |
| 8 | + online neg-TC | — | 79.24 | — | 79.24 |
| 9 | + both online (neg) | — | 72.45 | — | 72.45 |

### Validator accuracy (threshold 0) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 63.89 | 63.89 | — | — |
| 1 | RankAlign baseline | 65.28 | 65.28 | 65.28 | 65.28 |
| 2 | + offline self-TC | 68.06 | — | 68.06 | — |
| 3 | + online self-TC | 69.44 | — | 69.44 | — |
| 4 | + online pairs | 72.22 | 72.22 | 72.22 | 72.22 |
| 5 | + both online (self) | 70.83 | — | 70.83 | — |
| 6 | SFT (NLL all) | 63.89 | 63.89 | 63.89 | 63.89 |
| 7 | + offline neg-TC | — | 56.94 | — | 56.94 |
| 8 | + online neg-TC | — | 62.50 | — | 62.50 |
| 9 | + both online (neg) | — | 63.89 | — | 63.89 |

### Pearson(gen, validator) — `tc` gen vs val_score — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 44.72 | 41.90 | — | — |
| 1 | RankAlign baseline | 57.15 | 44.71 | 57.85 | 33.90 |
| 2 | + offline self-TC | 38.27 | — | 36.83 | — |
| 3 | + online self-TC | 44.73 | — | 9.95 | — |
| 4 | + online pairs | 57.99 | 15.70 | 57.68 | 31.78 |
| 5 | + both online (self) | 24.79 | — | -11.33 | — |
| 6 | SFT (NLL all) | 12.87 | 10.71 | 11.39 | -23.75 |
| 7 | + offline neg-TC | — | -37.05 | — | -24.57 |
| 8 | + online neg-TC | — | 33.88 | — | -47.55 |
| 9 | + both online (neg) | — | 63.20 | — | -40.52 |
