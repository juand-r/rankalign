# Cross-task quick-iter (gemma-2-2b) — trained on rosch-furniture-and-bird, evaluated on **rosch-sport**

**Train task:** rosch-furniture-and-bird (model checkpoints). **Eval task:** rosch-sport (92 items, 46/46 yes/no). **Validator:** log-odds (`--validator-log-odds`). **Metrics from `summarize_scores_file.py`, generator column variant `tc`** (TC-corrected gen score where applicable).

Models were trained on a *different* rosch category set (furniture+bird) and are evaluated here as out-of-distribution. This IS a generalization read.

**All numeric cells are raw values × 100** (i.e. ROC-AUC and accuracy are in percentage points; Pearson is in 0–100 units).

Long-form metrics: [quickiter_metrics_long_crosstask.csv](quickiter_metrics_long_crosstask.csv)

### Generator ROC-AUC (`tc` column) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 80.41 | 78.69 | — | — |
| 1 | RankAlign baseline | 78.50 | 51.39 | 80.01 | 59.52 |
| 2 | + offline self-TC | 83.55 | — | 82.35 | — |
| 3 | + online self-TC | 79.66 | — | 77.39 | — |
| 4 | + online pairs | 86.86 | 78.19 | 86.39 | 67.91 |
| 5 | + both online (self) | 64.22 | — | 61.60 | — |
| 6 | SFT (NLL all) | 74.60 | 19.59 | 73.75 | 60.59 |
| 7 | + offline neg-TC | — | 31.62 | — | 58.08 |
| 8 | + online neg-TC | — | 62.74 | — | 51.35 |
| 9 | + both online (neg) | — | 87.78 | — | 65.10 |

### Validator ROC-AUC (same across gen variants; shown for reference) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 86.29 | 86.29 | — | — |
| 1 | RankAlign baseline | 87.29 | 87.29 | 87.29 | 87.29 |
| 2 | + offline self-TC | 83.36 | — | 83.36 | — |
| 3 | + online self-TC | 86.29 | — | 86.29 | — |
| 4 | + online pairs | 85.54 | 85.54 | 85.54 | 85.54 |
| 5 | + both online (self) | 86.06 | — | 86.06 | — |
| 6 | SFT (NLL all) | 87.95 | 87.95 | 87.95 | 87.95 |
| 7 | + offline neg-TC | — | 78.17 | — | 78.17 |
| 8 | + online neg-TC | — | 88.00 | — | 88.00 |
| 9 | + both online (neg) | — | 91.59 | — | 91.59 |

### Validator accuracy (threshold 0) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 77.17 | 77.17 | — | — |
| 1 | RankAlign baseline | 59.78 | 59.78 | 59.78 | 59.78 |
| 2 | + offline self-TC | 50.00 | — | 50.00 | — |
| 3 | + online self-TC | 73.91 | — | 73.91 | — |
| 4 | + online pairs | 77.17 | 77.17 | 77.17 | 77.17 |
| 5 | + both online (self) | 52.17 | — | 52.17 | — |
| 6 | SFT (NLL all) | 78.26 | 78.26 | 78.26 | 78.26 |
| 7 | + offline neg-TC | — | 47.83 | — | 47.83 |
| 8 | + online neg-TC | — | 50.00 | — | 50.00 |
| 9 | + both online (neg) | — | 67.39 | — | 67.39 |

### Pearson(gen, validator) — `tc` gen vs val_score — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 54.98 | 38.56 | — | — |
| 1 | RankAlign baseline | 54.88 | 8.53 | 55.43 | 19.87 |
| 2 | + offline self-TC | 60.46 | — | 56.89 | — |
| 3 | + online self-TC | 61.19 | — | 45.82 | — |
| 4 | + online pairs | 57.48 | 47.28 | 56.05 | 20.61 |
| 5 | + both online (self) | 36.79 | — | 26.59 | — |
| 6 | SFT (NLL all) | 36.26 | -20.40 | 33.66 | 4.72 |
| 7 | + offline neg-TC | — | -26.81 | — | -3.72 |
| 8 | + online neg-TC | — | 25.03 | — | -10.99 |
| 9 | + both online (neg) | — | 77.09 | — | 14.56 |
