# Cross-task quick-iter (gemma-2-2b) — trained on rosch-furniture-and-bird, evaluated on **rosch-clothing**

**Train task:** rosch-furniture-and-bird (model checkpoints). **Eval task:** rosch-clothing (88 items, 44/44 yes/no). **Validator:** log-odds (`--validator-log-odds`). **Metrics from `summarize_scores_file.py`, generator column variant `tc`** (TC-corrected gen score where applicable).

Models were trained on a *different* rosch category set (furniture+bird) and are evaluated here as out-of-distribution. This IS a generalization read.

**All numeric cells are raw values × 100** (i.e. ROC-AUC and accuracy are in percentage points; Pearson is in 0–100 units).

Long-form metrics: [quickiter_metrics_long_crosstask.csv](quickiter_metrics_long_crosstask.csv)

### Generator ROC-AUC (`tc` column) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 69.06 | 54.83 | — | — |
| 1 | RankAlign baseline | 81.66 | 62.32 | 80.66 | 72.03 |
| 2 | + offline self-TC | 79.29 | — | 79.03 | — |
| 3 | + online self-TC | 83.76 | — | 85.25 | — |
| 4 | + online pairs | 78.41 | 53.95 | 78.10 | 67.30 |
| 5 | + both online (self) | 65.88 | — | 68.44 | — |
| 6 | SFT (NLL all) | 79.57 | 51.94 | 79.86 | 72.34 |
| 7 | + offline neg-TC | — | 35.46 | — | 69.40 |
| 8 | + online neg-TC | — | 70.30 | — | 56.12 |
| 9 | + both online (neg) | — | 81.74 | — | 68.54 |

### Validator ROC-AUC (same across gen variants; shown for reference) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 94.37 | 94.37 | — | — |
| 1 | RankAlign baseline | 91.17 | 91.17 | 91.17 | 91.17 |
| 2 | + offline self-TC | 88.27 | — | 88.27 | — |
| 3 | + online self-TC | 92.67 | — | 92.67 | — |
| 4 | + online pairs | 92.98 | 92.98 | 92.98 | 92.98 |
| 5 | + both online (self) | 95.25 | — | 95.25 | — |
| 6 | SFT (NLL all) | 95.35 | 95.35 | 95.35 | 95.35 |
| 7 | + offline neg-TC | — | 91.27 | — | 91.27 |
| 8 | + online neg-TC | — | 89.46 | — | 89.46 |
| 9 | + both online (neg) | — | 93.75 | — | 93.75 |

### Validator accuracy (threshold 0) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 81.82 | 81.82 | — | — |
| 1 | RankAlign baseline | 67.05 | 67.05 | 67.05 | 67.05 |
| 2 | + offline self-TC | 78.41 | — | 78.41 | — |
| 3 | + online self-TC | 82.95 | — | 82.95 | — |
| 4 | + online pairs | 81.82 | 81.82 | 81.82 | 81.82 |
| 5 | + both online (self) | 67.05 | — | 67.05 | — |
| 6 | SFT (NLL all) | 87.50 | 87.50 | 87.50 | 87.50 |
| 7 | + offline neg-TC | — | 57.95 | — | 57.95 |
| 8 | + online neg-TC | — | 61.36 | — | 61.36 |
| 9 | + both online (neg) | — | 65.91 | — | 65.91 |

### Pearson(gen, validator) — `tc` gen vs val_score — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 40.81 | 17.94 | — | — |
| 1 | RankAlign baseline | 74.03 | 39.94 | 70.31 | 54.13 |
| 2 | + offline self-TC | 64.71 | — | 62.21 | — |
| 3 | + online self-TC | 68.81 | — | 63.56 | — |
| 4 | + online pairs | 60.77 | -2.63 | 56.48 | 38.10 |
| 5 | + both online (self) | 43.49 | — | 38.35 | — |
| 6 | SFT (NLL all) | 46.53 | 6.72 | 46.05 | 26.71 |
| 7 | + offline neg-TC | — | -23.49 | — | 36.84 |
| 8 | + online neg-TC | — | 52.36 | — | 5.41 |
| 9 | + both online (neg) | — | 75.24 | — | 24.72 |
