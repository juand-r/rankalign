# Cross-task quick-iter (gemma-2-2b) — trained on rosch-furniture-and-bird, evaluated on **rosch-fruit**

**Train task:** rosch-furniture-and-bird (model checkpoints). **Eval task:** rosch-fruit (84 items, 42/42 yes/no). **Validator:** log-odds (`--validator-log-odds`). **Metrics from `summarize_scores_file.py`, generator column variant `tc`** (TC-corrected gen score where applicable).

Models were trained on a *different* rosch category set (furniture+bird) and are evaluated here as out-of-distribution. This IS a generalization read.

**All numeric cells are raw values × 100** (i.e. ROC-AUC and accuracy are in percentage points; Pearson is in 0–100 units).

Long-form metrics: [quickiter_metrics_long_crosstask.csv](quickiter_metrics_long_crosstask.csv)

### Generator ROC-AUC (`tc` column) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 76.73 | 84.98 | — | — |
| 1 | RankAlign baseline | 85.71 | 80.61 | 86.00 | 80.95 |
| 2 | + offline self-TC | 77.35 | — | 74.18 | — |
| 3 | + online self-TC | 86.62 | — | 72.68 | — |
| 4 | + online pairs | 85.66 | 53.32 | 86.88 | 79.56 |
| 5 | + both online (self) | 78.29 | — | 66.41 | — |
| 6 | SFT (NLL all) | 68.25 | 75.60 | 68.59 | 55.90 |
| 7 | + offline neg-TC | — | 22.05 | — | 72.48 |
| 8 | + online neg-TC | — | 77.55 | — | 26.36 |
| 9 | + both online (neg) | — | 92.94 | — | 53.37 |

### Validator ROC-AUC (same across gen variants; shown for reference) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 91.72 | 91.72 | — | — |
| 1 | RankAlign baseline | 95.58 | 95.58 | 95.58 | 95.58 |
| 2 | + offline self-TC | 68.71 | — | 68.71 | — |
| 3 | + online self-TC | 91.61 | — | 91.61 | — |
| 4 | + online pairs | 90.93 | 90.93 | 90.93 | 90.93 |
| 5 | + both online (self) | 93.25 | — | 93.25 | — |
| 6 | SFT (NLL all) | 90.93 | 90.93 | 90.93 | 90.93 |
| 7 | + offline neg-TC | — | 88.89 | — | 88.89 |
| 8 | + online neg-TC | — | 88.78 | — | 88.78 |
| 9 | + both online (neg) | — | 91.38 | — | 91.38 |

### Validator accuracy (threshold 0) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 83.33 | 83.33 | — | — |
| 1 | RankAlign baseline | 88.10 | 88.10 | 88.10 | 88.10 |
| 2 | + offline self-TC | 58.33 | — | 58.33 | — |
| 3 | + online self-TC | 69.05 | — | 69.05 | — |
| 4 | + online pairs | 82.14 | 82.14 | 82.14 | 82.14 |
| 5 | + both online (self) | 82.14 | — | 82.14 | — |
| 6 | SFT (NLL all) | 84.52 | 84.52 | 84.52 | 84.52 |
| 7 | + offline neg-TC | — | 84.52 | — | 84.52 |
| 8 | + online neg-TC | — | 51.19 | — | 51.19 |
| 9 | + both online (neg) | — | 72.62 | — | 72.62 |

### Pearson(gen, validator) — `tc` gen vs val_score — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 69.00 | 53.72 | — | — |
| 1 | RankAlign baseline | 64.68 | 54.68 | 64.57 | 52.24 |
| 2 | + offline self-TC | 46.95 | — | 45.04 | — |
| 3 | + online self-TC | 72.26 | — | 61.80 | — |
| 4 | + online pairs | 73.55 | 10.82 | 73.04 | 34.79 |
| 5 | + both online (self) | 63.60 | — | 45.86 | — |
| 6 | SFT (NLL all) | 46.90 | 27.55 | 47.01 | 5.21 |
| 7 | + offline neg-TC | — | -42.74 | — | 47.17 |
| 8 | + online neg-TC | — | 47.43 | — | -18.05 |
| 9 | + both online (neg) | — | 83.45 | — | 7.09 |
