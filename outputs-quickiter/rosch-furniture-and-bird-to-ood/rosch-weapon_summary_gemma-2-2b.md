# Cross-task quick-iter (gemma-2-2b) — trained on rosch-furniture-and-bird, evaluated on **rosch-weapon**

**Train task:** rosch-furniture-and-bird (model checkpoints). **Eval task:** rosch-weapon (84 items, 42/42 yes/no). **Validator:** log-odds (`--validator-log-odds`). **Metrics from `summarize_scores_file.py`, generator column variant `tc`** (TC-corrected gen score where applicable).

Models were trained on a *different* rosch category set (furniture+bird) and are evaluated here as out-of-distribution. This IS a generalization read.

**All numeric cells are raw values × 100** (i.e. ROC-AUC and accuracy are in percentage points; Pearson is in 0–100 units).

Long-form metrics: [quickiter_metrics_long_crosstask.csv](quickiter_metrics_long_crosstask.csv)

### Generator ROC-AUC (`tc` column) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 77.72 | 86.99 | — | — |
| 1 | RankAlign baseline | 74.55 | 75.94 | 72.48 | 57.43 |
| 2 | + offline self-TC | 79.28 | — | 77.95 | — |
| 3 | + online self-TC | 72.31 | — | 72.14 | — |
| 4 | + online pairs | 73.78 | 71.80 | 73.61 | 53.46 |
| 5 | + both online (self) | 63.04 | — | 61.00 | — |
| 6 | SFT (NLL all) | 67.46 | 60.97 | 66.55 | 47.45 |
| 7 | + offline neg-TC | — | 36.79 | — | 44.64 |
| 8 | + online neg-TC | — | 55.70 | — | 38.38 |
| 9 | + both online (neg) | — | 79.54 | — | 51.22 |

### Validator ROC-AUC (same across gen variants; shown for reference) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 90.59 | 90.59 | — | — |
| 1 | RankAlign baseline | 88.44 | 88.44 | 88.44 | 88.44 |
| 2 | + offline self-TC | 83.84 | — | 83.84 | — |
| 3 | + online self-TC | 91.10 | — | 91.10 | — |
| 4 | + online pairs | 86.68 | 86.68 | 86.68 | 86.68 |
| 5 | + both online (self) | 86.51 | — | 86.51 | — |
| 6 | SFT (NLL all) | 86.96 | 86.96 | 86.96 | 86.96 |
| 7 | + offline neg-TC | — | 84.07 | — | 84.07 |
| 8 | + online neg-TC | — | 89.85 | — | 89.85 |
| 9 | + both online (neg) | — | 88.61 | — | 88.61 |

### Validator accuracy (threshold 0) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 83.33 | 83.33 | — | — |
| 1 | RankAlign baseline | 69.05 | 69.05 | 69.05 | 69.05 |
| 2 | + offline self-TC | 59.52 | — | 59.52 | — |
| 3 | + online self-TC | 82.14 | — | 82.14 | — |
| 4 | + online pairs | 84.52 | 84.52 | 84.52 | 84.52 |
| 5 | + both online (self) | 70.24 | — | 70.24 | — |
| 6 | SFT (NLL all) | 83.33 | 83.33 | 83.33 | 83.33 |
| 7 | + offline neg-TC | — | 61.90 | — | 61.90 |
| 8 | + online neg-TC | — | 60.71 | — | 60.71 |
| 9 | + both online (neg) | — | 73.81 | — | 73.81 |

### Pearson(gen, validator) — `tc` gen vs val_score — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 45.22 | 50.01 | — | — |
| 1 | RankAlign baseline | 71.98 | 54.70 | 70.62 | 45.61 |
| 2 | + offline self-TC | 46.80 | — | 47.05 | — |
| 3 | + online self-TC | 66.01 | — | 67.67 | — |
| 4 | + online pairs | 53.24 | 22.87 | 53.43 | 23.36 |
| 5 | + both online (self) | 47.80 | — | 34.51 | — |
| 6 | SFT (NLL all) | 37.08 | 11.75 | 36.84 | 3.39 |
| 7 | + offline neg-TC | — | -30.12 | — | 15.84 |
| 8 | + online neg-TC | — | 38.80 | — | -20.61 |
| 9 | + both online (neg) | — | 83.01 | — | 5.12 |
