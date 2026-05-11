# Cross-task quick-iter (gemma-2-2b) — trained on rosch-furniture-and-bird, evaluated on **rosch-vehicle**

**Train task:** rosch-furniture-and-bird (model checkpoints). **Eval task:** rosch-vehicle (82 items, 41/41 yes/no). **Validator:** log-odds (`--validator-log-odds`). **Metrics from `summarize_scores_file.py`, generator column variant `tc`** (TC-corrected gen score where applicable).

Models were trained on a *different* rosch category set (furniture+bird) and are evaluated here as out-of-distribution. This IS a generalization read.

**All numeric cells are raw values × 100** (i.e. ROC-AUC and accuracy are in percentage points; Pearson is in 0–100 units).

Long-form metrics: [quickiter_metrics_long_crosstask.csv](quickiter_metrics_long_crosstask.csv)

### Generator ROC-AUC (`tc` column) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 85.69 | 77.04 | — | — |
| 1 | RankAlign baseline | 84.65 | 51.10 | 82.93 | 61.18 |
| 2 | + offline self-TC | 87.06 | — | 86.32 | — |
| 3 | + online self-TC | 79.57 | — | 81.86 | — |
| 4 | + online pairs | 82.81 | 51.96 | 82.03 | 55.15 |
| 5 | + both online (self) | 75.94 | — | 74.18 | — |
| 6 | SFT (NLL all) | 68.71 | 60.53 | 68.02 | 44.97 |
| 7 | + offline neg-TC | — | 42.68 | — | 59.40 |
| 8 | + online neg-TC | — | 63.24 | — | 28.11 |
| 9 | + both online (neg) | — | 84.47 | — | 53.39 |

### Validator ROC-AUC (same across gen variants; shown for reference) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 96.61 | 96.61 | — | — |
| 1 | RankAlign baseline | 93.46 | 93.46 | 93.46 | 93.46 |
| 2 | + offline self-TC | 84.77 | — | 84.77 | — |
| 3 | + online self-TC | 90.66 | — | 90.66 | — |
| 4 | + online pairs | 94.82 | 94.82 | 94.82 | 94.82 |
| 5 | + both online (self) | 90.54 | — | 90.54 | — |
| 6 | SFT (NLL all) | 97.74 | 97.74 | 97.74 | 97.74 |
| 7 | + offline neg-TC | — | 93.16 | — | 93.16 |
| 8 | + online neg-TC | — | 85.13 | — | 85.13 |
| 9 | + both online (neg) | — | 92.33 | — | 92.33 |

### Validator accuracy (threshold 0) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 82.93 | 82.93 | — | — |
| 1 | RankAlign baseline | 68.29 | 68.29 | 68.29 | 68.29 |
| 2 | + offline self-TC | 74.39 | — | 74.39 | — |
| 3 | + online self-TC | 74.39 | — | 74.39 | — |
| 4 | + online pairs | 87.80 | 87.80 | 87.80 | 87.80 |
| 5 | + both online (self) | 56.10 | — | 56.10 | — |
| 6 | SFT (NLL all) | 92.68 | 92.68 | 92.68 | 92.68 |
| 7 | + offline neg-TC | — | 52.44 | — | 52.44 |
| 8 | + online neg-TC | — | 51.22 | — | 51.22 |
| 9 | + both online (neg) | — | 78.05 | — | 78.05 |

### Pearson(gen, validator) — `tc` gen vs val_score — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 55.03 | 32.30 | — | — |
| 1 | RankAlign baseline | 47.96 | 33.50 | 42.27 | 21.24 |
| 2 | + offline self-TC | 51.01 | — | 49.17 | — |
| 3 | + online self-TC | 40.51 | — | 30.60 | — |
| 4 | + online pairs | 40.50 | -17.36 | 37.09 | -8.01 |
| 5 | + both online (self) | 38.54 | — | 28.42 | — |
| 6 | SFT (NLL all) | 9.67 | 15.27 | 7.36 | -26.56 |
| 7 | + offline neg-TC | — | -1.36 | — | 17.01 |
| 8 | + online neg-TC | — | 6.33 | — | -34.29 |
| 9 | + both online (neg) | — | 70.61 | — | -4.90 |
