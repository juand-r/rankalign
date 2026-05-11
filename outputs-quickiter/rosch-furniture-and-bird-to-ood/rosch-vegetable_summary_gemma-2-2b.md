# Cross-task quick-iter (gemma-2-2b) — trained on rosch-furniture-and-bird, evaluated on **rosch-vegetable**

**Train task:** rosch-furniture-and-bird (model checkpoints). **Eval task:** rosch-vegetable (86 items, 43/43 yes/no). **Validator:** log-odds (`--validator-log-odds`). **Metrics from `summarize_scores_file.py`, generator column variant `tc`** (TC-corrected gen score where applicable).

Models were trained on a *different* rosch category set (furniture+bird) and are evaluated here as out-of-distribution. This IS a generalization read.

**All numeric cells are raw values × 100** (i.e. ROC-AUC and accuracy are in percentage points; Pearson is in 0–100 units).

Long-form metrics: [quickiter_metrics_long_crosstask.csv](quickiter_metrics_long_crosstask.csv)

### Generator ROC-AUC (`tc` column) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 82.77 | 92.10 | — | — |
| 1 | RankAlign baseline | 75.09 | 92.08 | 72.96 | 60.06 |
| 2 | + offline self-TC | 72.31 | — | 69.47 | — |
| 3 | + online self-TC | 83.21 | — | 75.80 | — |
| 4 | + online pairs | 80.07 | 88.37 | 80.18 | 70.25 |
| 5 | + both online (self) | 61.55 | — | 58.49 | — |
| 6 | SFT (NLL all) | 74.58 | 91.32 | 74.01 | 59.14 |
| 7 | + offline neg-TC | — | 41.35 | — | 58.52 |
| 8 | + online neg-TC | — | 67.09 | — | 32.21 |
| 9 | + both online (neg) | — | 79.77 | — | 45.92 |

### Validator ROC-AUC (same across gen variants; shown for reference) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 94.75 | 94.75 | — | — |
| 1 | RankAlign baseline | 95.57 | 95.57 | 95.57 | 95.57 |
| 2 | + offline self-TC | 92.48 | — | 92.48 | — |
| 3 | + online self-TC | 94.97 | — | 94.97 | — |
| 4 | + online pairs | 94.48 | 94.48 | 94.48 | 94.48 |
| 5 | + both online (self) | 96.32 | — | 96.32 | — |
| 6 | SFT (NLL all) | 95.78 | 95.78 | 95.78 | 95.78 |
| 7 | + offline neg-TC | — | 94.21 | — | 94.21 |
| 8 | + online neg-TC | — | 89.29 | — | 89.29 |
| 9 | + both online (neg) | — | 95.67 | — | 95.67 |

### Validator accuracy (threshold 0) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 84.88 | 84.88 | — | — |
| 1 | RankAlign baseline | 77.91 | 77.91 | 77.91 | 77.91 |
| 2 | + offline self-TC | 79.07 | — | 79.07 | — |
| 3 | + online self-TC | 89.53 | — | 89.53 | — |
| 4 | + online pairs | 84.88 | 84.88 | 84.88 | 84.88 |
| 5 | + both online (self) | 81.40 | — | 81.40 | — |
| 6 | SFT (NLL all) | 86.05 | 86.05 | 86.05 | 86.05 |
| 7 | + offline neg-TC | — | 58.14 | — | 58.14 |
| 8 | + online neg-TC | — | 51.16 | — | 51.16 |
| 9 | + both online (neg) | — | 75.58 | — | 75.58 |

### Pearson(gen, validator) — `tc` gen vs val_score — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 65.82 | 60.35 | — | — |
| 1 | RankAlign baseline | 60.74 | 66.52 | 55.16 | 16.03 |
| 2 | + offline self-TC | 59.53 | — | 52.28 | — |
| 3 | + online self-TC | 67.40 | — | 53.23 | — |
| 4 | + online pairs | 69.64 | 49.13 | 65.90 | 28.83 |
| 5 | + both online (self) | 40.80 | — | 19.29 | — |
| 6 | SFT (NLL all) | 53.05 | 56.93 | 52.01 | 23.56 |
| 7 | + offline neg-TC | — | -20.61 | — | 11.75 |
| 8 | + online neg-TC | — | 42.50 | — | -26.88 |
| 9 | + both online (neg) | — | 73.62 | — | 0.77 |
