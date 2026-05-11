# Rosch quick-iter (gemma-2-2b) — rosch-furniture-and-bird

**Task:** combined furniture + bird (186 items, 93/93 yes/no). **Validator:** log-odds (`--validator-log-odds`). **Metrics from `summarize_scores_file.py`, generator column variant `tc`** (TC-corrected gen score where applicable).

Spearman omitted per your usual reporting preference.

**All numeric cells are raw values × 100** (i.e. ROC-AUC and accuracy are in percentage points; Pearson is in 0–100 units).

Long-form metrics: [quickiter_metrics_long.csv](quickiter_metrics_long.csv)

### Generator ROC-AUC (`tc` column)

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 74.66 | 73.75 | — | — |
| 1 | RankAlign baseline | 84.75 | 86.44 | 86.81 | 80.46 |
| 2 | + offline self-TC | 87.76 | — | 95.10 | — |
| 3 | + online self-TC | 95.75 | — | 95.72 | — |
| 4 | + online pairs | 85.44 | 77.89 | 82.99 | 75.85 |
| 5 | + both online (self) | 92.02 | — | 87.03 | — |
| 6 | SFT (NLL all) | 99.53 | 80.99 | 99.58 | 98.95 |
| 7 | + offline neg-TC | — | 61.08 | — | 92.83 |
| 8 | + online neg-TC | — | 88.73 | — | 70.73 |
| 9 | + both online (neg) | — | 90.76 | — | 74.59 |

### Validator ROC-AUC (same across gen variants for a file; shown for reference)

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 96.05 | 96.05 | — | — |
| 1 | RankAlign baseline | 91.09 | 91.09 | 91.09 | 91.09 |
| 2 | + offline self-TC | 82.07 | — | 82.07 | — |
| 3 | + online self-TC | 93.72 | — | 93.72 | — |
| 4 | + online pairs | 95.42 | 95.42 | 95.42 | 95.42 |
| 5 | + both online (self) | 89.35 | — | 89.35 | — |
| 6 | SFT (NLL all) | 93.22 | 93.22 | 93.22 | 93.22 |
| 7 | + offline neg-TC | — | 83.80 | — | 83.80 |
| 8 | + online neg-TC | — | 86.68 | — | 86.68 |
| 9 | + both online (neg) | — | 94.48 | — | 94.48 |

### Validator accuracy (threshold 0)

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 87.63 | 87.63 | — | — |
| 1 | RankAlign baseline | 76.34 | 76.34 | 76.34 | 76.34 |
| 2 | + offline self-TC | 60.75 | — | 60.75 | — |
| 3 | + online self-TC | 87.63 | — | 87.63 | — |
| 4 | + online pairs | 88.71 | 88.71 | 88.71 | 88.71 |
| 5 | + both online (self) | 73.12 | — | 73.12 | — |
| 6 | SFT (NLL all) | 87.63 | 87.63 | 87.63 | 87.63 |
| 7 | + offline neg-TC | — | 66.13 | — | 66.13 |
| 8 | + online neg-TC | — | 58.06 | — | 58.06 |
| 9 | + both online (neg) | — | 71.51 | — | 71.51 |

### Pearson(gen, validator) — `tc` gen vs val_score

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 50.10 | 34.64 | — | — |
| 1 | RankAlign baseline | 64.13 | 59.19 | 60.17 | 52.36 |
| 2 | + offline self-TC | 68.57 | — | 59.21 | — |
| 3 | + online self-TC | 88.18 | — | 78.71 | — |
| 4 | + online pairs | 73.40 | 42.61 | 65.40 | 52.10 |
| 5 | + both online (self) | 69.75 | — | 55.70 | — |
| 6 | SFT (NLL all) | 61.03 | 33.88 | 62.88 | 57.60 |
| 7 | + offline neg-TC | — | -9.48 | — | 46.55 |
| 8 | + online neg-TC | — | 50.88 | — | 18.30 |
| 9 | + both online (neg) | — | 88.66 | — | 52.81 |
