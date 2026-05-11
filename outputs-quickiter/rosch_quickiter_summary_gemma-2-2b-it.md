# Rosch quick-iter (gemma-2-2b-it) — rosch-furniture-and-bird

**Task:** rosch-furniture-and-bird (186 items, 93/93 yes/no). **Validator:** log-odds (`--validator-log-odds`). **Metrics from `summarize_scores_file.py`, generator column variant `tc`** (TC-corrected gen score where applicable).

Train ≡ test by construction — these tables are a memorization probe; do NOT read them as cross-task generalization.

**All numeric cells are raw values × 100** (i.e. ROC-AUC and accuracy are in percentage points; Pearson is in 0–100 units).

Long-form metrics: [quickiter_metrics_long.csv](quickiter_metrics_long.csv)

### Generator ROC-AUC (`tc` column) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b-it) | 87.37 | 93.44 | — | — |
| 1 | RankAlign baseline | 85.16 | 74.99 | 88.87 | 83.54 |
| 2 | + offline self-TC | 84.76 | — | 93.63 | — |
| 3 | + online self-TC | 89.82 | — | 77.37 | — |
| 4 | + online pairs | 78.10 | 79.06 | 82.21 | 76.55 |
| 5 | + both online (self) | 84.15 | — | 73.52 | — |
| 6 | SFT (NLL all) | 98.18 | 86.71 | 96.70 | 96.18 |
| 7 | + offline neg-TC | — | 86.73 | — | 91.50 |
| 8 | + online neg-TC | — | 89.98 | — | 77.70 |
| 9 | + both online (neg) | — | 80.84 | — | 75.46 |

### Validator ROC-AUC (same across gen variants; shown for reference) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b-it) | 96.16 | 96.16 | — | — |
| 1 | RankAlign baseline | 96.05 | 96.05 | 96.05 | 96.05 |
| 2 | + offline self-TC | 96.15 | — | 96.15 | — |
| 3 | + online self-TC | 92.57 | — | 92.57 | — |
| 4 | + online pairs | 85.71 | 85.71 | 85.71 | 85.71 |
| 5 | + both online (self) | 93.99 | — | 93.99 | — |
| 6 | SFT (NLL all) | 96.90 | 96.90 | 96.90 | 96.90 |
| 7 | + offline neg-TC | — | 88.37 | — | 88.37 |
| 8 | + online neg-TC | — | 93.87 | — | 93.87 |
| 9 | + both online (neg) | — | 83.82 | — | 83.82 |

### Validator accuracy (threshold 0) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b-it) | 88.17 | 88.17 | — | — |
| 1 | RankAlign baseline | 88.71 | 88.71 | 88.71 | 88.71 |
| 2 | + offline self-TC | 90.32 | — | 90.32 | — |
| 3 | + online self-TC | 83.33 | — | 83.33 | — |
| 4 | + online pairs | 68.28 | 68.28 | 68.28 | 68.28 |
| 5 | + both online (self) | 58.06 | — | 58.06 | — |
| 6 | SFT (NLL all) | 50.00 | 50.00 | 50.00 | 50.00 |
| 7 | + offline neg-TC | — | 76.34 | — | 76.34 |
| 8 | + online neg-TC | — | 87.10 | — | 87.10 |
| 9 | + both online (neg) | — | 50.00 | — | 50.00 |

### Pearson(gen, validator) — `tc` gen vs val_score — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b-it) | 56.48 | 67.11 | — | — |
| 1 | RankAlign baseline | 66.31 | 41.72 | 63.99 | 53.54 |
| 2 | + offline self-TC | 66.43 | — | 78.64 | — |
| 3 | + online self-TC | 80.74 | — | 43.50 | — |
| 4 | + online pairs | 68.15 | 59.96 | 65.37 | 57.73 |
| 5 | + both online (self) | 77.62 | — | 45.27 | — |
| 6 | SFT (NLL all) | 69.67 | 66.40 | 62.49 | 62.49 |
| 7 | + offline neg-TC | — | 65.73 | — | 70.05 |
| 8 | + online neg-TC | — | 62.61 | — | 52.88 |
| 9 | + both online (neg) | — | 70.65 | — | 59.61 |
