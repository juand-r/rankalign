# Ambigqa quick-iter (gemma-2-2b) — ambigqa-train-as-test

**Task:** ambigqa-train-as-test (7,992 items, 1,998/5,994 yes/no). **Validator:** log-odds (`--validator-log-odds`). **Metrics from `summarize_scores_file.py`, generator column variant `tc`** (TC-corrected gen score where applicable).

Train ≡ test by construction — these tables are a memorization probe; do NOT read them as cross-task generalization.

**All numeric cells are raw values × 100** (i.e. ROC-AUC and accuracy are in percentage points; Pearson is in 0–100 units).

Long-form metrics: [quickiter_metrics_long_ambigqa.csv](quickiter_metrics_long_ambigqa.csv)

### Generator ROC-AUC (`tc` column) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 51.47 | 64.25 | — | — |
| 1 | RankAlign baseline | 60.91 | 61.19 | 65.73 | 71.82 |
| 2 | + offline self-TC | 54.22 | — | 57.66 | — |
| 3 | + online self-TC | 55.95 | — | 57.66 | — |
| 4 | + online pairs | 53.16 | 59.10 | 61.38 | 66.77 |
| 5 | + both online (self) | 49.15 | — | 51.53 | — |
| 6 | SFT (NLL all) | 90.23 | 89.22 | 91.26 | 92.81 |
| 7 | + offline neg-TC | — | 62.42 | — | 58.85 |
| 8 | + online neg-TC | — | 57.31 | — | 61.41 |
| 9 | + both online (neg) | — | 41.02 | — | 46.47 |

### Validator ROC-AUC (same across gen variants; shown for reference) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 56.09 | 56.09 | — | — |
| 1 | RankAlign baseline | 58.29 | 58.29 | 58.29 | 58.29 |
| 2 | + offline self-TC | 53.18 | — | 53.18 | — |
| 3 | + online self-TC | 57.84 | — | 57.84 | — |
| 4 | + online pairs | 47.84 | 47.84 | 47.84 | 47.84 |
| 5 | + both online (self) | 51.41 | — | 51.41 | — |
| 6 | SFT (NLL all) | 72.61 | 72.61 | 72.61 | 72.61 |
| 7 | + offline neg-TC | — | 55.54 | — | 55.54 |
| 8 | + online neg-TC | — | 56.99 | — | 56.99 |
| 9 | + both online (neg) | — | 48.72 | — | 48.72 |

### Validator accuracy (threshold 0) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 38.84 | 38.84 | — | — |
| 1 | RankAlign baseline | 33.00 | 33.00 | 33.00 | 33.00 |
| 2 | + offline self-TC | 52.29 | — | 52.29 | — |
| 3 | + online self-TC | 38.94 | — | 38.94 | — |
| 4 | + online pairs | 25.79 | 25.79 | 25.79 | 25.79 |
| 5 | + both online (self) | 28.68 | — | 28.68 | — |
| 6 | SFT (NLL all) | 75.00 | 75.00 | 75.00 | 75.00 |
| 7 | + offline neg-TC | — | 25.46 | — | 25.46 |
| 8 | + online neg-TC | — | 36.65 | — | 36.65 |
| 9 | + both online (neg) | — | 25.05 | — | 25.05 |

### Pearson(gen, validator) — `tc` gen vs val_score — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 42.50 | 38.92 | — | — |
| 1 | RankAlign baseline | 42.46 | 18.85 | 44.97 | 39.66 |
| 2 | + offline self-TC | 50.22 | — | 47.86 | — |
| 3 | + online self-TC | 43.93 | — | 32.54 | — |
| 4 | + online pairs | 31.72 | 23.84 | 31.00 | 28.02 |
| 5 | + both online (self) | 57.25 | — | 27.38 | — |
| 6 | SFT (NLL all) | 47.42 | 26.21 | 42.07 | 31.19 |
| 7 | + offline neg-TC | — | 19.25 | — | 32.87 |
| 8 | + online neg-TC | — | 58.23 | — | 31.42 |
| 9 | + both online (neg) | — | -6.66 | — | 0.42 |
