# Ambigqa quick-iter (gemma-2-2b-it) — ambigqa-train-as-test

**Task:** ambigqa-train-as-test (7,992 items, 1,998/5,994 yes/no). **Validator:** log-odds (`--validator-log-odds`). **Metrics from `summarize_scores_file.py`, generator column variant `tc`** (TC-corrected gen score where applicable).

Train ≡ test by construction — these tables are a memorization probe; do NOT read them as cross-task generalization.

**All numeric cells are raw values × 100** (i.e. ROC-AUC and accuracy are in percentage points; Pearson is in 0–100 units).

Long-form metrics: [quickiter_metrics_long_ambigqa.csv](quickiter_metrics_long_ambigqa.csv)

### Generator ROC-AUC (`tc` column) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b-it) | 52.14 | 66.40 | — | — |
| 1 | RankAlign baseline | 63.80 | 57.09 | 73.52 | 77.07 |
| 2 | + offline self-TC | 54.65 | — | 63.60 | — |
| 3 | + online self-TC | 67.23 | — | 63.56 | — |
| 4 | + online pairs | 64.00 | 58.22 | 76.07 | 78.48 |
| 5 | + both online (self) | 53.83 | — | 61.41 | — |
| 6 | SFT (NLL all) | 88.11 | 75.10 | 93.80 | 93.68 |
| 7 | + offline neg-TC | — | 61.97 | — | 64.34 |
| 8 | + online neg-TC | — | 65.32 | — | 69.75 |
| 9 | + both online (neg) | — | 63.16 | — | 65.62 |

### Validator ROC-AUC (same across gen variants; shown for reference) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b-it) | 58.84 | 58.84 | — | — |
| 1 | RankAlign baseline | 62.15 | 62.15 | 62.15 | 62.15 |
| 2 | + offline self-TC | 57.09 | — | 57.09 | — |
| 3 | + online self-TC | 52.71 | — | 52.71 | — |
| 4 | + online pairs | 63.74 | 63.74 | 63.74 | 63.74 |
| 5 | + both online (self) | 47.42 | — | 47.42 | — |
| 6 | SFT (NLL all) | 57.68 | 57.68 | 57.68 | 57.68 |
| 7 | + offline neg-TC | — | 54.98 | — | 54.98 |
| 8 | + online neg-TC | — | 54.54 | — | 54.54 |
| 9 | + both online (neg) | — | 48.63 | — | 48.63 |

### Validator accuracy (threshold 0) — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b-it) | 61.11 | 61.11 | — | — |
| 1 | RankAlign baseline | 74.87 | 74.87 | 74.87 | 74.87 |
| 2 | + offline self-TC | 57.53 | — | 57.53 | — |
| 3 | + online self-TC | 75.00 | — | 75.00 | — |
| 4 | + online pairs | 55.07 | 55.07 | 55.07 | 55.07 |
| 5 | + both online (self) | 75.00 | — | 75.00 | — |
| 6 | SFT (NLL all) | 58.88 | 58.88 | 58.88 | 58.88 |
| 7 | + offline neg-TC | — | 51.14 | — | 51.14 |
| 8 | + online neg-TC | — | 65.47 | — | 65.47 |
| 9 | + both online (neg) | — | 25.00 | — | 25.00 |

### Pearson(gen, validator) — `tc` gen vs val_score — values × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b-it) | 38.76 | 27.97 | — | — |
| 1 | RankAlign baseline | 42.34 | 7.62 | 39.84 | 30.66 |
| 2 | + offline self-TC | 23.91 | — | 22.70 | — |
| 3 | + online self-TC | 27.25 | — | 13.20 | — |
| 4 | + online pairs | 30.80 | 29.35 | 35.11 | 34.40 |
| 5 | + both online (self) | 23.75 | — | 6.64 | — |
| 6 | SFT (NLL all) | 7.28 | 19.74 | 9.34 | 0.92 |
| 7 | + offline neg-TC | — | 47.84 | — | 34.35 |
| 8 | + online neg-TC | — | 32.95 | — | 3.69 |
| 9 | + both online (neg) | — | 18.13 | — | 2.76 |
