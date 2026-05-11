# Ambigqa quick-iter (gemma-2-2b-it) — ambigqa-train-as-test

**Task:** ambigqa-train-as-test (7,992 items, 1,998/5,994 yes/no). **Validator:** log-odds (`--validator-log-odds`). **Metrics from `summarize_scores_file.py`, generator column variant `tc`** (TC-corrected gen score where applicable).

Train ≡ test by construction — these tables are a memorization probe; do NOT read them as cross-task generalization.

Long-form metrics: [quickiter_metrics_long_ambigqa.csv](quickiter_metrics_long_ambigqa.csv)

### Generator ROC-AUC (`tc` column)

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b-it) | 0.5214 | 0.6640 | — | — |
| 1 | RankAlign baseline | 0.6380 | 0.5709 | 0.7352 | 0.7707 |
| 2 | + offline self-TC | 0.5465 | — | 0.6360 | — |
| 3 | + online self-TC | 0.6723 | — | 0.6356 | — |
| 4 | + online pairs | 0.6400 | 0.5822 | 0.7607 | 0.7848 |
| 5 | + both online (self) | 0.5383 | — | 0.6141 | — |
| 6 | SFT (NLL all) | 0.8811 | 0.7510 | 0.9380 | 0.9368 |
| 7 | + offline neg-TC | — | 0.6197 | — | 0.6434 |
| 8 | + online neg-TC | — | 0.6532 | — | 0.6975 |
| 9 | + both online (neg) | — | 0.6316 | — | 0.6562 |

### Validator ROC-AUC (same across gen variants; shown for reference)

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b-it) | 0.5884 | 0.5884 | — | — |
| 1 | RankAlign baseline | 0.6215 | 0.6215 | 0.6215 | 0.6215 |
| 2 | + offline self-TC | 0.5709 | — | 0.5709 | — |
| 3 | + online self-TC | 0.5271 | — | 0.5271 | — |
| 4 | + online pairs | 0.6374 | 0.6374 | 0.6374 | 0.6374 |
| 5 | + both online (self) | 0.4742 | — | 0.4742 | — |
| 6 | SFT (NLL all) | 0.5768 | 0.5768 | 0.5768 | 0.5768 |
| 7 | + offline neg-TC | — | 0.5498 | — | 0.5498 |
| 8 | + online neg-TC | — | 0.5454 | — | 0.5454 |
| 9 | + both online (neg) | — | 0.4863 | — | 0.4863 |

### Validator accuracy (threshold 0)

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b-it) | 0.6111 | 0.6111 | — | — |
| 1 | RankAlign baseline | 0.7487 | 0.7487 | 0.7487 | 0.7487 |
| 2 | + offline self-TC | 0.5753 | — | 0.5753 | — |
| 3 | + online self-TC | 0.7500 | — | 0.7500 | — |
| 4 | + online pairs | 0.5507 | 0.5507 | 0.5507 | 0.5507 |
| 5 | + both online (self) | 0.7500 | — | 0.7500 | — |
| 6 | SFT (NLL all) | 0.5888 | 0.5888 | 0.5888 | 0.5888 |
| 7 | + offline neg-TC | — | 0.5114 | — | 0.5114 |
| 8 | + online neg-TC | — | 0.6547 | — | 0.6547 |
| 9 | + both online (neg) | — | 0.2500 | — | 0.2500 |

### Pearson(gen, validator) — `tc` gen vs val_score

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b-it) | 0.3876 | 0.2797 | — | — |
| 1 | RankAlign baseline | 0.4234 | 0.0762 | 0.3984 | 0.3066 |
| 2 | + offline self-TC | 0.2391 | — | 0.2270 | — |
| 3 | + online self-TC | 0.2725 | — | 0.1320 | — |
| 4 | + online pairs | 0.3080 | 0.2935 | 0.3511 | 0.3440 |
| 5 | + both online (self) | 0.2375 | — | 0.0664 | — |
| 6 | SFT (NLL all) | 0.0728 | 0.1974 | 0.0934 | 0.0092 |
| 7 | + offline neg-TC | — | 0.4784 | — | 0.3435 |
| 8 | + online neg-TC | — | 0.3295 | — | 0.0369 |
| 9 | + both online (neg) | — | 0.1813 | — | 0.0276 |
