# Rosch quick-iter (gemma-2-2b) — rosch-furniture-and-bird

**Task:** combined furniture + bird (186 items, 93/93 yes/no). **Validator:** log-odds (`--validator-log-odds`). **Metrics from `summarize_scores_file.py`, generator column variant `tc`** (TC-corrected gen score where applicable).

Spearman omitted per your usual reporting preference.

Long-form metrics: [quickiter_metrics_long.csv](quickiter_metrics_long.csv)

### Generator ROC-AUC (`tc` column)

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 0.7466 | 0.7375 | — | — |
| 1 | RankAlign baseline | 0.8475 | 0.8644 | 0.8681 | 0.8046 |
| 2 | + offline self-TC | 0.8776 | — | 0.9510 | — |
| 3 | + online self-TC | 0.9575 | — | 0.9572 | — |
| 4 | + online pairs | 0.8544 | 0.7789 | 0.8299 | 0.7585 |
| 5 | + both online (self) | 0.9202 | — | 0.8703 | — |
| 6 | SFT (NLL all) | 0.9953 | 0.8099 | 0.9958 | 0.9895 |
| 7 | + offline neg-TC | — | 0.6108 | — | 0.9283 |
| 8 | + online neg-TC | — | 0.8873 | — | 0.7073 |
| 9 | + both online (neg) | — | 0.9076 | — | 0.7459 |

### Validator ROC-AUC (same across gen variants for a file; shown for reference)

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 0.9605 | 0.9605 | — | — |
| 1 | RankAlign baseline | 0.9109 | 0.9109 | 0.9109 | 0.9109 |
| 2 | + offline self-TC | 0.8207 | — | 0.8207 | — |
| 3 | + online self-TC | 0.9372 | — | 0.9372 | — |
| 4 | + online pairs | 0.9542 | 0.9542 | 0.9542 | 0.9542 |
| 5 | + both online (self) | 0.8935 | — | 0.8935 | — |
| 6 | SFT (NLL all) | 0.9322 | 0.9322 | 0.9322 | 0.9322 |
| 7 | + offline neg-TC | — | 0.8380 | — | 0.8380 |
| 8 | + online neg-TC | — | 0.8668 | — | 0.8668 |
| 9 | + both online (neg) | — | 0.9448 | — | 0.9448 |

### Validator accuracy (threshold 0)

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 0.8763 | 0.8763 | — | — |
| 1 | RankAlign baseline | 0.7634 | 0.7634 | 0.7634 | 0.7634 |
| 2 | + offline self-TC | 0.6075 | — | 0.6075 | — |
| 3 | + online self-TC | 0.8763 | — | 0.8763 | — |
| 4 | + online pairs | 0.8871 | 0.8871 | 0.8871 | 0.8871 |
| 5 | + both online (self) | 0.7312 | — | 0.7312 | — |
| 6 | SFT (NLL all) | 0.8763 | 0.8763 | 0.8763 | 0.8763 |
| 7 | + offline neg-TC | — | 0.6613 | — | 0.6613 |
| 8 | + online neg-TC | — | 0.5806 | — | 0.5806 |
| 9 | + both online (neg) | — | 0.7151 | — | 0.7151 |

### Pearson(gen, validator) — `tc` gen vs val_score

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 0.5010 | 0.3464 | — | — |
| 1 | RankAlign baseline | 0.6413 | 0.5919 | 0.6017 | 0.5236 |
| 2 | + offline self-TC | 0.6857 | — | 0.5921 | — |
| 3 | + online self-TC | 0.8818 | — | 0.7871 | — |
| 4 | + online pairs | 0.7340 | 0.4261 | 0.6540 | 0.5210 |
| 5 | + both online (self) | 0.6975 | — | 0.5570 | — |
| 6 | SFT (NLL all) | 0.6103 | 0.3388 | 0.6288 | 0.5760 |
| 7 | + offline neg-TC | — | -0.0948 | — | 0.4655 |
| 8 | + online neg-TC | — | 0.5088 | — | 0.1830 |
| 9 | + both online (neg) | — | 0.8866 | — | 0.5281 |
