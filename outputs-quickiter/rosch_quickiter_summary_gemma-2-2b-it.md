# Rosch quick-iter (gemma-2-2b-it) — rosch-furniture-and-bird

**Task:** rosch-furniture-and-bird (186 items, 93/93 yes/no). **Validator:** log-odds (`--validator-log-odds`). **Metrics from `summarize_scores_file.py`, generator column variant `tc`** (TC-corrected gen score where applicable).

Train ≡ test by construction — these tables are a memorization probe; do NOT read them as cross-task generalization.

Long-form metrics: [quickiter_metrics_long.csv](quickiter_metrics_long.csv)

### Generator ROC-AUC (`tc` column)

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b-it) | 0.8737 | 0.9344 | — | — |
| 1 | RankAlign baseline | 0.8516 | 0.7499 | 0.8887 | 0.8354 |
| 2 | + offline self-TC | 0.8476 | — | 0.9363 | — |
| 3 | + online self-TC | 0.8982 | — | 0.7737 | — |
| 4 | + online pairs | 0.7810 | 0.7906 | 0.8221 | 0.7655 |
| 5 | + both online (self) | 0.8415 | — | 0.7352 | — |
| 6 | SFT (NLL all) | 0.9818 | 0.8671 | 0.9670 | 0.9618 |
| 7 | + offline neg-TC | — | 0.8673 | — | 0.9150 |
| 8 | + online neg-TC | — | 0.8998 | — | 0.7770 |
| 9 | + both online (neg) | — | 0.8084 | — | 0.7546 |

### Validator ROC-AUC (same across gen variants; shown for reference)

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b-it) | 0.9616 | 0.9616 | — | — |
| 1 | RankAlign baseline | 0.9605 | 0.9605 | 0.9605 | 0.9605 |
| 2 | + offline self-TC | 0.9615 | — | 0.9615 | — |
| 3 | + online self-TC | 0.9257 | — | 0.9257 | — |
| 4 | + online pairs | 0.8571 | 0.8571 | 0.8571 | 0.8571 |
| 5 | + both online (self) | 0.9399 | — | 0.9399 | — |
| 6 | SFT (NLL all) | 0.9690 | 0.9690 | 0.9690 | 0.9690 |
| 7 | + offline neg-TC | — | 0.8837 | — | 0.8837 |
| 8 | + online neg-TC | — | 0.9387 | — | 0.9387 |
| 9 | + both online (neg) | — | 0.8382 | — | 0.8382 |

### Validator accuracy (threshold 0)

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b-it) | 0.8817 | 0.8817 | — | — |
| 1 | RankAlign baseline | 0.8871 | 0.8871 | 0.8871 | 0.8871 |
| 2 | + offline self-TC | 0.9032 | — | 0.9032 | — |
| 3 | + online self-TC | 0.8333 | — | 0.8333 | — |
| 4 | + online pairs | 0.6828 | 0.6828 | 0.6828 | 0.6828 |
| 5 | + both online (self) | 0.5806 | — | 0.5806 | — |
| 6 | SFT (NLL all) | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| 7 | + offline neg-TC | — | 0.7634 | — | 0.7634 |
| 8 | + online neg-TC | — | 0.8710 | — | 0.8710 |
| 9 | + both online (neg) | — | 0.5000 | — | 0.5000 |

### Pearson(gen, validator) — `tc` gen vs val_score

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b-it) | 0.5648 | 0.6711 | — | — |
| 1 | RankAlign baseline | 0.6631 | 0.4172 | 0.6399 | 0.5354 |
| 2 | + offline self-TC | 0.6643 | — | 0.7864 | — |
| 3 | + online self-TC | 0.8074 | — | 0.4350 | — |
| 4 | + online pairs | 0.6815 | 0.5996 | 0.6537 | 0.5773 |
| 5 | + both online (self) | 0.7762 | — | 0.4527 | — |
| 6 | SFT (NLL all) | 0.6967 | 0.6640 | 0.6249 | 0.6249 |
| 7 | + offline neg-TC | — | 0.6573 | — | 0.7005 |
| 8 | + online neg-TC | — | 0.6261 | — | 0.5288 |
| 9 | + both online (neg) | — | 0.7065 | — | 0.5961 |
