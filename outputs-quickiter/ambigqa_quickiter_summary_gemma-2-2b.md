# Ambigqa quick-iter (gemma-2-2b) — ambigqa-train-as-test

**Task:** ambigqa-train-as-test (7,992 items, 1,998/5,994 yes/no). **Validator:** log-odds (`--validator-log-odds`). **Metrics from `summarize_scores_file.py`, generator column variant `tc`** (TC-corrected gen score where applicable).

Train ≡ test by construction — these tables are a memorization probe; do NOT read them as cross-task generalization.

Long-form metrics: [quickiter_metrics_long_ambigqa.csv](quickiter_metrics_long_ambigqa.csv)

### Generator ROC-AUC (`tc` column)

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 0.5147 | 0.6425 | — | — |
| 1 | RankAlign baseline | 0.6091 | 0.6119 | 0.6573 | 0.7182 |
| 2 | + offline self-TC | 0.5422 | — | 0.5766 | — |
| 3 | + online self-TC | 0.5595 | — | 0.5766 | — |
| 4 | + online pairs | 0.5316 | 0.5910 | 0.6138 | 0.6677 |
| 5 | + both online (self) | 0.4915 | — | 0.5153 | — |
| 6 | SFT (NLL all) | 0.9023 | 0.8922 | 0.9126 | 0.9281 |
| 7 | + offline neg-TC | — | 0.6242 | — | 0.5885 |
| 8 | + online neg-TC | — | 0.5731 | — | 0.6141 |
| 9 | + both online (neg) | — | 0.4102 | — | 0.4647 |

### Validator ROC-AUC (same across gen variants; shown for reference)

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 0.5609 | 0.5609 | — | — |
| 1 | RankAlign baseline | 0.5829 | 0.5829 | 0.5829 | 0.5829 |
| 2 | + offline self-TC | 0.5318 | — | 0.5318 | — |
| 3 | + online self-TC | 0.5784 | — | 0.5784 | — |
| 4 | + online pairs | 0.4784 | 0.4784 | 0.4784 | 0.4784 |
| 5 | + both online (self) | 0.5141 | — | 0.5141 | — |
| 6 | SFT (NLL all) | 0.7261 | 0.7261 | 0.7261 | 0.7261 |
| 7 | + offline neg-TC | — | 0.5554 | — | 0.5554 |
| 8 | + online neg-TC | — | 0.5699 | — | 0.5699 |
| 9 | + both online (neg) | — | 0.4872 | — | 0.4872 |

### Validator accuracy (threshold 0)

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 0.3884 | 0.3884 | — | — |
| 1 | RankAlign baseline | 0.3300 | 0.3300 | 0.3300 | 0.3300 |
| 2 | + offline self-TC | 0.5229 | — | 0.5229 | — |
| 3 | + online self-TC | 0.3894 | — | 0.3894 | — |
| 4 | + online pairs | 0.2579 | 0.2579 | 0.2579 | 0.2579 |
| 5 | + both online (self) | 0.2868 | — | 0.2868 | — |
| 6 | SFT (NLL all) | 0.7500 | 0.7500 | 0.7500 | 0.7500 |
| 7 | + offline neg-TC | — | 0.2546 | — | 0.2546 |
| 8 | + online neg-TC | — | 0.3665 | — | 0.3665 |
| 9 | + both online (neg) | — | 0.2505 | — | 0.2505 |

### Pearson(gen, validator) — `tc` gen vs val_score

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 0.4250 | 0.3892 | — | — |
| 1 | RankAlign baseline | 0.4246 | 0.1885 | 0.4497 | 0.3966 |
| 2 | + offline self-TC | 0.5022 | — | 0.4786 | — |
| 3 | + online self-TC | 0.4393 | — | 0.3254 | — |
| 4 | + online pairs | 0.3172 | 0.2384 | 0.3100 | 0.2802 |
| 5 | + both online (self) | 0.5725 | — | 0.2738 | — |
| 6 | SFT (NLL all) | 0.4742 | 0.2621 | 0.4207 | 0.3119 |
| 7 | + offline neg-TC | — | 0.1925 | — | 0.3287 |
| 8 | + online neg-TC | — | 0.5823 | — | 0.3142 |
| 9 | + both online (neg) | — | -0.0666 | — | 0.0042 |
