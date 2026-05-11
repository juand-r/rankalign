# Cross-task mean across 8 OOD rosch tasks (gemma-2-2b)

**Train task:** rosch-furniture-and-bird-all (models trained here, 8 OOD eval tasks below).

**OOD tasks:** rosch-carpenters-tool, rosch-clothing, rosch-fruit, rosch-sport, rosch-toy, rosch-vegetable, rosch-vehicle, rosch-weapon.

**Validator:** log-odds (`--validator-log-odds`). **Generator column variant:** `tc` (TC-corrected gen score where applicable).

**All numeric cells are raw values × 100** (percentage points for ROC/accuracy; 0–100 units for Pearson). Reported as **mean (std)** across the 8 OOD tasks.

Long-form metrics: [quickiter_metrics_long_crosstask.csv](quickiter_metrics_long_crosstask.csv)

### Generator ROC-AUC — mean (std) across 8 OOD tasks — × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 77.83 (5.58) | 78.07 (11.25) | — | — |
| 1 | RankAlign baseline | 76.42 (8.67) | 71.01 (14.62) | 75.63 (8.83) | 61.50 (10.61) |
| 2 | + offline self-TC | 75.69 (9.08) | — | 74.26 (9.00) | — |
| 3 | + online self-TC | 76.72 (8.84) | — | 72.80 (9.79) | — |
| 4 | + online pairs | 77.88 (8.37) | 67.44 (13.78) | 77.76 (8.62) | 61.06 (12.18) |
| 5 | + both online (self) | 65.33 (8.82) | — | 61.46 (8.65) | — |
| 6 | SFT (NLL all) | 69.34 (7.77) | 60.72 (20.48) | 69.07 (7.82) | 52.25 (11.88) |
| 7 | + offline neg-TC | — | 37.22 (9.10) | — | 55.63 (12.17) |
| 8 | + online neg-TC | — | 63.54 (7.89) | — | 37.21 (12.14) |
| 9 | + both online (neg) | — | 79.71 (9.70) | — | 50.25 (13.39) |

### Validator ROC-AUC — mean (std) across 8 OOD tasks — × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 88.38 (8.49) | 88.38 (8.49) | — | — |
| 1 | RankAlign baseline | 86.67 (10.52) | 86.67 (10.52) | 86.67 (10.52) | 86.67 (10.52) |
| 2 | + offline self-TC | 80.49 (8.98) | — | 80.49 (8.98) | — |
| 3 | + online self-TC | 86.86 (8.47) | — | 86.86 (8.47) | — |
| 4 | + online pairs | 86.44 (9.22) | 86.44 (9.22) | 86.44 (9.22) | 86.44 (9.22) |
| 5 | + both online (self) | 86.83 (9.84) | — | 86.83 (9.84) | — |
| 6 | SFT (NLL all) | 88.56 (9.16) | 88.56 (9.16) | 88.56 (9.16) | 88.56 (9.16) |
| 7 | + offline neg-TC | — | 84.45 (9.19) | — | 84.45 (9.19) |
| 8 | + online neg-TC | — | 83.54 (10.68) | — | 83.54 (10.68) |
| 9 | + both online (neg) | — | 85.96 (12.09) | — | 85.96 (12.09) |

### Validator accuracy (thr 0) — mean (std) across 8 OOD tasks — × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 76.83 (10.42) | 76.83 (10.42) | — | — |
| 1 | RankAlign baseline | 69.40 (9.50) | 69.40 (9.50) | 69.40 (9.50) | 69.40 (9.50) |
| 2 | + offline self-TC | 66.40 (10.43) | — | 66.40 (10.43) | — |
| 3 | + online self-TC | 75.61 (8.66) | — | 75.61 (8.66) | — |
| 4 | + online pairs | 78.18 (10.60) | 78.18 (10.60) | 78.18 (10.60) | 78.18 (10.60) |
| 5 | + both online (self) | 67.77 (10.82) | — | 67.77 (10.82) | — |
| 6 | SFT (NLL all) | 79.65 (11.39) | 79.65 (11.39) | 79.65 (11.39) | 79.65 (11.39) |
| 7 | + offline neg-TC | — | 59.94 (10.87) | — | 59.94 (10.87) |
| 8 | + online neg-TC | — | 54.92 (5.51) | — | 54.92 (5.51) |
| 9 | + both online (neg) | — | 69.17 (7.23) | — | 69.17 (7.23) |

### Pearson(gen, validator) — mean (std) across 8 OOD tasks — × 100

| # | trained model | self | neg | basetyp | basetypneg |
| --- | --- | --- | --- | --- | --- |
| 0 | Base HF (gemma-2-2b) | 54.65 (10.41) | 40.50 (13.98) | — | — |
| 1 | RankAlign baseline | 60.40 (9.32) | 44.81 (17.96) | 59.22 (9.29) | 32.39 (16.29) |
| 2 | + offline self-TC | 53.45 (9.08) | — | 50.30 (7.72) | — |
| 3 | + online self-TC | 57.72 (13.41) | — | 46.96 (19.34) | — |
| 4 | + online pairs | 58.52 (10.15) | 21.00 (24.20) | 56.59 (10.43) | 20.99 (16.97) |
| 5 | + both online (self) | 40.38 (12.17) | — | 25.28 (17.28) | — |
| 6 | SFT (NLL all) | 34.61 (15.75) | 14.47 (21.79) | 33.55 (16.33) | -0.64 (20.47) |
| 7 | + offline neg-TC | — | -19.83 (21.42) | — | 10.00 (25.33) |
| 8 | + online neg-TC | — | 36.16 (14.65) | — | -23.27 (16.17) |
| 9 | + both online (neg) | — | 73.22 (8.56) | — | -2.12 (21.03) |
