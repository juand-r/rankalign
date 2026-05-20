# Persona-v0 base + trained-model results

All cells: **mean (std) across the 8 persona-v0-<slug> test tasks, x 100**.
`raw` column = gen_score; `tc` column = gen_score_typcorr.

Eval flags by row:
- `0.Base`  : `--self-typcorr` or `--neg-typcorr` only (the base model IS the typicality reference, so no `--base-typcorr`).
- `1`–`9` finetuned rows : `--self-typcorr --base-typcorr` or `--neg-typcorr --base-typcorr`.

Source CSV: `output-metrics/persona_v0_tables_long.csv`
(derived from `output-metrics/persona_v0_summary.csv`).

## gemma-2-9b-it  Generator ROC-AUC  (eval = self-typcorr)

| variant | raw | tc |
| --- | --- | --- |
| 0.Base | 40.47 (9.35) | 43.93 (12.63) |
| 1.SFT-lo | 65.32 (36.92) | 68.10 (41.40) |
| 2.RankAlign | 40.74 (24.04) | 54.33 (34.00) |
| 3.New+fsx | 40.09 (10.56) | 42.86 (13.17) |
| 4.New+fsx+selfTC | 41.26 (10.98) | 50.20 (15.53) |
| 5.RankAlign+fsx+selfTC | 38.00 (10.20) | 47.73 (33.42) |
| 6.RankAlign+selfTC | 38.50 (10.46) | 50.90 (35.90) |

## gemma-2-9b-it  Validator ROC-AUC  (eval = self-typcorr)

| variant | raw | tc |
| --- | --- | --- |
| 0.Base | 48.07 (43.07) | 48.07 (43.07) |
| 1.SFT-lo | 89.08 (11.55) | 89.08 (11.55) |
| 2.RankAlign | 51.88 (44.18) | 51.88 (44.18) |
| 3.New+fsx | 66.83 (35.68) | 66.83 (35.68) |
| 4.New+fsx+selfTC | 67.33 (40.50) | 67.33 (40.50) |
| 5.RankAlign+fsx+selfTC | 44.56 (45.61) | 44.56 (45.61) |
| 6.RankAlign+selfTC | 44.22 (46.27) | 44.22 (46.27) |

## gemma-2-9b-it  Validator accuracy (threshold 0)  (eval = self-typcorr)

| variant | raw | tc |
| --- | --- | --- |
| 0.Base | 52.85 (34.81) | 52.85 (34.81) |
| 1.SFT-lo | 76.59 (13.35) | 76.59 (13.35) |
| 2.RankAlign | 53.73 (27.26) | 53.73 (27.26) |
| 3.New+fsx | 64.15 (28.78) | 64.15 (28.78) |
| 4.New+fsx+selfTC | 65.08 (33.00) | 65.08 (33.00) |
| 5.RankAlign+fsx+selfTC | 49.83 (36.50) | 49.83 (36.50) |
| 6.RankAlign+selfTC | 47.14 (38.44) | 47.14 (38.44) |

## gemma-2-9b-it  Pearson(gen_score, val_score)  (eval = self-typcorr)

| variant | raw | tc |
| --- | --- | --- |
| 0.Base | -7.26 (19.51) | -9.59 (16.77) |
| 1.SFT-lo | 18.90 (39.71) | 24.27 (43.31) |
| 2.RankAlign | 35.47 (29.00) | 48.08 (15.33) |
| 3.New+fsx | -8.52 (17.42) | -4.80 (19.51) |
| 4.New+fsx+selfTC | -1.92 (18.24) | 16.65 (17.13) |
| 5.RankAlign+fsx+selfTC | 13.21 (28.13) | 46.37 (27.51) |
| 6.RankAlign+selfTC | 15.11 (26.92) | 50.96 (26.51) |

## gemma-2-9b-it  Generator ROC-AUC  (eval = neg-typcorr)

| variant | raw | tc |
| --- | --- | --- |
| 0.Base | 40.47 (9.35) | 50.92 (9.06) |
| 1.SFT-lo | 65.32 (36.92) | 70.20 (38.90) |
| 2.RankAlign | 40.74 (24.04) | 56.78 (35.79) |
| 3.New+fsx | 40.09 (10.56) | 48.86 (16.25) |
| 7.New+fsx+negTC | 41.30 (10.53) | 53.54 (17.25) |
| 8.RankAlign+fsx+negTC | 37.44 (10.73) | 52.03 (33.93) |
| 9.RankAlign+negTC | 36.58 (8.19) | 50.61 (33.81) |

## gemma-2-9b-it  Validator ROC-AUC  (eval = neg-typcorr)

| variant | raw | tc |
| --- | --- | --- |
| 0.Base | 48.07 (43.07) | 48.07 (43.07) |
| 1.SFT-lo | 89.08 (11.55) | 89.08 (11.55) |
| 2.RankAlign | 51.88 (44.18) | 51.88 (44.18) |
| 3.New+fsx | 66.83 (35.68) | 66.83 (35.68) |
| 7.New+fsx+negTC | 72.50 (32.30) | 72.50 (32.30) |
| 8.RankAlign+fsx+negTC | 41.21 (47.86) | 41.21 (47.86) |
| 9.RankAlign+negTC | 45.22 (45.15) | 45.22 (45.15) |

## gemma-2-9b-it  Validator accuracy (threshold 0)  (eval = neg-typcorr)

| variant | raw | tc |
| --- | --- | --- |
| 0.Base | 52.85 (34.81) | 52.85 (34.81) |
| 1.SFT-lo | 76.59 (13.35) | 76.59 (13.35) |
| 2.RankAlign | 53.73 (27.26) | 53.73 (27.26) |
| 3.New+fsx | 64.15 (28.78) | 64.15 (28.78) |
| 7.New+fsx+negTC | 67.60 (24.15) | 67.60 (24.15) |
| 8.RankAlign+fsx+negTC | 50.62 (33.86) | 50.62 (33.86) |
| 9.RankAlign+negTC | 51.26 (35.17) | 51.26 (35.17) |

## gemma-2-9b-it  Pearson(gen_score, val_score)  (eval = neg-typcorr)

| variant | raw | tc |
| --- | --- | --- |
| 0.Base | -7.26 (19.51) | 16.62 (7.10) |
| 1.SFT-lo | 18.90 (39.71) | 27.20 (41.22) |
| 2.RankAlign | 35.47 (29.00) | 53.85 (10.80) |
| 3.New+fsx | -8.52 (17.42) | 12.31 (18.55) |
| 7.New+fsx+negTC | -6.30 (15.80) | 27.06 (10.77) |
| 8.RankAlign+fsx+negTC | 10.10 (30.09) | 49.38 (27.94) |
| 9.RankAlign+negTC | 8.97 (29.01) | 53.76 (19.89) |

## gemma-2-2b-it  Generator ROC-AUC  (eval = self-typcorr)

| variant | raw | tc |
| --- | --- | --- |
| 0.Base | 41.63 (8.43) | 47.41 (10.57) |
| 1.SFT-lo | 62.99 (28.63) | 67.67 (34.51) |
| 2.RankAlign | 25.08 (17.77) | 33.41 (27.84) |
| 3.New+fsx | 38.21 (10.33) | 38.48 (9.45) |
| 4.New+fsx+selfTC | 36.07 (10.36) | 40.67 (9.78) |
| 5.RankAlign+fsx+selfTC | 26.85 (11.26) | 24.74 (22.43) |
| 6.RankAlign+selfTC | 27.71 (13.23) | 27.42 (26.29) |

## gemma-2-2b-it  Validator ROC-AUC  (eval = self-typcorr)

| variant | raw | tc |
| --- | --- | --- |
| 0.Base | 40.61 (48.53) | 40.61 (48.53) |
| 1.SFT-lo | 89.17 (15.87) | 89.17 (15.87) |
| 2.RankAlign | 39.75 (49.73) | 39.75 (49.73) |
| 3.New+fsx | 49.49 (42.39) | 49.49 (42.39) |
| 4.New+fsx+selfTC | 62.72 (30.73) | 62.72 (30.73) |
| 5.RankAlign+fsx+selfTC | 41.18 (48.50) | 41.18 (48.50) |
| 6.RankAlign+selfTC | 42.35 (48.00) | 42.35 (48.00) |

## gemma-2-2b-it  Validator accuracy (threshold 0)  (eval = self-typcorr)

| variant | raw | tc |
| --- | --- | --- |
| 0.Base | 42.42 (40.09) | 42.42 (40.09) |
| 1.SFT-lo | 50.00 (0.00) | 50.00 (0.00) |
| 2.RankAlign | 42.74 (45.11) | 42.74 (45.11) |
| 3.New+fsx | 52.55 (34.48) | 52.55 (34.48) |
| 4.New+fsx+selfTC | 50.05 (0.08) | 50.05 (0.08) |
| 5.RankAlign+fsx+selfTC | 49.55 (35.26) | 49.55 (35.26) |
| 6.RankAlign+selfTC | 47.68 (42.24) | 47.68 (42.24) |

## gemma-2-2b-it  Pearson(gen_score, val_score)  (eval = self-typcorr)

| variant | raw | tc |
| --- | --- | --- |
| 0.Base | 6.76 (16.62) | 13.16 (15.67) |
| 1.SFT-lo | 30.50 (28.53) | 43.26 (31.02) |
| 2.RankAlign | 37.35 (37.48) | 55.83 (18.21) |
| 3.New+fsx | -10.41 (15.77) | -6.02 (18.73) |
| 4.New+fsx+selfTC | -19.00 (14.04) | -14.77 (12.46) |
| 5.RankAlign+fsx+selfTC | 16.00 (33.83) | 39.53 (31.17) |
| 6.RankAlign+selfTC | 23.96 (34.02) | 51.70 (24.32) |

## gemma-2-2b-it  Generator ROC-AUC  (eval = neg-typcorr)

| variant | raw | tc |
| --- | --- | --- |
| 0.Base | 41.63 (8.43) | 53.16 (25.54) |
| 1.SFT-lo | 62.99 (28.63) | 71.45 (28.84) |
| 2.RankAlign | 25.08 (17.77) | 39.75 (35.80) |
| 3.New+fsx | 38.21 (10.33) | 42.79 (16.79) |
| 7.New+fsx+negTC | 38.06 (10.34) | 46.89 (16.77) |
| 8.RankAlign+fsx+negTC | 26.89 (9.97) | 33.13 (33.77) |
| 9.RankAlign+negTC | 26.65 (10.19) | 33.00 (34.27) |

## gemma-2-2b-it  Validator ROC-AUC  (eval = neg-typcorr)

| variant | raw | tc |
| --- | --- | --- |
| 0.Base | 40.61 (48.53) | 40.61 (48.53) |
| 1.SFT-lo | 89.17 (15.87) | 89.17 (15.87) |
| 2.RankAlign | 39.75 (49.73) | 39.75 (49.73) |
| 3.New+fsx | 49.49 (42.39) | 49.49 (42.39) |
| 7.New+fsx+negTC | 60.46 (34.65) | 60.46 (34.65) |
| 8.RankAlign+fsx+negTC | 40.70 (49.02) | 40.70 (49.02) |
| 9.RankAlign+negTC | 39.17 (50.10) | 39.17 (50.10) |

## gemma-2-2b-it  Validator accuracy (threshold 0)  (eval = neg-typcorr)

| variant | raw | tc |
| --- | --- | --- |
| 0.Base | 42.42 (40.09) | 42.42 (40.09) |
| 1.SFT-lo | 50.00 (0.00) | 50.00 (0.00) |
| 2.RankAlign | 42.74 (45.11) | 42.74 (45.11) |
| 3.New+fsx | 52.55 (34.48) | 52.55 (34.48) |
| 7.New+fsx+negTC | 54.48 (7.90) | 54.48 (7.90) |
| 8.RankAlign+fsx+negTC | 45.59 (31.67) | 45.59 (31.67) |
| 9.RankAlign+negTC | 50.03 (0.17) | 50.03 (0.17) |

## gemma-2-2b-it  Pearson(gen_score, val_score)  (eval = neg-typcorr)

| variant | raw | tc |
| --- | --- | --- |
| 0.Base | 6.76 (16.62) | 42.25 (21.42) |
| 1.SFT-lo | 30.50 (28.53) | 46.79 (25.96) |
| 2.RankAlign | 37.35 (37.48) | 66.29 (9.95) |
| 3.New+fsx | -10.41 (15.77) | 11.25 (18.86) |
| 7.New+fsx+negTC | -11.58 (16.54) | 9.89 (13.39) |
| 8.RankAlign+fsx+negTC | 21.00 (33.42) | 61.35 (14.08) |
| 9.RankAlign+negTC | 17.72 (33.17) | 61.78 (14.69) |
