# membership-sans-rosch-v0 → rosch — mean(std) across 10 rosch tasks

Aggregated from every `scores_*membership-sans-rosch-v0-all*_rosch-*` CSV we have on disk, grouped by (model, epoch, training recipe, eval_ref). Each cell shows **mean × 100 (std × 100)** of the corresponding metric across the 10 rosch eval tasks. Generator score variant = `tc` (typicality-corrected gen score where applicable).

Older cohorts live in `outputs/`; the canonical (May 11 / May 13) cohorts live in `outputs-quickiter/`.

## gemma-2-2b, epoch0  (n_tasks per row shown in `n`)

### Generator ROC-AUC — × 100

| date | source | recipe | eval_ref | n | value |
|---|---|---|---|---|---|
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x` | `basetyp` | 10 | 78.92 (8.48) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x` | `basetypneg` | 10 | 66.94 (13.14) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x` | `neg` | 10 | 87.21 (7.62) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x` | `self` | 10 | 80.19 (7.71) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x_online-pairs` | `basetyp` | 10 | 80.68 (9.79) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x_online-pairs` | `basetypneg` | 10 | 67.36 (15.20) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x_online-pairs` | `neg` | 10 | 84.44 (12.86) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x_online-pairs` | `self` | 10 | 81.55 (9.70) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x` | `basetyp` | 10 | 82.51 (5.85) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x` | `basetypneg` | 10 | 76.25 (6.34) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x` | `neg` | 10 | 73.63 (11.23) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x` | `self` | 10 | 83.44 (5.69) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x` | `basetypneg` | 10 | 77.60 (11.13) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x` | `neg` | 10 | 77.59 (15.04) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_online-pairs_online-tc` | `basetypneg` | 10 | 45.50 (10.90) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_online-pairs_online-tc` | `neg` | 10 | 77.89 (9.37) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_online-tc` | `basetypneg` | 10 | 45.10 (10.71) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_online-tc` | `neg` | 10 | 77.43 (9.31) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x` | `basetyp` | 10 | 82.78 (7.90) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x` | `self` | 10 | 83.69 (8.12) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_online-pairs_online-tc` | `basetyp` | 10 | 81.32 (8.94) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_online-pairs_online-tc` | `self` | 10 | 82.19 (9.02) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_online-tc` | `basetyp` | 10 | 70.08 (11.89) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_online-tc` | `self` | 10 | 76.28 (9.42) |

### Validator ROC-AUC — × 100

| date | source | recipe | eval_ref | n | value |
|---|---|---|---|---|---|
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x` | `basetyp` | 10 | 88.26 (9.75) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x` | `basetypneg` | 10 | 88.26 (9.75) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x` | `neg` | 10 | 88.26 (9.75) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x` | `self` | 10 | 88.26 (9.75) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x_online-pairs` | `basetyp` | 10 | 86.86 (9.70) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x_online-pairs` | `basetypneg` | 10 | 86.86 (9.70) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x_online-pairs` | `neg` | 10 | 86.86 (9.70) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x_online-pairs` | `self` | 10 | 86.86 (9.70) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x` | `basetyp` | 10 | 87.69 (7.89) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x` | `basetypneg` | 10 | 87.69 (7.89) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x` | `neg` | 10 | 87.69 (7.89) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x` | `self` | 10 | 87.69 (7.89) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x` | `basetypneg` | 10 | 88.80 (9.31) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x` | `neg` | 10 | 88.80 (9.31) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_online-pairs_online-tc` | `basetypneg` | 10 | 89.52 (9.20) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_online-pairs_online-tc` | `neg` | 10 | 89.52 (9.20) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_online-tc` | `basetypneg` | 10 | 89.53 (8.60) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_online-tc` | `neg` | 10 | 89.53 (8.60) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x` | `basetyp` | 10 | 88.56 (10.09) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x` | `self` | 10 | 88.56 (10.09) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_online-pairs_online-tc` | `basetyp` | 10 | 87.47 (9.00) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_online-pairs_online-tc` | `self` | 10 | 87.47 (9.00) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_online-tc` | `basetyp` | 10 | 88.11 (9.60) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_online-tc` | `self` | 10 | 88.11 (9.60) |

### Validator accuracy (thr 0) — × 100

| date | source | recipe | eval_ref | n | value |
|---|---|---|---|---|---|
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x` | `basetyp` | 10 | 77.07 (10.81) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x` | `basetypneg` | 10 | 77.07 (10.81) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x` | `neg` | 10 | 77.07 (10.81) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x` | `self` | 10 | 77.07 (10.81) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x_online-pairs` | `basetyp` | 10 | 74.17 (12.26) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x_online-pairs` | `basetypneg` | 10 | 74.17 (12.26) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x_online-pairs` | `neg` | 10 | 74.17 (12.26) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x_online-pairs` | `self` | 10 | 74.17 (12.26) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x` | `basetyp` | 10 | 50.57 (1.79) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x` | `basetypneg` | 10 | 50.57 (1.79) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x` | `neg` | 10 | 50.57 (1.79) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x` | `self` | 10 | 50.57 (1.79) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x` | `basetypneg` | 10 | 78.17 (11.65) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x` | `neg` | 10 | 78.17 (11.65) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_online-pairs_online-tc` | `basetypneg` | 10 | 78.79 (11.02) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_online-pairs_online-tc` | `neg` | 10 | 78.79 (11.02) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_online-tc` | `basetypneg` | 10 | 80.46 (9.57) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_online-tc` | `neg` | 10 | 80.46 (9.57) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x` | `basetyp` | 10 | 78.90 (10.14) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x` | `self` | 10 | 78.90 (10.14) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_online-pairs_online-tc` | `basetyp` | 10 | 74.91 (10.83) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_online-pairs_online-tc` | `self` | 10 | 74.91 (10.83) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_online-tc` | `basetyp` | 10 | 61.76 (12.84) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_online-tc` | `self` | 10 | 61.76 (12.84) |

### Pearson(gen, validator) — × 100

| date | source | recipe | eval_ref | n | value |
|---|---|---|---|---|---|
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x` | `basetyp` | 10 | 62.14 (6.87) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x` | `basetypneg` | 10 | 33.61 (16.26) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x` | `neg` | 10 | 65.27 (7.25) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x` | `self` | 10 | 65.47 (6.43) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x_online-pairs` | `basetyp` | 10 | 56.59 (6.75) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x_online-pairs` | `basetypneg` | 10 | 28.31 (21.80) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x_online-pairs` | `neg` | 10 | 60.92 (13.79) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x_online-pairs` | `self` | 10 | 61.50 (8.53) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x` | `basetyp` | 10 | 60.72 (10.75) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x` | `basetypneg` | 10 | 43.29 (17.40) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x` | `neg` | 10 | 44.06 (18.05) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x` | `self` | 10 | 62.06 (10.81) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x` | `basetypneg` | 10 | 55.91 (13.96) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x` | `neg` | 10 | 54.79 (14.69) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_online-pairs_online-tc` | `basetypneg` | 10 | -8.30 (18.13) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_online-pairs_online-tc` | `neg` | 10 | 71.53 (9.98) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_online-tc` | `basetypneg` | 10 | -12.38 (16.79) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_online-tc` | `neg` | 10 | 67.22 (13.59) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x` | `basetyp` | 10 | 66.27 (8.43) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x` | `self` | 10 | 68.62 (9.52) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_online-pairs_online-tc` | `basetyp` | 10 | 62.09 (6.89) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_online-pairs_online-tc` | `self` | 10 | 64.86 (7.10) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_online-tc` | `basetyp` | 10 | 34.02 (24.66) |
| 20260511 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_online-tc` | `self` | 10 | 55.11 (12.27) |


## gemma-2-2b, epoch2  (n_tasks per row shown in `n`)

### Generator ROC-AUC — × 100

| date | source | recipe | eval_ref | n | value |
|---|---|---|---|---|---|
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x` | `basetyp` | 10 | 80.95 (8.60) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x` | `basetypneg` | 10 | 69.36 (12.32) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x` | `neg` | 10 | 82.96 (9.11) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x` | `self` | 10 | 81.63 (8.14) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x_online-pairs` | `basetyp` | 10 | 80.44 (9.61) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x_online-pairs` | `basetypneg` | 10 | 64.47 (14.93) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x_online-pairs` | `neg` | 10 | 75.60 (13.21) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x_online-pairs` | `self` | 10 | 80.94 (9.30) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_semi0.1` | `self` | 10 | 84.82 (7.50) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x` | `basetyp` | 10 | 81.53 (6.20) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x` | `basetypneg` | 10 | 76.53 (7.95) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x` | `neg` | 10 | 74.86 (12.71) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x` | `self` | 10 | 82.96 (5.33) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_labelonly0.1` | `self` | 10 | 76.99 (6.56) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_full-completion_semi0.1` | `self` | 10 | 80.82 (9.40) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x` | `basetypneg` | 10 | 76.26 (10.47) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x` | `neg` | 10 | 80.82 (12.87) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_online-pairs_online-tc` | `basetypneg` | 10 | 53.22 (11.30) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_online-pairs_online-tc` | `neg` | 10 | 79.52 (9.87) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_online-tc` | `basetypneg` | 10 | 61.09 (11.57) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_online-tc` | `neg` | 10 | 84.93 (7.76) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_vallogodds_semi0.1` | `neg` | 10 | 82.01 (11.55) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-neg_full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_semi0.1` | `neg` | 10 | 80.99 (14.68) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x` | `basetyp` | 10 | 86.37 (7.10) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x` | `self` | 10 | 86.70 (6.94) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_online-pairs_online-tc` | `basetyp` | 10 | 81.36 (7.37) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_online-pairs_online-tc` | `self` | 10 | 81.98 (7.42) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_online-tc` | `basetyp` | 10 | 82.62 (7.84) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_online-tc` | `self` | 10 | 83.77 (7.94) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_vallogodds_semi0.1` | `self` | 10 | 81.31 (8.21) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-self_full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_semi0.1` | `self` | 10 | 85.14 (7.86) |

### Validator ROC-AUC — × 100

| date | source | recipe | eval_ref | n | value |
|---|---|---|---|---|---|
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x` | `basetyp` | 10 | 86.81 (10.42) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x` | `basetypneg` | 10 | 86.81 (10.42) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x` | `neg` | 10 | 86.81 (10.42) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x` | `self` | 10 | 86.81 (10.42) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x_online-pairs` | `basetyp` | 10 | 87.84 (9.82) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x_online-pairs` | `basetypneg` | 10 | 87.84 (9.82) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x_online-pairs` | `neg` | 10 | 87.84 (9.82) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x_online-pairs` | `self` | 10 | 87.84 (9.82) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_semi0.1` | `self` | 10 | 88.75 (8.28) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x` | `basetyp` | 10 | 89.39 (7.57) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x` | `basetypneg` | 10 | 89.39 (7.57) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x` | `neg` | 10 | 89.39 (7.57) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x` | `self` | 10 | 89.39 (7.57) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_labelonly0.1` | `self` | 10 | 89.50 (7.75) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_full-completion_semi0.1` | `self` | 10 | 88.52 (10.15) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x` | `basetypneg` | 10 | 87.46 (10.15) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x` | `neg` | 10 | 87.46 (10.15) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_online-pairs_online-tc` | `basetypneg` | 10 | 88.49 (9.12) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_online-pairs_online-tc` | `neg` | 10 | 88.49 (9.12) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_online-tc` | `basetypneg` | 10 | 89.53 (8.63) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_online-tc` | `neg` | 10 | 89.53 (8.63) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_vallogodds_semi0.1` | `neg` | 10 | 89.14 (9.49) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-neg_full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_semi0.1` | `neg` | 10 | 88.07 (10.33) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x` | `basetyp` | 10 | 89.80 (8.91) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x` | `self` | 10 | 89.80 (8.91) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_online-pairs_online-tc` | `basetyp` | 10 | 89.72 (9.15) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_online-pairs_online-tc` | `self` | 10 | 89.72 (9.15) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_online-tc` | `basetyp` | 10 | 88.27 (8.22) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_online-tc` | `self` | 10 | 88.27 (8.22) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_vallogodds_semi0.1` | `self` | 10 | 88.09 (8.70) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-self_full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_semi0.1` | `self` | 10 | 88.84 (7.68) |

### Validator accuracy (thr 0) — × 100

| date | source | recipe | eval_ref | n | value |
|---|---|---|---|---|---|
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x` | `basetyp` | 10 | 76.70 (11.09) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x` | `basetypneg` | 10 | 76.70 (11.09) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x` | `neg` | 10 | 76.70 (11.09) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x` | `self` | 10 | 76.70 (11.09) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x_online-pairs` | `basetyp` | 10 | 72.80 (11.81) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x_online-pairs` | `basetypneg` | 10 | 72.80 (11.81) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x_online-pairs` | `neg` | 10 | 72.80 (11.81) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x_online-pairs` | `self` | 10 | 72.80 (11.81) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_semi0.1` | `self` | 10 | 76.73 (12.11) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x` | `basetyp` | 10 | 52.17 (2.65) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x` | `basetypneg` | 10 | 52.17 (2.65) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x` | `neg` | 10 | 52.17 (2.65) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x` | `self` | 10 | 52.17 (2.65) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_labelonly0.1` | `self` | 10 | 50.28 (0.89) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_full-completion_semi0.1` | `self` | 10 | 81.21 (13.12) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x` | `basetypneg` | 10 | 75.12 (10.82) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x` | `neg` | 10 | 75.12 (10.82) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_online-pairs_online-tc` | `basetypneg` | 10 | 79.32 (9.67) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_online-pairs_online-tc` | `neg` | 10 | 79.32 (9.67) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_online-tc` | `basetypneg` | 10 | 75.59 (12.02) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_online-tc` | `neg` | 10 | 75.59 (12.02) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_vallogodds_semi0.1` | `neg` | 10 | 78.02 (10.08) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-neg_full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_semi0.1` | `neg` | 10 | 79.86 (11.32) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x` | `basetyp` | 10 | 80.33 (9.86) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x` | `self` | 10 | 80.33 (9.86) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_online-pairs_online-tc` | `basetyp` | 10 | 81.57 (11.29) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_online-pairs_online-tc` | `self` | 10 | 81.57 (11.29) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_online-tc` | `basetyp` | 10 | 60.31 (9.07) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_online-tc` | `self` | 10 | 60.31 (9.07) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_vallogodds_semi0.1` | `self` | 10 | 78.56 (11.53) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-self_full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_semi0.1` | `self` | 10 | 76.10 (12.60) |

### Pearson(gen, validator) — × 100

| date | source | recipe | eval_ref | n | value |
|---|---|---|---|---|---|
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x` | `basetyp` | 10 | 63.47 (6.31) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x` | `basetypneg` | 10 | 37.67 (15.97) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x` | `neg` | 10 | 54.78 (13.27) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x` | `self` | 10 | 66.39 (7.63) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x_online-pairs` | `basetyp` | 10 | 63.67 (7.71) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x_online-pairs` | `basetypneg` | 10 | 28.50 (19.81) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x_online-pairs` | `neg` | 10 | 41.76 (15.36) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_force-same-x_online-pairs` | `self` | 10 | 67.44 (8.91) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_semi0.1` | `self` | 10 | 66.68 (9.98) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x` | `basetyp` | 10 | 58.29 (10.17) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x` | `basetypneg` | 10 | 45.24 (17.24) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x` | `neg` | 10 | 47.44 (15.11) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x` | `self` | 10 | 60.68 (9.02) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_labelonly0.1` | `self` | 10 | 44.40 (12.17) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_full-completion_semi0.1` | `self` | 10 | 65.89 (6.79) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x` | `basetypneg` | 10 | 55.18 (15.62) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x` | `neg` | 10 | 60.81 (12.59) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_online-pairs_online-tc` | `basetypneg` | 10 | 4.72 (16.56) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_online-pairs_online-tc` | `neg` | 10 | 73.43 (9.14) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_online-tc` | `basetypneg` | 10 | 15.78 (18.69) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_online-tc` | `neg` | 10 | 78.53 (5.20) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_vallogodds_semi0.1` | `neg` | 10 | 59.74 (12.63) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-neg_full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_semi0.1` | `neg` | 10 | 58.39 (12.50) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x` | `basetyp` | 10 | 71.26 (5.42) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x` | `self` | 10 | 72.62 (5.74) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_online-pairs_online-tc` | `basetyp` | 10 | 65.14 (8.56) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_online-pairs_online-tc` | `self` | 10 | 68.47 (8.85) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_online-tc` | `basetyp` | 10 | 62.83 (11.68) |
| 20260513 | `outputs-quickiter` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_online-tc` | `self` | 10 | 67.15 (9.72) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_vallogodds_semi0.1` | `self` | 10 | 61.96 (8.06) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-self_full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_semi0.1` | `self` | 10 | 66.79 (9.23) |


## gemma-2-2b-it, epoch2  (n_tasks per row shown in `n`)

### Generator ROC-AUC — × 100

| date | source | recipe | eval_ref | n | value |
|---|---|---|---|---|---|
| 20260502 | `outputs` | `d2g_random_alpha1.0_full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_semi0.1` | `self` | 10 | 85.96 (8.00) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_labelonly0.1` | `self` | 10 | 82.47 (7.70) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_full-completion_semi0.1` | `self` | 10 | 88.17 (7.35) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_vallogodds_semi0.1` | `neg` | 10 | 86.91 (8.52) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-neg_full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_semi0.1` | `neg` | 10 | 87.49 (7.43) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_vallogodds_semi0.1` | `self` | 10 | 86.97 (5.62) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-self_full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_semi0.1` | `self` | 10 | 87.19 (6.88) |

### Validator ROC-AUC — × 100

| date | source | recipe | eval_ref | n | value |
|---|---|---|---|---|---|
| 20260502 | `outputs` | `d2g_random_alpha1.0_full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_semi0.1` | `self` | 10 | 91.34 (7.67) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_labelonly0.1` | `self` | 10 | 88.05 (6.81) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_full-completion_semi0.1` | `self` | 10 | 90.06 (7.59) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_vallogodds_semi0.1` | `neg` | 10 | 89.51 (6.80) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-neg_full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_semi0.1` | `neg` | 10 | 90.36 (8.28) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_vallogodds_semi0.1` | `self` | 10 | 89.67 (8.37) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-self_full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_semi0.1` | `self` | 10 | 89.61 (8.66) |

### Validator accuracy (thr 0) — × 100

| date | source | recipe | eval_ref | n | value |
|---|---|---|---|---|---|
| 20260502 | `outputs` | `d2g_random_alpha1.0_full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_semi0.1` | `self` | 10 | 84.70 (7.64) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_labelonly0.1` | `self` | 10 | 50.00 (0.00) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_full-completion_semi0.1` | `self` | 10 | 81.10 (8.79) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_vallogodds_semi0.1` | `neg` | 10 | 68.23 (7.92) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-neg_full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_semi0.1` | `neg` | 10 | 76.19 (9.66) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_vallogodds_semi0.1` | `self` | 10 | 79.12 (7.70) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-self_full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_semi0.1` | `self` | 10 | 81.77 (11.03) |

### Pearson(gen, validator) — × 100

| date | source | recipe | eval_ref | n | value |
|---|---|---|---|---|---|
| 20260502 | `outputs` | `d2g_random_alpha1.0_full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_semi0.1` | `self` | 10 | 73.81 (7.06) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_labelonly0.1` | `self` | 10 | 62.25 (10.92) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_full-completion_semi0.1` | `self` | 10 | 76.27 (6.49) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_vallogodds_semi0.1` | `neg` | 10 | 64.04 (9.66) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-neg_full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_semi0.1` | `neg` | 10 | 68.84 (10.01) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-self_full-completion_force-same-x_vallogodds_semi0.1` | `self` | 10 | 73.15 (5.82) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-self_full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_semi0.1` | `self` | 10 | 78.18 (5.71) |


## gemma-2-9b-it, epoch2  (n_tasks per row shown in `n`)

### Generator ROC-AUC — × 100

| date | source | recipe | eval_ref | n | value |
|---|---|---|---|---|---|
| 20260501 | `outputs` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_labelonly0.1_merged` | `self` | 10 | 84.87 (4.81) |
| 20260501 | `outputs` | `d2g_random_alpha1.0_full-completion_semi0.1_merged` | `self` | 10 | 90.10 (5.44) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_vallogodds_semi0.1_merged` | `neg` | 10 | 88.12 (8.37) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-neg_full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_semi0.1_merged` | `neg` | 10 | 88.93 (8.35) |

### Validator ROC-AUC — × 100

| date | source | recipe | eval_ref | n | value |
|---|---|---|---|---|---|
| 20260501 | `outputs` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_labelonly0.1_merged` | `self` | 10 | 93.54 (6.82) |
| 20260501 | `outputs` | `d2g_random_alpha1.0_full-completion_semi0.1_merged` | `self` | 10 | 94.78 (6.19) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_vallogodds_semi0.1_merged` | `neg` | 10 | 94.92 (6.11) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-neg_full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_semi0.1_merged` | `neg` | 10 | 94.78 (6.06) |

### Validator accuracy (thr 0) — × 100

| date | source | recipe | eval_ref | n | value |
|---|---|---|---|---|---|
| 20260501 | `outputs` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_labelonly0.1_merged` | `self` | 10 | 87.53 (7.17) |
| 20260501 | `outputs` | `d2g_random_alpha1.0_full-completion_semi0.1_merged` | `self` | 10 | 87.14 (7.81) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_vallogodds_semi0.1_merged` | `neg` | 10 | 87.39 (7.47) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-neg_full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_semi0.1_merged` | `neg` | 10 | 87.15 (7.45) |

### Pearson(gen, validator) — × 100

| date | source | recipe | eval_ref | n | value |
|---|---|---|---|---|---|
| 20260501 | `outputs` | `d2g_random_alpha1.0_full-completion_pref0.0_nllv1.0_nllg1.0_labelonly0.1_merged` | `self` | 10 | 58.81 (6.62) |
| 20260501 | `outputs` | `d2g_random_alpha1.0_full-completion_semi0.1_merged` | `self` | 10 | 70.00 (9.58) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_vallogodds_semi0.1_merged` | `neg` | 10 | 64.10 (15.02) |
| 20260502 | `outputs` | `d2g_random_alpha1.0_tc-neg_full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds_semi0.1_merged` | `neg` | 10 | 65.18 (14.43) |

