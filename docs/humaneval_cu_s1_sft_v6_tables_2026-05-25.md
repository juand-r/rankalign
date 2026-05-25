# humaneval-v2.1correct-upper x gemma-4-31B-it — s1 (SFT-lo) baseline [v6]

Snapshot: **2026-05-25**

**Experiment:** correct-upper | **Model:** gemma-4-31B-it | **Setting:** s1 = SFT-lo (`pref0.0 nllv1.0 nllg1.0 labelonly0.1`; no consistency-ft, no fsx, no TC) | **Dataset:** humaneval-v2.1correct-upper (82 problems)

> **Caveats:** (1) These scores are **v6** (`delta0.15-epoch1`), not v7 (`delta2.14`). For a pure-SFT baseline the rankalign `delta` does not affect training, but it is a different run version — flag for the reader. (2) Only the **base-typicality** eval variants were run, so **PMI self / Neg self were never computed** for s1 (shown as `---`).

Source: `outputs_gemma4_from_pod-v7/s1_sft_v6/` (164 CSVs = 82 basetyp + 82 basetypneg), from mll `/datastor2/jdr/rankalign/outputs_gemma4_from_pod`. Rebuild: `python scripts/_build_s1_sft_v6_table.py`.

## GenROC x 100

| Method | Raw | PMI self | PMI base | Neg self | Neg base |
| --- | --- | --- | --- | --- | --- |
| 1 SFT labelonly 10% (s1, v6) | 80.11 ± 1.09 | --- | 84.48 ± 1.10 | --- | 87.64 ± 1.33 |

## Pearson(gen, val) x 100

| Method | Raw | PMI self | PMI base | Neg self | Neg base |
| --- | --- | --- | --- | --- | --- |
| 1 SFT labelonly 10% (s1, v6) | 43.21 ± 1.98 | --- | 28.50 ± 3.03 | --- | 44.11 ± 2.99 |

## Spearman(gen, val) x 100

| Method | Raw | PMI self | PMI base | Neg self | Neg base |
| --- | --- | --- | --- | --- | --- |
| 1 SFT labelonly 10% (s1, v6) | 52.16 ± 1.94 | --- | 43.87 ± 2.81 | --- | 49.66 ± 2.75 |

## ValROC x 100

| Method | Raw | PMI self | PMI base | Neg self | Neg base |
| --- | --- | --- | --- | --- | --- |
| 1 SFT labelonly 10% (s1, v6) | 91.98 ± 0.97 | --- | 91.98 ± 0.97 | --- | 91.98 ± 0.97 |

## ValAcc x 100

| Method | Raw | PMI self | PMI base | Neg self | Neg base |
| --- | --- | --- | --- | --- | --- |
| 1 SFT labelonly 10% (s1, v6) | 64.86 ± 1.75 | --- | 64.86 ± 1.75 | --- | 64.86 ± 1.75 |

