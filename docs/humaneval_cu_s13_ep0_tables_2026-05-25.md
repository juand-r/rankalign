# humaneval-v2.1correct-upper x gemma-4-31B-it — s13 (SFT-lo + consistency-ft), epoch0

Snapshot: **2026-05-25**

**Experiment:** correct-upper | **Model:** gemma-4-31B-it | **Setting:** s13 = SFT-lo + `--consistency-ft` (`pref0.0 nllv1.0 nllg1.0 cft labelonly0.1`), **epoch0** | **Dataset:** humaneval-v2.1correct-upper (82 problems)

> **Notes:** (1) **epoch0** checkpoint (training crashed on mll before further epochs). (2) Only the **self-normalized** eval variants were run (Raw, PMI self, Neg self); PMI base / Neg base were intentionally dropped. (3) Adapter: `latkes/rankalign-v7-gemma-4-31B-it-d2.14-e0-...-cft-lo0.1-fix1`.

Source: `outputs_gemma4_from_pod-v7/s13_ep0/`. Rebuild: `python scripts/_build_s13_ep0_table.py`.

## GenROC x 100

| Method | Raw | PMI self | PMI base | Neg self | Neg base |
| --- | --- | --- | --- | --- | --- |
| 13 SFT + consistency-ft (s13, e0) | 79.43 ± 1.10 | 79.97 ± 1.21 | --- | 74.95 ± 1.43 | --- |

## Pearson(gen, val) x 100

| Method | Raw | PMI self | PMI base | Neg self | Neg base |
| --- | --- | --- | --- | --- | --- |
| 13 SFT + consistency-ft (s13, e0) | 52.35 ± 1.69 | 35.80 ± 2.96 | --- | 48.96 ± 2.27 | --- |

## Spearman(gen, val) x 100

| Method | Raw | PMI self | PMI base | Neg self | Neg base |
| --- | --- | --- | --- | --- | --- |
| 13 SFT + consistency-ft (s13, e0) | 54.80 ± 2.08 | 45.71 ± 2.56 | --- | 45.67 ± 2.50 | --- |

## ValROC x 100

| Method | Raw | PMI self | PMI base | Neg self | Neg base |
| --- | --- | --- | --- | --- | --- |
| 13 SFT + consistency-ft (s13, e0) | 93.82 ± 0.82 | 93.82 ± 0.82 | --- | 93.82 ± 0.82 | --- |

## ValAcc x 100

| Method | Raw | PMI self | PMI base | Neg self | Neg base |
| --- | --- | --- | --- | --- | --- |
| 13 SFT + consistency-ft (s13, e0) | 88.18 ± 1.00 | 88.18 ± 1.00 | --- | 88.18 ± 1.00 | --- |

