# v7/fix1 Tables — snapshot 2026-05-24

Snapshot of what the v7 table builders produce **right now** (2026-05-24
~19:30 local), as the overnight backlog continues to drain. Re-run the
commands at the bottom to refresh.

The numbers below are **GenROC × 100, mean ± stderr** across the 10 rosch
test categories (rosch tables) or 6 persona-v1 test personas (persona
tables). Cells showing `—` in the Raw / PMI columns are settings whose
score CSVs are not on disk yet (still training, eval pending, or never
launched). Cells showing `---` are NA per the v7 dispatcher's eval policy
(only one TC variant is ever evaluated per setting).

> **Salvage note** (corrected 2026-05-24 19:50): rosch 9b-it × {s4, s7,
> s11, s12} and rosch 2b-it × {s3, s4, s5, s7} appear blank below because
> those eight overnight evals crashed with OSError 36 before writing CSVs.
> Their aggregate metrics were salvaged from `.out` logs — see
> [`docs/salvaged_rosch_metrics.md`](salvaged_rosch_metrics.md). The
> salvaged `gen_roc` is the **TC-corrected** generator metric (NOT raw —
> all eight evals had `--{self,neg}-typcorr --base-typcorr`, so
> `args.typicality_correction = True`). For `--self-typcorr --base-typcorr`
> evals (s3/s4/s5/s11) the salvage value belongs in **PMI base**
> (`basetyp-` prefix); for `--neg-typcorr --base-typcorr` evals (s7/s12)
> it belongs in **Neg base** (`basetypneg-` prefix). Cells annotated
> below with `(salvage: …)` reflect this.

## rosch / gemma-2-9b-it / GenROC

Trained on `membership-sans-rosch-v0-all`; eval'd on the 10 rosch
categories (all OOD).

| Method | Raw | PMI self | PMI base | Neg self | Neg base |
| --- | --- | --- | --- | --- | --- |
| 0 Base | 66.91 ± 2.55 | 77.73 ± 2.18 | — | 81.20 ± 3.97 | — |
| 1 SFT labelonly 10% | — | — | — | --- | --- |
| 2 RankAlign | 88.37 ± 1.46 | — | 89.11 ± 1.25 | --- | --- |
| 3 New + fsx [-TC] | 88.78 ± 1.45 | — | 88.50 ± 1.11 | --- | --- |
| 4 New + PMI + fsx | — | — | *(salvage: 91.99 ± 1.12, job 41982)* | --- | --- |
| 5 RA + PMI + fsx [-NLL] | 83.60 ± 2.16 (n=9) | — | 91.48 ± 1.72 (n=9) | --- | --- |
| 6 RA + PMI [+TC] | 82.60 ± 2.35 | — | 91.32 ± 1.78 | --- | --- |
| 11 New + PMI [-fsx] | — | — | *(salvage: 92.39 ± 1.31, job 42052)* | --- | --- |
| 7 New + NegTC + fsx | — | --- | --- | — | *(salvage: 92.32 ± 1.56, job 41992)* |
| 8 RA + NegTC + fsx [-NLL] | — | --- | --- | — | — |
| 9 RA + NegTC [+TC] | — | --- | --- | — | — |
| 12 New + NegTC [-fsx] | — | --- | --- | — | *(salvage: 92.58 ± 1.48, job 42062)* |

Notes
- `s5 (n=9)`: 9 out of 10 rosch categories evaluated — the 10th
  (`rosch-bird`) was missed because that eval was launched in the brief
  window before the abs-path fix and the bird CSV is in the legacy
  abs-path-embedded form A. The 8 others were md5-hash-truncated; the
  rosch builder now resolves those via `_resolve_full_basename`.
  `n=9` counts `rosch-clothing`/`fruit`/`furniture`/`sport`/`toy`/
  `vegetable`/`vehicle`/`weapon` plus `carpenters-tool`.

## rosch / gemma-2-2b-it / GenROC

Trained on `membership-sans-rosch-v0-all`; eval'd on the 10 rosch
categories (all OOD).

| Method | Raw | PMI self | PMI base | Neg self | Neg base |
| --- | --- | --- | --- | --- | --- |
| 0 Base | 64.67 ± 2.90 | 72.31 ± 3.67 | — | 79.83 ± 3.43 | — |
| 1 SFT labelonly 10% | 84.32 ± 1.51 | — | 83.26 ± 1.64 | --- | --- |
| 2 RankAlign | 79.21 ± 2.80 | — | 78.17 ± 3.22 | --- | --- |
| 3 New + fsx [-TC] | — | — | *(salvage: 81.79 ± 2.39, job 42077)* | --- | --- |
| 4 New + PMI + fsx | — | — | *(salvage: 82.09 ± 2.68, job 42065)* | --- | --- |
| 5 RA + PMI + fsx [-NLL] | 74.77 ± 2.88 (n=9) | — | 83.78 ± 2.15 (n=9) *(salvage n=10: 82.34 ± 2.41, job 42085)* | --- | --- |
| 6 RA + PMI [+TC] | 76.66 ± 3.23 | — | 83.95 ± 2.59 | --- | --- |
| 11 New + PMI [-fsx] | 77.10 ± 3.68 | — | 84.97 ± 2.63 | --- | --- |
| 7 New + NegTC + fsx | — | --- | --- | — | *(salvage: 81.69 ± 2.26, job 42069)* |
| 8 RA + NegTC + fsx [-NLL] | — | --- | --- | — | — |
| 9 RA + NegTC [+TC] | — | --- | --- | — | — |
| 12 New + NegTC [-fsx] | 77.93 ± 3.38 | --- | --- | — | 83.55 ± 1.74 |

Note: `s5 (n=9)` missing one category (same hash-truncation residue —
re-eval not needed for paper, salvage already captured the aggregates).

## persona-v1 / gemma-2-2b-it / GenROC

Trained on `persona-v1-all`; eval'd on 6 persona-v1 personas (3 ID +
3 OOD). Two cells (s4, s12) currently being evaluated by jobs
**41974** and **42095** — should fill to 6/6 within ~50 minutes of this
snapshot.

| Method | Raw | PMI self | PMI base | Neg self | Neg base |
| --- | --- | --- | --- | --- | --- |
| 0 Base | 48.56 ± 5.17 | 57.40 ± 3.01 | --- | 75.82 ± 3.28 | --- |
| 1 SFT labelonly 10% | 54.84 ± 8.84 | — | 59.41 ± 8.76 | --- | --- |
| 2 RankAlign | 78.30 ± 10.26 | — | 86.43 ± 6.93 | --- | --- |
| 3 New + fsx [-TC] | 66.87 ± 11.84 | — | 73.16 ± 10.87 | --- | --- |
| 4 New + PMI + fsx | 65.33 ± 11.90 (n=5) | — | 71.57 ± 12.22 (n=5) | --- | --- |
| 5 RA + PMI + fsx [-NLL] | 66.40 ± 13.27 | — | 72.57 ± 11.93 | --- | --- |
| 6 RA + PMI [+TC] | 74.02 ± 10.68 | — | 85.65 ± 6.82 | --- | --- |
| 11 New + PMI [-fsx] | 60.42 ± 11.93 | — | 65.31 ± 12.79 | --- | --- |
| 7 New + NegTC + fsx | 58.97 ± 11.77 | --- | --- | — | 70.62 ± 10.32 |
| 8 RA + NegTC + fsx [-NLL] | — | --- | --- | — | — |
| 9 RA + NegTC [+TC] | — | --- | --- | — | — |
| 12 New + NegTC [-fsx] | 61.05 ± 11.36 | --- | --- | — | 77.65 ± 8.15 |

## persona-v1 / gemma-2-9b-it / GenROC

Same setup, 9b-it base. **Six of the 12 trained settings are still
training** as of 19:30 local (jobs 41979, 41989, 42009, 42019, 42049,
42059 — see `docs/overnight_progress.md`); those rows will fill in
automatically once their chained evals complete over the next 3-9 h.

| Method | Raw | PMI self | PMI base | Neg self | Neg base |
| --- | --- | --- | --- | --- | --- |
| 0 Base | 42.96 ± 5.31 | 40.88 ± 5.40 | --- | 57.19 ± 2.20 | --- |
| 1 SFT labelonly 10% | — | — | — | --- | --- |
| 2 RankAlign | 85.19 ± 7.54 | — | 95.43 ± 2.55 | --- | --- |
| 3 New + fsx [-TC] | — | — | — | --- | --- |
| 4 New + PMI + fsx | — | — | — | --- | --- |
| 5 RA + PMI + fsx [-NLL] | 67.84 ± 11.06 | — | 85.91 ± 6.69 | --- | --- |
| 6 RA + PMI [+TC] | 69.08 ± 11.58 | — | 83.72 ± 7.72 | --- | --- |
| 11 New + PMI [-fsx] | 62.60 ± 10.87 | — | 71.83 ± 10.52 | --- | --- |
| 7 New + NegTC + fsx | — | --- | --- | — | — |
| 8 RA + NegTC + fsx [-NLL] | — | --- | --- | — | — |
| 9 RA + NegTC [+TC] | — | --- | --- | — | — |
| 12 New + NegTC [-fsx] | 62.04 ± 10.60 | --- | --- | — | 76.64 ± 8.17 |

## What's not covered here

- **ifeval** (gemma-2-2b / 2b-it / 9b-it): zero score CSVs on disk.
  All ifeval × 9b-it cells were cancelled earlier today; ifeval × 2b /
  2b-it were never re-launched. The v7 ifeval table-builder is not
  written yet (only rosch + persona-v1 v7 builders exist).
- **humaneval-v2.1correct-upper × gemma-4-31B-it × s13**: just
  launched (job 42114); train running, no CSVs yet.

## Reproducing these tables

```bash
source /u/jdr/venvs/venv_lexcons/bin/activate
cd /datastor1/jdr/gv-gap/rankalign

ROSCH_MODEL=9b-it    ROSCH_METRIC=gen_roc python scripts/_build_rosch_table_v7.py
ROSCH_MODEL=2b-it    ROSCH_METRIC=gen_roc python scripts/_build_rosch_table_v7.py
PERSONA_BASE=2b-it   PERSONA_METRIC=gen_roc python scripts/_build_persona_v1_table_v7.py
PERSONA_BASE=9b-it   PERSONA_METRIC=gen_roc python scripts/_build_persona_v1_table_v7.py
```

CSVs land in `metrics-from-scores/` automatically:
- `metrics-from-scores/rosch_v7_{model}_gen_roc_table_long.csv` (per-task)
- `metrics-from-scores/rosch_v7_{model}_gen_roc_table_cells.csv` (cell aggregates)
- same pattern for persona

Other supported metrics: `pearson`, `spearman`, `val_roc`, `val_acc`.

## Once more cells finish: how to fold in the missing rows

1. **Salvaged rosch rows** (s4/s7/s11/s12 9b-it, s3/s4/s5/s7 2b-it): the
   cell-aggregate numbers are in [`docs/salvaged_rosch_metrics.md`](salvaged_rosch_metrics.md).
   Either:
   - hand-paste the row into the table builder's output, OR
   - re-run those eight evals to regenerate the per-pair CSVs (would
     take ~8 GPU-hours and isn't necessary for paper numbers).
2. **In-flight 2b-it persona evals** (jobs 41974 + 42095): re-run the
   persona builder when those exit; cells will go from n=5 / n=3 to n=6
   automatically.
3. **9b-it persona trains in flight** (s1, s3, s4, s7, s11, s12 — six
   trains, will produce six chained evals): re-run the 9b-it persona
   builder when those land.
4. **9b-it membership s1**: train 42021 will produce one rosch eval (s1
   row) when it finishes.
