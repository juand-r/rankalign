# TASK 2 — Eval inventory: where the scores and metrics live

**Compiled 2026-06-02.** Three things you asked for, per eval: (1) which eval was done,
(2) where the `scores_*.csv` files are (and where they DON'T exist because a run failed),
(3) where the metrics CSV is.

Companion machine-generated coverage table: **`v7/eval_inventory_v7.md`** (per
model×task×setting×epoch, count of distinct eval tasks with a `scores_` file of each
eval-time-TC type). Raw per-file rows: `_raw/eval_coverage.csv`.

---

## 1. Where the raw `scores_*.csv` files live

All under **`private_projects/rankalign/outputs_gemma4_from_pod-v7/`** (local) and mirrored on
**`/datastor2/jdr/rankalign/outputs_gemma4_from_pod-v7/`**. 21,338 score CSVs total. Subdir map:

| Subdir | Model | Task(s) | Notes |
|--------|-------|---------|-------|
| `correct_upper_s1s4s7/` | gemma-4-31B-it | humaneval-cu | s2(e1), s3/s11/s12(e2) full-name files |
| `correct_multi_s1s2/`, `correct_multi_s3s4s7/` | gemma-4-31B-it | humaneval-cm | s1/s2/s11/s12(e2) + base |
| `s13_ep0/` | gemma-4-31B-it | humaneval-cu | s13 **epoch0** (self+neg, 82 each) |
| `s1_sft_v6/` | gemma-4-31B-it | humaneval-cu | s1 **v6** (basetyp+basetypneg, 82 each, epoch1) |
| `ra9b_ifeval/` | gemma-2-9b-it | ifeval | `eval_model_sN` scheme; s1/s2/s3/s4/s7 |
| `ra9b_persona_member/` | gemma-2-9b-it | persona + membership(rosch) | `eval_model_sN`; s1/s2/s3/s4/s7 |
| `qw35_ifeval/` | qwen3.5-9b | ifeval | s1/s2/s4/s7/s13 |
| `qw35_persona_member/` | qwen3.5-9b (+ base refs) | persona + membership | **grab-bag — see warning below** |
| `outputs_gemma4_from_pod-v7b/{v7b,qw35}_*` | gemma-2-9b-it, qwen | ifeval/persona/member | v7b = fixed-delta-0.15 re-runs (partial) |

> **`qw35_persona_member/` is a grab-bag (17,686 files).** Only a small fraction are the
> actual qwen persona/membership evals. It also holds: **17,592 base-model reference**
> re-runs (massively duplicated across 8,629 timestamps), and **12,429 legacy cross-task
> files** from *other* experiments that have nothing to do with the qwen paper rows —
> plausibleqa (8107), hypernym (1876), ambigqa (1530), gsm8k (666). Those are confirmed
> not part of the v7 paper eval (see `_raw/eval_unparsed.txt`). Treat this dir with care.

### Filename → fields (two schemes)
- **Full-name** (gemma-4 cu/cm; some Qwen): `scores_{prefix}v7-…-delta{D}-epoch{E}--{task}…fix1_{evaltask}_test_…csv` — epoch is in the name.
- **`eval_model_sN`** (ra9b, qw35): `scores_{prefix}eval_model_s{N}_{evaltask}_test_…csv` — **model+task from the subdir, epoch NOT in the name.** Evaluated epoch must be read from the eval wrapper or `eval_coverage_matrix.md`.
- `{prefix}` = eval-time TC: `self-`=PMI self, `neg-`=Neg self, `basetyp-`=PMI base, `basetypneg-`=Neg base, none=raw-only. Full convention: `../score_filename_convention.md`.

---

## 2. Where the computed metrics CSVs live

### `metrics-from-scores/` — the gemma-4 humaneval paper tables (authoritative)
For `humaneval_v2.1correct-upper` and `humaneval_v2.1correct-multi`, model `g4-31B-it`, each of
5 metrics (`gen_roc`, `pearson`, `spearman`, `val_acc`, `val_roc`) × `{_table_cells.csv,
_table_long.csv}`:
```
metrics-from-scores/humaneval_v2.1correct-upper_g4-31B-it_<metric>_table_cells.csv
metrics-from-scores/humaneval_v2.1correct-upper_g4-31B-it_<metric>_table_long.csv
metrics-from-scores/humaneval_v2.1correct-multi_g4-31B-it_<metric>_table_*.csv
```
Plus provenance for the dedup:
- `humaneval_v2.1correct-*_g4-31B-it_files_used.csv` — exactly which `scores_` files fed each cell.
- `humaneval_v2.1correct-*_g4-31B-it_dups_collapsed.csv` — which duplicate re-runs were collapsed.

Also here: `persona_v1_{gemma-2-2b,gemma-2-2b-it,gemma-2-9b-it}_{all,id,ood}_{gen_roc,pearson}_table_*.csv`
and `rosch_9b-it_gen_roc_table_*.csv`.

### `output-metrics/` — older / other-task metrics
persona_v0, gsm8k_v1 base summaries, humaneval_v1 length/typicality plots. Mostly pre-paper.

### `docs/` markdown result tables (human-readable snapshots — what's in the paper)
| File | What |
|------|------|
| `humaneval_cu_v7_tables_2026-05-25.md` | **Main gemma-4 correct-upper table** (Base, s1, s2 RankAlign, s3/4/7 …) |
| `humaneval_cm_v7_tables_2026-05-25.md` | gemma-4 correct-multi table |
| `humaneval_cu_s13_ep0_tables_2026-05-25.md` | s13 epoch0 (cu) |
| `humaneval_cu_s1_sft_v6_tables_2026-05-25.md` | s1 SFT (cu) from v6 model |
| `pod-results-{genroc,pearson,spearman,valacc,valroc}-20260525.md` | gemma-2 ra9b + qwen, 5 metrics |
| `morning-results-*-20260525.md` | later snapshot of the same |
| `v7_ra9b_results_2026-05-25.md` | gemma-2-9b-it ifeval/persona/member |
| `v7_persona_table_*.md`, `v7_rosch_table_*.md` | gemma-2 persona / rosch |
| `salvaged_rosch_metrics.md` | **recovered-from-stdout** rosch metrics (see §3) |

---

## 3. Evals that FAILED / have no scores files (your "bug → only stdout" case)

### The rosch OSError-36 crash (2026-05-24) — metrics salvaged from stdout, NO score CSVs
Eight overnight Slurm rosch eval jobs crashed mid-run with
`OSError [Errno 36] File name too long` (filenames for high-flag cells exceeded the 255-char
`NAME_MAX`; fixed later in commits `6890d295` + `a42558a9`). **The eval itself completed for
every task** — only the per-pair CSV *write* failed. So:
- **No `scores_*.csv` exist** for these 8 cells.
- The per-task aggregate metrics (gen_roc, disc_roc, disc_acc, corr, spearman) were printed to
  the job `.out` logs and **scraped** by `scripts/_extract_rosch_metrics_from_logs.py` into
  **`docs/salvaged_rosch_metrics.md`** (8 cells, 10/10 rosch tasks each).
- Jobs `41982,42052,42065,42077,42085` = `--self-typcorr --base-typcorr` → **PMI base** column;
  `41992,42062,42069` = `--neg-typcorr --base-typcorr` → **Neg base** column.
- These salvaged numbers are **TC-corrected** (NOT raw) — the doc corrects an earlier mislabel.
- Re-running is only needed if you want the per-pair scatter CSVs; the table numbers are recovered.

### Other known gaps (from `eval_coverage_matrix.md`, ~May 24–25 snapshot)
- **All s13 cells** were `done(e0)` or `running` and showed `0/N` scores at snapshot time —
  s13 humaneval-cu later produced epoch0 scores (`s13_ep0/`, 82+82); s13 qwen ifeval/member
  produced partial. s13 was the last-minute setting; coverage is thinnest here.
- gemma-4 **correct-multi s12** Neg base shows only **46/82** in the current files (incomplete).
- gemma-4 **correct-upper s11/s12** show **81/82** (one task short).
- gemma-4 **correct-upper s2 Neg base = 79/82** (humaneval_1/10/100 fail only in `basetypneg` mode).

---

## 4. How to regenerate
- Coverage table: `python _build_eval_inventory.py` (reads `_raw/scores_v7_local_listing.txt`).
- Refresh the raw listing: `find outputs_gemma4_from_pod-v7 outputs_gemma4_from_pod-v7b -name 'scores_*.csv' -printf '%P\n' > _raw/scores_v7_local_listing.txt`.

## 5. [PENDING datastor1] / follow-ups
- The **gemma-2-2b / gemma-2-2b-it / gemma-2-9b-it** scores (persona/membership/ifeval + older
  v6 tasks) live in **`/datastor2/jdr/rankalign/outputs/`** — **20,402** `scores_*.csv` confirmed
  there (gemma-2-2b 6,735; gemma-2-2b-it 3,244; gemma-2-9b-it 10,041; gemma-4-31B-it 164). Mixed
  v6+v7 and many tasks. NB: `ls scores_*.csv` blows the arg limit there — use `find … -name`.
  Their metrics are already computed (`metrics-from-scores/persona_v1_gemma-2-2b*` +
  `pod-results-*.md`); a per-cell parse of this dir (like `_build_eval_inventory.py`) is a
  follow-up if you want the full coverage grid for the gemma-2 family.
- ifeval eval-task count is **99 prompts** in ra9b files but the s13 scope used **21** — confirm
  which subset each paper table used (OOD=21 vs full=99). See `feedback_ood_vs_id_eval`.
- Evaluated epoch for all `eval_model_sN` (ra9b/qw35) cells must be cross-checked against the
  eval wrapper config — filenames don't carry it. TASK 3 resolves the paper-relevant ones.
