# Overnight: delta-bins sweep + main 9-job training run

Last update: 2026-05-23 00:55 UTC-5 (start)

## Plan

1. Wait for 15-job delta-bins sweep (jobs 41454–41468) to hit 4h walltime.
2. Launch eval on each of 15 models at latest epoch dir.
   - 9b-it (41454–58): epoch0 (only 1 epoch will finish)
   - 2b (41459–63), 2b-it (41464–68): epoch1 if available, else epoch0
   - Eval flags: `--self-typcorr --base-typcorr --base-model <BASE> --log-odds`
   - 6 persona-v1 tasks (3 ID: psychopathy, machiavellianism, narcissism;
     3 OOD: desire-to-create-allies, interest-in-music, interest-in-science)
   - disc-shots = few (matches training)
3. Build tables: per (base_model, delta-bins, split) with cols
   {gen_roc, val_acc, val_roc, pearson} mean ± stderr across 3 tasks per split.
4. Pick best `delta-bins` value via judgment.
5. Launch 9 main training jobs:
   - persona-v1 × {gemma-2-9b-it (8h), gemma-2-2b-it (4h), gemma-2-2b (4h)}
     × {#3 New, #4 New+selfTC, #7 New+negTC}
   - All disc-shots=few, picked DELTA_BINS, ranking_loss_ref_fix.py
6. Auto-launch evals when each main job ends (latest-epoch checkpoint):
   - #3 → self+base, neg+base (2 evals)
   - #4 → self+base only
   - #7 → neg+base only
   - Total: 12 main eval jobs.
7. Build final results table for the 9 main runs.

## Constraints

- Max 20 GPUs concurrent.
- Nothing big on login node — everything via sbatch / `~/.local/bin/run`.
- Status updates appended to this file as I make progress.

## Sweep job summary (snapshot at start)

| Job   | Base              | bins | δ      | epochs at T+2h21m |
|-------|-------------------|------|--------|-------------------|
| 41454 | gemma-2-9b-it     | 5    | 4.9851 | 0 (62% of ep1)    |
| 41455 | gemma-2-9b-it     | 10   | 2.4926 | 0 (64%)           |
| 41456 | gemma-2-9b-it     | 30   | 0.8309 | 0 (62%)           |
| 41457 | gemma-2-9b-it     | 50   | 0.4985 | 0 (63%)           |
| 41458 | gemma-2-9b-it     | 100  | 0.2493 | 0 (63%)           |
| 41459 | gemma-2-2b        | 5    | 0.9504 | 1                 |
| 41460 | gemma-2-2b        | 10   | 0.4752 | 1                 |
| 41461 | gemma-2-2b        | 30   | 0.1584 | 1                 |
| 41462 | gemma-2-2b        | 50   | 0.0950 | 1                 |
| 41463 | gemma-2-2b        | 100  | 0.0475 | 1                 |
| 41464 | gemma-2-2b-it     | 5    | 3.1346 | 1                 |
| 41465 | gemma-2-2b-it     | 10   | 1.5673 | 1                 |
| 41466 | gemma-2-2b-it     | 30   | 0.5224 | 1                 |
| 41467 | gemma-2-2b-it     | 50   | 0.3135 | 1                 |
| 41468 | gemma-2-2b-it     | 100  | 0.1567 | 1                 |

## Progress log

- **2026-05-23 00:55** — start. Setup done. Plan/launchers/extractor verified.
  - Wrote `scripts/run_eval_deltabins_sweep.sh` (eval launcher for sweep models).
  - Wrote `scripts/_deltabins_table.py` (results table builder; scratch).
  - Sanity-checked `summarize_scores_file.py` integration.
  - 9b-it `--merged` suffix vs 2b/2b-it plain-dir convention preserved (matches existing fix1 launcher).
- **2026-05-23 02:14** — all 15 sweep jobs hit TIMEOUT at exactly 4h walltime.
  - 9b-it (41454–58): 1 epoch trained, `epoch0` dirs + `_merged` saved.
  - 2b (41459–63), 2b-it (41464–68): 2 epochs trained, `epoch1` dirs saved.
- **2026-05-23 02:17** — submitted 15 eval jobs (41476–41490).
  - 5 9b-it use `_merged` epoch0; 5 2b + 5 2b-it use plain epoch1.
  - Flags: `--self-typcorr --base-typcorr --base-model <BASE> --log-odds`, disc-shots=few.
  - 6 persona-v1 tasks each. 2h walltime per job. All started PENDING.
- **2026-05-23 03:04** — all 15 evals COMPLETED (exit 0).
  - 2b: ~21 min/job; 2b-it: ~22-30 min/job; 9b-it: ~38 min/job.
  - Built 6-table results doc: [docs/delta_bins_sweep_results.md](docs/delta_bins_sweep_results.md).
  - **Picked bins=10** as best delta-bins value (best gen_roc(tc) ID across all 3 bases;
    best pearson(tc) ID for 2b-it and 9b-it; competitive on OOD).
- **2026-05-23 03:06** — launched 9 main training jobs with DELTA_BINS=10:
  - 41491-93: gemma-2-9b-it #3, #4, #7  (8h walltime)
  - 41494-96: gemma-2-2b-it #3, #4, #7  (4h walltime)
  - 41497-99: gemma-2-2b #3, #4, #7      (4h walltime)
- **2026-05-23 04:30** — 41499 (2b #7) was 9x slower on overloaded node-006
  (9.4 s/it vs 1.05 s/it on its siblings). Cancelled, resubmitted as 41500
  with `--exclude=slurm-node-006`. 41500 landed on node-005 and started fresh.
  - Active main jobs: 41491-98, 41500.
- **2026-05-23 07:35** — 2b/2b-it main9 jobs (41494-98) hit 4h TIMEOUT with
  epoch1 saved (= 2 epochs trained). 41500 still RUNNING.
- **2026-05-23 07:37** — submitted main9 evals for 2b-it (#3, #4, #7) -> 41501-04.
  - Caught a bug in `run_eval_main9_fix1.sh`: when sweep + main9 share the
    `--tc-self` suffix (variant 4 only), the launcher picked the wrong delta
    (alphabetically first sweep dir). Cancelled 41503; fixed launcher to
    prefer the most-recently-modified internal file on epoch ties.
- **2026-05-23 07:37** — resubmitted 2b-it #4 (41505) and submitted 2b #3+#4
  (41506-08) with the corrected launcher.
- **2026-05-23 08:02** — all 7 2b/2b-it main9 evals COMPLETED (~20 min/job).
  - 41500 (2b #7 RESUB) still mid-epoch-3, will TIMEOUT with epoch1 saved.
  - Submitted 2b #7 eval as 41509 using epoch1.
  - Wrote `scripts/_main9_table.py` to build final results table.
- **2026-05-23 08:30** — 41500 (2b #7) TIMEOUT'd with epoch1 saved.
  41509 (2b #7 eval) COMPLETED.
- **2026-05-23 09:58** — 9b-it (41491-93) saved epoch1 (2 epochs trained).
  Submitted 4 main9 evals for 9b-it: 41510-13.
- **2026-05-23 10:39** — all 4 9b-it main9 evals COMPLETED (~32 min/job).
- **2026-05-23 10:40** — final results table built:
  [docs/main9_results.md](docs/main9_results.md). Full coverage:
  9 models × 12 (variant × eval_kind) cells × 6 personas = 72 score files
  consumed; 72 sweep score files correctly filtered out by bins=10 delta match.
- **2026-05-23 ~11:06** — 9b-it main9 trains will hit 8h walltime
  (auto-terminate). Evals already done — no further action needed.

## Summary of jobs

- 15 sweep trains (41454–68): TIMEOUT, all saved.
- 15 sweep evals (41476–90): COMPLETED.
- 9 main9 trains (41491–99 + 41500 RESUB): 6 TIMEOUT confirmed, 3 (9b-it)
  still running — will TIMEOUT at 11:06am with epoch1 saved.
- 12 main9 evals (41501–13): COMPLETED.

44 jobs total. No failures, full data coverage.

## Top-level findings

- **bins=10** picked from sweep (best gen_roc(tc) ID for all 3 bases; best
  pearson(tc) ID for 2b-it/9b-it; competitive on OOD).
- **ID gen_roc(tc) winners**: variant #3 (New, no TC at train) generally
  outperforms #4 (selfTC) and #7 (negTC). Especially clear on 2b-it
  (0.979–0.980 vs 0.941–0.942) and 2b (0.951–0.959 vs 0.910–0.922).
- **OOD gen_roc(tc) winners**: neg+base TC at eval beats self+base TC at
  eval, regardless of training variant. E.g. 9b-it: #3 neg+base 0.632 vs
  #3 self+base 0.486. Pattern is consistent across all 3 bases.
- **Val side is at ceiling** on ID (val_roc=1.000 across all 9 models),
  meaning gen_roc < 1.0 reflects gen-vs-val misalignment, not val failure.

See [docs/main9_results.md](docs/main9_results.md) for the full per-base
ID/OOD tables, and [docs/delta_bins_sweep_results.md](docs/delta_bins_sweep_results.md)
for the bins sweep that motivated bins=10.
