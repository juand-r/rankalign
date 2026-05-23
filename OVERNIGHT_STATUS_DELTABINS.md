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

(appended below as I work)
