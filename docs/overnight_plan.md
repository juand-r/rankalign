# Overnight Plan (2026-05-24)

Concrete plan for executing the overnight job list in
[`overnight_run_instructions.md`](overnight_run_instructions.md).
Companion to [`overnight_progress.md`](overnight_progress.md) for
live status.

## Inventory of work

Settings × models × datasets:

| Phase | Settings                | Models                              | Datasets                                       | Cells |
|-------|--------------------------|-------------------------------------|------------------------------------------------|-------|
| 1     | s4, s7, s2, s3, s1       | gemma-2-2b-it, gemma-2-9b-it        | persona-v1, membership-sans-rosch-v0, ifeval-concat | 30 |
| 2     | s5, s6, s11, s12         | same                                | same                                           | 24 |

Each cell = one train sbatch + one eval sbatch (eval uses
`--dependency=afterany:<train_jobid>`). So phase 1 = 60 jobs total in
queue if launched all at once (over the 32-cap), and phase 2 adds 48.

Within phase 1 I sub-divide by model:

- 1A: gemma-2-2b-it × 5 settings × 3 datasets = 15 trains + 15 evals.
- 1B: gemma-2-9b-it × 5 settings × 3 datasets = 15 trains + 15 evals.

## Order of operations

### Step 0 — Smoke validation (gate)

Wait for 41745 to start + finish (or fail). It's the smoke for
`--shape-budget-mode global` plus the per-item disc-gating perf fix
that we're about to lean on for many real jobs.

If 41745 is still PENDING after 60 min: launch a tiny replacement
smoke with smaller resource ask (4 cpu, 16G, 30 min) so it gets
scheduled. While waiting, prepare scripts. Don't gate phase 1 on
this — the math of the perf fix is `if A: forward; else: zeros` which
is structurally safe; but verify before phase 2.

### Step 1A — gemma-2-2b-it priority sweep (15 + 15 jobs)

For each of the 5 priority settings (s4, s7, s2, s3, s1) × 3 datasets:

- Train: 1 GPU, 5h walltime, 4 cpu, 32G mem.
  - For ifeval-concat: 1 GPU is fine for 2b-it (small model + LoRA
    gating); use 5h walltime.
- Eval: 1 GPU, 2h walltime, 4 cpu, 32G mem; chained on
  `afterany:<train_jobid>`.

Submit all 30 in one go. Slurm will schedule what it can. Eval jobs
sit PENDING with reason `Dependency` until trains release.

Note on script reuse: `run_train_persona_v1_fix1.sh` already exists;
it covers s3/s4/s7 (the comb variants). I need new wrappers for s1
(sft) and s2 (rankalign) using `--script ranking_loss_ref_fix.py`.

### Step 1B — gemma-2-9b-it priority sweep (15 + 15 jobs)

Launched after 1A queue drops below ~15 (so total stays under 32).

- Train, persona-v1: 1 GPU, 8h walltime.
- Train, membership-sans-rosch-v0: 1 GPU, 8h walltime.
- Train, ifeval-concat: 2 GPUs (model parallel via `device_map=auto`),
  10h walltime.
- All evals: 1 GPU, 2h, dep=afterany.

### Step 2 — Less important settings (24 + 24 jobs)

Only after phase 1 queue is ~empty. s5, s6, s11, s12 × 2 models × 3
datasets. Same scheme.

## Dependency-chain pattern

Every train submission yields a job-id which I capture and reuse for
the eval submission:

```
TRAIN_OUT=$(run 1 5 --cpu 4 --mem 32G scripts/run_train_semi.sh ...)
TRAIN_JOBID=$(echo "$TRAIN_OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+$')

# Eval is sbatch directly so we can pass --dependency
sbatch --dependency=afterany:$TRAIN_JOBID --partition=allnodes --gres=gpu:1 \
       --cpus-per-task=4 --mem=32G --time=2:00:00 \
       --output=/datastor2/jdr/logs/%j.out --error=/datastor2/jdr/logs/%j.err \
       --wrap "test -d $MODEL_PATH || { echo SKIP; exit 0; }; \
               cd $REPO && PYTHONUNBUFFERED=1 scripts/run_eval_semi.sh $MODEL_PATH \
               $EVAL_FLAGS -- $TASKS"
```

`afterany` (not `afterok`) so we still try eval if training got
walltime-killed mid-final-epoch — most of our trainings save the
LoRA at every epoch, so a walltime kill at e.g. epoch2 still leaves
epoch1 on disk that we can eval.

The eval `--wrap` body checks for the model dir; if missing (training
died before any epoch saved), it exits 0 with `SKIP` instead of
crashing.

## Eval recipe per setting

Per IRP §3 we always test with both base-typcorr and the matching
self/neg variant. Concretely each eval sbatch runs
`scripts/run_eval_semi.sh <model> --base-typcorr --base-model <base>
--<self|neg>-typcorr --log-odds [--disc-shots-zero?] -- <tasks...>`.

| # | TC at eval                        |
|---|-----------------------------------|
| 1 | both `--self-typcorr` and `--neg-typcorr` (separately) + `--base-typcorr` |
| 2 | both                              |
| 3 | both (since training had no TC)   |
| 4 | `--self-typcorr` + `--base-typcorr` |
| 5 | `--self-typcorr` + `--base-typcorr` |
| 6 | `--self-typcorr` + `--base-typcorr` |
| 7 | `--neg-typcorr` + `--base-typcorr` |
|11 | `--self-typcorr` + `--base-typcorr` |
|12 | `--neg-typcorr` + `--base-typcorr` |

`disc-shots`: training uses `--disc-shots few` per user; at eval the
`run_eval_semi.sh` default is also `few` — match it. So I do NOT
pass `--disc-shots-zero` to the eval wrapper.

For settings #1/#2/#3 (no TC during train) I'd ideally run BOTH
`--self-typcorr` and `--neg-typcorr` evals so we can compare, but
that doubles the eval count. Compromise: run `--self-typcorr` only
for #1/#2/#3 to cap the eval ask. (We can re-eval with neg later
if needed.)

## Time and resource budget

Best-case rough wallclock:

- 2b-it persona-v1 train: ~1.5h. ifeval-concat: ~3h. membership: ~2h.
- 9b-it persona-v1 train: ~3h. ifeval-concat: ~6h on 2 GPUs.
  membership: ~5h.
- Eval: ~10-30 min per task; per-cell evals (6-21 tasks) take
  20 min to 2h.

If phase 1 fits within 8h wallclock and uses ≤20 GPUs at a time,
we're fine. Phase 2 starts when phase 1 evals are mostly done.

## Things to watch for

- /datastor1 disk: should be fine since all models go to /datastor2,
  but check on every loop tick.
- 41730 walltime kill at ~04:35 — fine, we have epoch1 already.
- 41745 if it never starts: tiny replacement smoke at 03:30+1h.
- Python errors specific to new flags (`--per-prompt-delta`,
  `--shape-budget-mode global`): if any train job logs these,
  diagnose immediately.
- Slurm node-006 has been spotty; consider `--exclude=slurm-node-006`
  if a job is pathologically slow.
