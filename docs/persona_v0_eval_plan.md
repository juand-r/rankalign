# Persona-v0 trained-model eval plan

This is the eval plan for the 18 finetuned `persona-v0` checkpoints (9 training
variants × {`google/gemma-2-9b-it`, `google/gemma-2-2b-it`}). It captures the
TC-flavor policy, the eval flags, and the auto-fire mechanism that submits
evals as soon as training finishes.

## Training (already submitted)

See `scripts/run_train_persona_v0.sh` for the launcher and
`docs/IMPORTANT-RESEARCH-PLAN.md` for the rationale behind the 9-variant set
(SFT baseline; vanilla RankAlign; comb+log-odds+force-same-x; and the 2×3
TC sub-grid covering self-TC and neg-TC × {`comb+fsx`, `pref-only+fsx`,
`pref-only`}).

Job IDs are recorded in:

- `overnight/persona_v0_train_jobids_gemma-2-9b-it.txt`
- `overnight/persona_v0_train_jobids_gemma-2-2b-it.txt`

Each variant trains for 3 epochs; we eval the **`epoch2`** checkpoint only
(matches the project convention used in `run_eval_membership_trained_models.sh`).

## Eval policy

All eval jobs go through `scripts/run_eval_persona_v0_trained.sh`, which
submits one slurm job per (variant × TC flavor). Every job runs all 8
`persona-v0-<slug>` test tasks sequentially via `scripts/run_eval_semi.sh`.

Common flags for every eval job:

```text
--base-typcorr --base-model <BASE> --log-odds --disc-shots-zero
```

- `--base-typcorr --base-model <BASE>`: the typicality reference is the *base
  instruct model* (`google/gemma-2-9b-it` or `google/gemma-2-2b-it`),
  matching the offline-TC training convention. See
  `docs/IMPORTANT-RESEARCH-PLAN.md` §3 / "Methodology fix".
- `--log-odds`: project convention for evals (matches the validator log-odds
  setting used during training for the comb+vlo variants and is harmless
  elsewhere).
- `--disc-shots-zero`: matches the persona-v0 baseline eval and the original
  Perez et al. methodology. Few-shot disc exemplars are still TBD; see
  `docs/datasets/persona_v0_notes.md`.

### TC-flavor matching policy

For variants that did **not** train with TC (i.e. SFT, vanilla RankAlign,
New+fsx), we run **both** self+base and neg+base evals — the cross-flavor
comparison on a non-TC-trained model is a useful baseline.

For variants that **did** train with TC, we only run the **matched** eval
flavor (self-trained → self+base; neg-trained → neg+base). Cross-TC eval on a
TC-trained model is not informative for the §3 research question.

| # | Variant                     | TC during train | Eval flavor(s)              | Jobs |
|---|------------------------------|-----------------|------------------------------|------|
| 1 | SFT-lo                       | none            | self+base, neg+base          | 2    |
| 2 | RankAlign                    | none            | self+base, neg+base          | 2    |
| 3 | New+fsx                      | none            | self+base, neg+base          | 2    |
| 4 | New+fsx+selfTC               | self            | self+base                    | 1    |
| 5 | RankAlign+fsx+selfTC         | self            | self+base                    | 1    |
| 6 | RankAlign+selfTC             | self            | self+base                    | 1    |
| 7 | New+fsx+negTC                | neg             | neg+base                     | 1    |
| 8 | RankAlign+fsx+negTC          | neg             | neg+base                     | 1    |
| 9 | RankAlign+negTC              | neg             | neg+base                     | 1    |
|   | **Per base model**           |                 |                              | **12** |
|   | **Across both bases**        |                 |                              | **24** |

Each eval job has 1 GPU, 6 CPU, 60GB, 2-hour walltime. Each job runs the 8
persona test tasks sequentially.

Score CSV prefixes:

- `--self-typcorr --base-typcorr` → `basetyp-<task>...csv`
- `--neg-typcorr  --base-typcorr` → `basetypneg-<task>...csv`

These are distinct from the baseline (untrained) eval prefixes (`self-` /
`neg-`), so trained-model scores live alongside baseline scores in the same
results directories without collision.

## Auto-fire mechanism

`scripts/schedule_evals_after_training.sh` submits two slurm wrapper jobs:

1. **Wrapper for 9b-it evals** — `--dependency=afterany:<all 9 train jobs for 9b-it>`,
   on completion runs `bash scripts/run_eval_persona_v0_trained.sh google/gemma-2-9b-it`.
2. **Wrapper for 2b-it evals** — same with the 2b-it train jobs and base model.

Each wrapper is a tiny CPU-only job (1 CPU, 4G, 30 min) that exists just to
fire the eval launcher when its training set finishes. The eval launcher
itself submits 12 GPU jobs as separate slurm jobs (so wrapper exit is fast).

`afterany` (not `afterok`) is used intentionally: if a single training variant
fails, we still want the other 8 variants' evals to fire. The eval launcher
already skips missing model dirs gracefully, so a failed train job only
removes itself from the eval matrix.

Wrapper job IDs and any captured stdout are written to:

- `overnight/persona_v0_eval_wrapper_jobids.txt`

## Manual launch (alternative)

If the auto-fire wrapper doesn't trigger or you want to launch evals manually
after verifying training output:

```bash
bash scripts/run_eval_persona_v0_trained.sh                        # gemma-2-9b-it
bash scripts/run_eval_persona_v0_trained.sh google/gemma-2-2b-it   # gemma-2-2b-it
```

The launcher is idempotent at the *task* level (`run_eval_semi.sh` skips
tasks whose score CSVs already exist), so re-runs are safe.

## After evals complete

Score files land under:

```text
results/<task>/scores/<base|trained-model-name>/<prefix><task>...csv
```

The downstream analysis (per-task ROC-AUC for gen / val / disc, paired
bootstrap, 4-table results format) follows the standard pattern documented in
`docs/results_table_format.md` and `docs/per_task_bootstrap_analysis.md`.
