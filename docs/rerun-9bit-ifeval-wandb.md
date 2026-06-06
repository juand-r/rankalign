# Rerun: gemma-2-9b-it × ifeval (delta-bins) WITH wandb

**Created 2026-06-06.**

## Why

The paper's **gemma-2-9b-it × ifeval delta-bins** checkpoints (settings s1, s2, s3,
s4, s7, s13) were **pod-trained with `--no-wandb`**, so there are **no training
curves in the cloud** for them. (The only finished wandb runs for this
model×task are the *fixed-delta* April mll batch — s1/s2/s7/s8 — see
[`inventory-jun1-2026/WANDB_RUN_MAP.md`](inventory-jun1-2026/WANDB_RUN_MAP.md).)

These reruns reproduce the same delta-bins configs through the **mll launcher**,
which keeps wandb **ON** by default, so we get the training curves.

## What gets launched

Per setting, [`scripts/_overnight_launch.sh`](../scripts/_overnight_launch.sh)
submits:
1. a **train** job via `~/.local/bin/run` → `scripts/run_train_semi.sh`
   (wandb ON, `--delta-bins 10`), and
2. a dependent **eval** job (`--dependency=afterany`).

Settings: **s1, s2, s3, s4, s7, s13** (6 cells). gemma-2-9b-it ifeval is 2 GPUs,
`TRAIN_HOURS=30`, `EVAL_HOURS=5` per cell (from `_overnight_launch.sh`).

## How the reruns are kept distinct from the paper artifacts

| Artifact | Paper (original) | Rerun |
|---|---|---|
| Checkpoints | `/datastor2/jdr/rankalign/models2` (+ HF `TAUR-dev/…`) | **`/datastor2/jdr/rankalign/models2-rerun-wandb`** |
| Eval score CSVs | `outputs/`, `outputs_gemma4_from_pod-v7/ra9b_ifeval` | **`/datastor2/jdr/rankalign/outputs-rerun-wandb`** |
| WandB run name | auto (`gemma-2-9b-it-ifeval-concat-g-delta0.15-bins10-…-lr1e-05`) | **`rerun-wandb-20260606-gemma-2-9b-it-ifeval-<setting>`** |

Filter the cloud project `juand-r/rankalign` by the **`rerun-wandb-`** name prefix
to see exactly these runs.

## Run it

```bash
# dry-run first (prints the exact train/eval commands, submits nothing)
DRYRUN=1 bash scripts/run_rerun_9bit_ifeval_wandb.sh

# submit all 6 settings
bash scripts/run_rerun_9bit_ifeval_wandb.sh

# subset
SETTINGS="s1 s13" bash scripts/run_rerun_9bit_ifeval_wandb.sh
```

Env overrides: `MODELS_DIR`, `OUTPUTS_DIR`, `WANDB_PREFIX`, `SETTINGS`, `DRYRUN`.

## Plumbing changes made for this (2026-06-06)

The trainer `ranking_loss_ref_fix.py` already accepts `--wandb_run_name`
(L2409); it was just not exposed by the launchers. Added a one-line passthrough
in each:

- **`scripts/run_train_semi.sh`** — parse `--wandb_run_name <name>` and forward
  it to the python trainer.
- **`scripts/_overnight_launch.sh`** — if env `WANDB_RUN_NAME` is set, append
  `--wandb_run_name "$WANDB_RUN_NAME"` to `COMMON_FLAGS`.
- **`scripts/run_rerun_9bit_ifeval_wandb.sh`** (new) — the wrapper that loops the
  6 settings, exporting `MODELS_DIR`, `OUTPUTS_DIR`, and a per-setting
  `WANDB_RUN_NAME`.

These changes are backward-compatible: when `WANDB_RUN_NAME` is unset the trainer
auto-generates the name exactly as before.

## Verify (dry-run output, 2026-06-06)

`DRYRUN=1 SETTINGS="s1 s13"` produced, e.g. for s13:

```
run 2 30 --cpu 4 --mem 96G scripts/run_train_semi.sh google/gemma-2-9b-it \
  ifeval-concat sft labelonly 0.1 --script ranking_loss_ref_fix.py \
  --disc-shots zero --delta-bins 10 --max-seq-len 1024 --no-force-same-x \
  --consistency-ft --gradient-checkpointing \
  --models-dir /datastor2/jdr/rankalign/models2-rerun-wandb \
  --wandb_run_name rerun-wandb-20260606-gemma-2-9b-it-ifeval-s13
```
