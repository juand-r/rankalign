#!/bin/bash
# run_rerun_9bit_ifeval_wandb.sh
#
# Re-train google/gemma-2-9b-it x ifeval (delta-bins) for settings
# s1 s2 s3 s4 s7 s13 WITH wandb logging enabled, into a SEPARATE checkpoint dir
# and with TAGGED wandb run names — so the new runs are trivially distinguishable
# from the paper artifacts.
#
# Why this exists: the paper's gemma-2-9b-it x ifeval delta-bins checkpoints were
# pod-trained with `--no-wandb`, so there are no training curves in the cloud for
# them. These reruns reproduce the same configs through the mll launcher
# (_overnight_launch.sh -> run_train_semi.sh), which keeps wandb ON by default
# and uses `--delta-bins 10`.
#
# What it does per setting: _overnight_launch.sh submits a TRAIN job (wandb ON)
# plus a dependent EVAL job (afterany).
#
# Distinguishers (both point AWAY from the paper artifacts):
#   1. Checkpoints  -> MODELS_DIR=/datastor2/jdr/rankalign/models2-rerun-wandb
#   2. Eval scores  -> OUTPUTS_DIR=/datastor2/jdr/rankalign/outputs-rerun-wandb
#   3. WandB runs   -> name "rerun-wandb-20260606-gemma-2-9b-it-ifeval-<setting>"
#                      (project "rankalign"; filter by the "rerun-wandb-" prefix)
#
# Usage:
#   bash scripts/run_rerun_9bit_ifeval_wandb.sh                 # submit all 6
#   DRYRUN=1 bash scripts/run_rerun_9bit_ifeval_wandb.sh        # print only, no submit
#   SETTINGS="s1 s2" bash scripts/run_rerun_9bit_ifeval_wandb.sh  # subset
#
# Env overrides:
#   MODELS_DIR    checkpoint dir   (default below)
#   OUTPUTS_DIR   eval-scores dir  (default below)
#   WANDB_PREFIX  run-name prefix  (default rerun-wandb-20260606)
#   SETTINGS      space-separated  (default "s1 s2 s3 s4 s7 s13")
#   DRYRUN=1      pass through to _overnight_launch.sh (no sbatch)

set -euo pipefail

DATASET="ifeval"
MODEL="gemma-2-9b-it"
SETTINGS="${SETTINGS:-s1 s2 s3 s4 s7 s13}"

# (1) Separate checkpoint dir — never touch the paper models2/ artifacts.
export MODELS_DIR="${MODELS_DIR:-/datastor2/jdr/rankalign/models2-rerun-wandb}"
# (2) Separate eval-outputs dir — keep rerun score CSVs apart from paper scores.
export OUTPUTS_DIR="${OUTPUTS_DIR:-/datastor2/jdr/rankalign/outputs-rerun-wandb}"

# (3) WandB run-name prefix — trivially filterable in the cloud project.
PREFIX="${WANDB_PREFIX:-rerun-wandb-20260606}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo "RERUN: $MODEL x $DATASET (delta-bins, wandb ON)"
echo "  SETTINGS    = $SETTINGS"
echo "  MODELS_DIR  = $MODELS_DIR"
echo "  OUTPUTS_DIR = $OUTPUTS_DIR"
echo "  wandb name  = ${PREFIX}-${MODEL}-${DATASET}-<setting>"
echo "  DRYRUN      = ${DRYRUN:-0}"
echo "============================================================"

for s in $SETTINGS; do
    # per-setting tagged wandb run name (unique + clearly a rerun).
    export WANDB_RUN_NAME="${PREFIX}-${MODEL}-${DATASET}-${s}"
    echo
    echo ">>> $MODEL $DATASET $s   wandb=$WANDB_RUN_NAME"
    bash "$SCRIPT_DIR/_overnight_launch.sh" "$DATASET" "$MODEL" "$s"
done

echo
echo "=== all settings dispatched: $SETTINGS ==="
