#!/bin/bash
# Wandb re-run of the Consistency-FT (s13) setting — Qwen3.5-9B on IFEval. s13 was the one method
# skipped in the v1/v2/v3 qwen-ifeval reruns; this fills that cell. Same mll pipeline + 3-epoch /
# epoch2 convention as the other rerun cells (s1/s2/s3/s4/s7), so the s13 cell is consistent with
# them. Writes into the v3 batch dirs so the existing table tooling picks it up.
#
# EPOCHS: the ORIGINAL s13 paper checkpoint used 1 epoch (fixed delta). This rerun uses the
# launcher default (3 epochs, delta-bins 10) to match the other rerun cells. For s13 (SFT,
# --preference_loss_weight 0) there is NO preference loss, so delta is unused -> delta-bins vs
# fixed-delta does not affect the result; only the epoch count differs from the original. To
# reproduce the 1-epoch original instead, prepend EPOCHS=1 to the sbatch --export below.
#
# Two passes, like the v3 batch:
#   1) base pass  (default)  -> trains + evals WITH --base-typicality  -> scores_basetyp-/basetypneg-
#   2) NO_BASE pass          -> eval-only (model exists -> skips train) -> scores_self-/neg-
# The NO_BASE pass runs afterok the base pass.
set -euo pipefail
REPO=/datastor2/jdr/rankalign
MODELS_DIR=$REPO/models2-rerun-wandb-v3
OUTPUTS_DIR=$REPO/outputs-rerun-wandb-v3
WANDB_SUFFIX=-v3
cd "$REPO"

jid=$(sbatch --parsable --job-name="qw-s13-base" \
    --export=ALL,MODELS_DIR=$MODELS_DIR,OUTPUTS_DIR=$OUTPUTS_DIR,WANDB_SUFFIX=$WANDB_SUFFIX \
    scripts/run_qwen35_cell_mll.sbatch ifeval s13)
echo "s13 base (train + basetyp/basetypneg eval) -> job $jid"

jid2=$(sbatch --parsable --dependency=afterok:$jid --job-name="qw-s13-nobase" \
    --export=ALL,NO_BASE=1,MODELS_DIR=$MODELS_DIR,OUTPUTS_DIR=$OUTPUTS_DIR,WANDB_SUFFIX=$WANDB_SUFFIX \
    scripts/run_qwen35_cell_mll.sbatch ifeval s13)
echo "s13 nobase (eval-only self/neg) -> job $jid2 (afterok:$jid)"
echo "models -> $MODELS_DIR ; scores -> $OUTPUTS_DIR ; wandb suffix $WANDB_SUFFIX"
