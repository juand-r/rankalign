#!/bin/bash
# Tiny smoke test for --force-same-x on membership-sans-rosch-v0.
#
# Usage:
#   bash scripts/run_smoke_fsx.sh on   # --force-same-x
#   bash scripts/run_smoke_fsx.sh off  # default (no fsx)

set -e

MODE="${1:-}"
PPD_FLAG=""
case "$MODE" in
    on)        FSX_FLAG="--force-same-x" ;;
    off)       FSX_FLAG="" ;;
    on-ppd)    FSX_FLAG="--force-same-x"; PPD_FLAG="--per-prompt-delta" ;;
    *)         echo "Usage: $0 on|off|on-ppd"; exit 1 ;;
esac

source /u/jdr/venvs/venv_lexcons/bin/activate
cd "$(dirname "$0")/.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "================================================================"
echo "Smoke test: force-same-x = $MODE"
echo "Task: membership-sans-rosch-v0  Model: google/gemma-2-2b"
echo "================================================================"

python scripts/ranking_loss_ref_fix.py \
    --model google/gemma-2-2b \
    --task membership-sans-rosch-v0 \
    --train_g_or_d g \
    --split_type random \
    --num_epochs 1 \
    --total_samples 30 \
    --delta 0.15 \
    --delta-bins 10 \
    --semi-supervised 0.1 \
    --nll_validator_weight 1 \
    --nll_generator_weight 1 \
    --preference_loss_weight 1 \
    --validator-log-odds \
    --all \
    --no-upload-hf \
    --no-wandb \
    --debug \
    --models-dir ./models-smoke-fsx \
    $FSX_FLAG $PPD_FLAG

echo ""
echo "Smoke test (fsx=$MODE) done."
