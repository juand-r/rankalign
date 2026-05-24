#!/bin/bash
# Smoke for --shape-budget-mode global. Same args as _smoke_bc_new.sh
# (fsx ON, ppd OFF) plus --shape-budget-mode global. Models go to
# /datastor2 because /datastor1 is chronically full
# (see .cursor/rules/save-models-to-datastor2.mdc).
set -e
source /u/jdr/venvs/venv_lexcons/bin/activate
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== smoke (GLOBAL: ranking_loss_ref_fix.py @ HEAD, --shape-budget-mode global) ==="
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
    --models-dir /datastor2/jdr/rankalign/models2-smoke-bc-global \
    --force-same-x \
    --shape-budget-mode global

echo "=== smoke GLOBAL done ==="
