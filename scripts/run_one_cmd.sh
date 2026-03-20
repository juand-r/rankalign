#!/bin/bash
#
# Run one train or one eval (for use with your job launcher, e.g. run 1 16 "bash scripts/run_one_cmd.sh ...").
#
# Usage:
#   Eval:  run_one_cmd.sh eval <task> <model_path>
#          model_path is relative to repo root (e.g. models/v6-...) or absolute
#   Train: run_one_cmd.sh train <task> [--typicality-correction]
#
# Example:
#   run 1 16 "bash scripts/run_one_cmd.sh eval hypernym-bananas models/v6-google--gemma-2-2b-delta0.15-epoch2--hypernym-bananas-all--d2g--random--alpha1.0--full-completion--pref0.0--nllv1.0--nllg1.0--vallogodds"
#   run 1 16 "bash scripts/run_one_cmd.sh train hypernym-diapers"
#
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
cd scripts

MODE="$1"
TASK="$2"
shift 2

if [ "$MODE" = "eval" ]; then
    MODEL="$1"
    [ -z "$MODEL" ] && { echo "Usage: run_one_cmd.sh eval <task> <model_path>"; exit 1; }
    if [[ "$MODEL" != /* ]]; then
        MODEL="../$MODEL"
    fi
    exec python3 eval.py \
        --model "$MODEL" \
        --task "$TASK" \
        --split_type random \
        --validator-log-odds \
        --typicality-correction \
        --save-scores-csv \
        --viz
elif [ "$MODE" = "train" ]; then
    # Optional: pass --typicality-correction as extra arg
    python3 ranking_loss_ref.py \
        --model google/gemma-2-2b \
        --num_epochs 3 \
        --task "$TASK" \
        --train_g_or_d g \
        --split_type random \
        --nll_validator_weight 1 \
        --nll_generator_weight 1 \
        --preference_loss_weight 0 \
        --all \
        --delta 0.15 \
        --total_samples 5110 \
        --validator-log-odds \
        "$@"
else
    echo "Usage: run_one_cmd.sh eval <task> <model_path>"
    echo "       run_one_cmd.sh train <task> [--typicality-correction]"
    exit 1
fi
