#!/bin/bash
# AmbigQA train-as-test, gemma-2-2b, RankAlign baseline ONLY.
# Sweep over delta ∈ {0.3, 0.5, 0.7} to fill in between the existing
# delta=0.15 (gen-ROC self = 71.82) and delta=1.0 (gen-ROC self = 61.07)
# data points.
#
# Recipe identical to scripts/run_train_ambigqa_2b_delta1.sh except --delta.
# See docs/delta_sweep_plan.md for the rationale and decision rules.
#
# Usage:
#   bash scripts/run_train_ambigqa_2b_deltasweep.sh

set -e

MODEL=google/gemma-2-2b
TASK=ambigqa-train-as-test
NUM_EPOCHS=3
SAVE_STEPS=999
MODELS_DIR=../models-quickiter

DELTAS=(0.3 0.5 0.7)

mkdir -p "$(dirname "$0")/$MODELS_DIR" overnight ~/logs

JOBIDS_FILE=overnight/ambigqa_deltasweep_jobids.txt
: > "$JOBIDS_FILE"

cd "$(dirname "$0")"

for D in "${DELTAS[@]}"; do
    PREF_BASE=" \
        --model $MODEL \
        --task $TASK \
        --train_g_or_d g \
        --split_type random \
        --num_epochs $NUM_EPOCHS \
        --delta $D \
        --save_steps $SAVE_STEPS \
        --all \
        --force-same-x \
        --nll_validator_weight 0 \
        --nll_generator_weight 0 \
        --preference_loss_weight 1 \
        --models-dir $MODELS_DIR \
        --no-upload-hf"

    echo "[delta=$D] submitting RankAlign baseline (ambigqa, 2b)"
    out=$(run 1 4 "python ranking_loss_ref_online.py $PREF_BASE" 2>&1)
    echo "$out"
    jid=$(echo "$out" | grep -oE 'Submitted batch job [0-9]+' | awk '{print $4}')
    if [[ -n "$jid" ]]; then
        echo "$jid  delta=$D" >> "../$JOBIDS_FILE"
    fi
    echo ""
done

echo "============================================================"
echo "Submitted ${#DELTAS[@]} training jobs."
echo "JIDs / deltas in $JOBIDS_FILE:"
cat "../$JOBIDS_FILE"
echo "============================================================"
