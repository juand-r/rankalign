#!/bin/bash
# Quick-iter training: 9 runs on membership-sans-rosch-v0 (~2k items across
# 165 categories — concept-coverage stress test for the OOD generalization
# question that rosch-furniture-and-bird couldn't answer (only 2 categories).
#
# All runs use:
#   - gemma-2-2b (full fine-tuning, no LoRA)
#   - g-mode (train generator scores to match validator ranking)
#   - --force-same-x (pairs only within same generator prompt; here that means
#     same membership category — e.g. "an example of a bird is X" pairs only
#     with other "an example of a bird is Y" rows)
#   - 3 epochs, delta=0.15, total_samples=5110 (script default)
#   - --no-upload-hf
#   - --save_steps 999 (save only last epoch + epoch 0)
#
# Variants identical to scripts/run_train_rosch_online_quickiter.sh; only the
# task differs. Submitted jobs print JIDs that the eval launcher consumes via
# overnight/membership_train_jobids.txt.
#
# Usage:
#   bash scripts/run_train_membership_quickiter.sh

set -e

cd "$(dirname "$0")/.."

MODEL=google/gemma-2-2b
TASK=membership-sans-rosch-v0
NUM_EPOCHS=3
DELTA=0.15
SAVE_STEPS=999
MODELS_DIR=./models-quickiter

mkdir -p "$MODELS_DIR" overnight
JOBIDS_FILE=overnight/membership_train_jobids.txt
: > "$JOBIDS_FILE"

# Membership-v0 has more categories (165) than rosch-fb (2), so the in-loop
# pair sampling has more groups but each group is smaller. Observed per-epoch
# wall-clock from the first attempt (jobs 37965-37973) was ~50-60 min/epoch
# for offline/SFT/pref-only and ~90-100 min/epoch for online-TC variants.
# These budgets give enough margin for all three epochs PLUS the final save:
#   - offline / SFT / pref-only / online-pairs-only -> 4 h
#   - online-TC / online-both                       -> 6 h

PREF_BASE=" \
    --model $MODEL \
    --task $TASK \
    --train_g_or_d g \
    --split_type random \
    --num_epochs $NUM_EPOCHS \
    --delta $DELTA \
    --save_steps $SAVE_STEPS \
    --all \
    --force-same-x \
    --nll_validator_weight 0 \
    --nll_generator_weight 0 \
    --preference_loss_weight 1 \
    --models-dir $MODELS_DIR \
    --no-upload-hf"

SFT_BASE=" \
    --model $MODEL \
    --task $TASK \
    --train_g_or_d g \
    --split_type random \
    --num_epochs $NUM_EPOCHS \
    --delta $DELTA \
    --save_steps $SAVE_STEPS \
    --all \
    --force-same-x \
    --nll_validator_weight 1 \
    --nll_generator_weight 1 \
    --preference_loss_weight 0 \
    --models-dir $MODELS_DIR \
    --no-upload-hf"

submit_one () {
    local label="$1"; shift
    local hours="$1"; shift
    local args="$*"
    echo ""
    echo ">>> $label"
    out=$(run 1 "$hours" "python scripts/ranking_loss_ref_online.py $args" 2>&1)
    echo "$out"
    jid=$(echo "$out" | grep -oE 'Submitted batch job [0-9]+' | awk '{print $4}')
    if [[ -n "$jid" ]]; then
        echo "$jid  $label" >> "$JOBIDS_FILE"
    fi
}

submit_one "[1/9] RankAlign baseline"                                   4 "$PREF_BASE"
submit_one "[2/9] RankAlign + offline self-TC"                          4 "$PREF_BASE --self-typicality"
submit_one "[3/9] RankAlign + ONLINE self-TC"                           6 "$PREF_BASE --self-typicality --online-typicality"
submit_one "[4/9] RankAlign + ONLINE pair selection"                    4 "$PREF_BASE --online-pair-selection"
submit_one "[5/9] RankAlign + ONLINE self-TC + ONLINE pair selection"   6 "$PREF_BASE --self-typicality --online-typicality --online-pair-selection"
submit_one "[6/9] SFT (NLL all)"                                        4 "$SFT_BASE"
submit_one "[7/9] RankAlign + offline neg-TC"                           4 "$PREF_BASE --neg-typicality"
submit_one "[8/9] RankAlign + ONLINE neg-TC"                            6 "$PREF_BASE --neg-typicality --online-typicality"
submit_one "[9/9] RankAlign + ONLINE neg-TC + ONLINE pair selection"    6 "$PREF_BASE --neg-typicality --online-typicality --online-pair-selection"

echo ""
echo "============================================================"
echo "Submitted 9 training jobs."
echo "JIDs / labels recorded in $JOBIDS_FILE"
echo "============================================================"
cat "$JOBIDS_FILE"
