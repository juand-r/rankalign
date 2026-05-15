#!/bin/bash
# Pref-loss-only knob ablation on membership-sans-rosch-v0 (gemma-2-2b).
#
# Three new training runs that close the gap identified in
# docs/membership_to_rosch_recipe_inventory.md:
#
#   #1  Plain RankAlign           (fsx=N, TC=N, vlo=N)   — missing reference point
#   #2  RankAlign + fsx + vlo     (fsx=Y, TC=N, vlo=Y)   — isolates +vlo on top of fsx
#   #3  RankAlign + fsx + vlo + TC-self  (fsx=Y, TC=self, vlo=Y) — kitchen-sink
#
# (Optional #4: RankAlign + TC-self, fsx=N, TC=self, vlo=N — uncomment below
# if we also want to ask "does TC alone help even without fsx?".)
#
# Same membership-sans-rosch-v0 task, same delta=0.15, 3 epochs, --all,
# total_samples per launcher default. Walltime budget matches the surrounding
# pref-only-loss runs in run_train_membership_quickiter.sh (4h).
#
# Usage:
#   bash scripts/run_train_membership_pref_knob_ablation.sh

set -e
cd "$(dirname "$0")/.."

MODEL=google/gemma-2-2b
TASK=membership-sans-rosch-v0
NUM_EPOCHS=3
DELTA=0.15
SAVE_STEPS=999
MODELS_DIR=./models-quickiter

mkdir -p "$MODELS_DIR" overnight
JOBIDS_FILE=overnight/membership_pref_ablation_train_jobids.txt
: > "$JOBIDS_FILE"

# Pref-loss-only base WITHOUT --force-same-x so we can toggle it per run.
PREF_BASE_NOFSX=" \
    --model $MODEL \
    --task $TASK \
    --train_g_or_d g \
    --split_type random \
    --num_epochs $NUM_EPOCHS \
    --delta $DELTA \
    --save_steps $SAVE_STEPS \
    --all \
    --nll_validator_weight 0 \
    --nll_generator_weight 0 \
    --preference_loss_weight 1 \
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

submit_one "[1/3] Plain RankAlign (no fsx, no TC, no vlo)"        4 "$PREF_BASE_NOFSX"
submit_one "[2/3] RankAlign+fsx + vlo (no TC)"                    4 "$PREF_BASE_NOFSX --force-same-x --validator-log-odds"
submit_one "[3/3] RankAlign+fsx + vlo + TC-self"                  4 "$PREF_BASE_NOFSX --force-same-x --validator-log-odds --self-typicality"

# Optional 4th run — uncomment to also ask "does TC alone help (no fsx)?"
# submit_one "[4/4] RankAlign + TC-self (no fsx, no vlo)"           4 "$PREF_BASE_NOFSX --self-typicality"

echo ""
echo "============================================================"
echo "Submitted training jobs."
echo "JIDs / labels recorded in $JOBIDS_FILE"
echo "============================================================"
cat "$JOBIDS_FILE"
