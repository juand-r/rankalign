#!/bin/bash
# Pref-loss-only knob ablation on membership-sans-rosch-v0 (gemma-2-2b).
#
# IMPORTANT — vlo and semi are both NO-OPS in g-mode + pref-only training.
#
# After code inspection (see docs/membership_to_rosch_recipe_inventory.md):
#   - `--semi-supervised 0.1`: collapses to plain preference loss when
#     pref=1, nllv=0, nllg=0. No effect on gradients.
#   - `--validator-log-odds`: only consulted in branches gated on
#     d-mode (preference-loss scoring) or nllv>0 (NLL validator loss).
#     In g-mode + pref-only, neither branch fires. No effect on gradients.
#
# So we DROP the previously-planned vlo runs (they would just produce
# duplicates of the no-vlo cells) and instead launch the runs that
# actually fill new informative cells in the (TC, fsx) sub-grid.
#
# Cells we already have (pref-only, gemma-2-2b epoch2):
#   (fsx=N, TC=none)  ✅ Plain RankAlign       (`full-completion_semi0.1`, May 2)
#   (fsx=Y, TC=none)  ✅ RankAlign+fsx         (May 13)
#   (fsx=Y, TC=self)  ✅ RankAlign+fsx+TC-self (May 13)
#   (fsx=Y, TC=neg)   ✅ RankAlign+fsx+TC-neg  (May 13)
#
# Cells still missing in the (TC, fsx) sub-grid:
#   #1  (fsx=N, TC=self)  RankAlign + TC-self  — pairs with Plain RankAlign
#       to ask "does TC help on its own without fsx?"
#       Pairs with fsx+TC-self (have) to ask "what does fsx add to TC?"
#
#   #2  (fsx=N, TC=neg)   RankAlign + TC-neg   — symmetric for neg side
#       Only worth running if we care about the neg-side comparison.
#
# (Optional #3: Plain RankAlign retrained on the May 13 commit. The
# existing Plain RankAlign is May 2; cohort drift between May 2 and
# May 13 is empirically ~5pt on should-be-identical runs, so a fresh
# retrain on the current commit would let us cleanly disentangle "fsx
# alone" from cohort drift.)
#
# Same membership-sans-rosch-v0 task, delta=0.15, 3 epochs, --all,
# 4h walltime budget (matches the surrounding pref-only-loss runs).
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

submit_one "[1/2] RankAlign + TC-self (no fsx, no vlo)"  4 "$PREF_BASE_NOFSX --self-typicality"
submit_one "[2/2] RankAlign + TC-neg  (no fsx, no vlo)"  4 "$PREF_BASE_NOFSX --neg-typicality"

# Optional 3rd run — uncomment to retrain Plain RankAlign on the current
# commit so cross-cohort drift can be subtracted out of "fsx alone" deltas.
# submit_one "[3/3] Plain RankAlign on May-13 commit"      4 "$PREF_BASE_NOFSX"

echo ""
echo "============================================================"
echo "Submitted training jobs."
echo "JIDs / labels recorded in $JOBIDS_FILE"
echo "============================================================"
cat "$JOBIDS_FILE"
