#!/bin/bash
# Fill in missing eval-ref combinations on the 3 May-2 membership-sans-rosch-v0
# pref-loss-only checkpoints, so the (vlo × TC × fsx × eval-ref) matrix is
# fully populated for the inventory in
# docs/membership_to_rosch_recipe_inventory.md.
#
# These are existing checkpoints under models/ — we are NOT retraining, just
# running additional evals under different typicality references. Output CSVs
# go into outputs/ to merge with the original May-2 cohort; the aggregation
# script (aggregate_old_membership_to_rosch.py) picks them up automatically.
#
# Args:
#   $1 = EVAL_TASK (e.g. rosch-bird, rosch-toy, ...)
#
# Usage examples:
#   bash scripts/run_eval_membership_may2_extra_refs.sh rosch-toy           # locally
#   run 1 1 "bash scripts/run_eval_membership_may2_extra_refs.sh rosch-toy" # via slurm
#
# Recap of which eval refs are still missing per checkpoint (as of 2026-05-14):
#
#   Plain RankAlign (full-completion_semi0.1)
#     have: self
#     need: neg, basetyp, basetypneg
#
#   RankAlign+fsx + vlo + TC-self (tc-self_full-completion_force-same-x_vallogodds_semi0.1)
#     have: self
#     need: neg, basetyp
#
#   RankAlign+fsx + vlo + TC-neg (tc-neg_full-completion_force-same-x_vallogodds_semi0.1)
#     have: neg
#     need: self, basetypneg

set -e

EVAL_TASK="${1:?usage: $0 <eval_task>  (e.g. rosch-toy)}"

source /u/jdr/venvs/venv_lexcons/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/datastor1/jdr/.cache/huggingface

cd "$(dirname "$0")"

BASE_MODEL=google/gemma-2-2b
TRAIN_TASK=membership-sans-rosch-v0
EPOCH=2
MODELS_DIR=../models
OUTPUTS_DIR=../outputs

mkdir -p "$OUTPUTS_DIR"

PREFIX="v6-google--gemma-2-2b-delta0.15-epoch${EPOCH}--${TRAIN_TASK}-all--d2g--random--alpha1.0"

declare -A MODELS
MODELS[A_PlainRA]="${MODELS_DIR}/${PREFIX}--full-completion--semi0.1"
MODELS[B_fsxVloTCself]="${MODELS_DIR}/${PREFIX}--tc-self--full-completion--force-same-x--vallogodds--semi0.1"
MODELS[C_fsxVloTCneg]="${MODELS_DIR}/${PREFIX}--tc-neg--full-completion--force-same-x--vallogodds--semi0.1"

# Eval refs we still need per model.
declare -A EVAL_TC
EVAL_TC[A_PlainRA]="neg basetyp basetypneg"
EVAL_TC[B_fsxVloTCself]="neg basetyp"
EVAL_TC[C_fsxVloTCneg]="self basetypneg"

VARIANTS=(A_PlainRA B_fsxVloTCself C_fsxVloTCneg)

run_one_eval () {
    local model_path="$1"
    local tc_tag="$2"
    case "$tc_tag" in
        self)        TC_FLAGS="--self-typicality" ;;
        neg)         TC_FLAGS="--neg-typicality" ;;
        basetyp)     TC_FLAGS="--base-typicality --base-model-name $BASE_MODEL" ;;
        basetypneg)  TC_FLAGS="--base-typicality --neg-typicality --base-model-name $BASE_MODEL" ;;
        *) echo "Unknown TC tag: $tc_tag"; return 1 ;;
    esac
    echo "  -- Eval [$tc_tag]  ($TC_FLAGS) + --validator-log-odds  on $EVAL_TASK"
    python eval_by_claude.py \
        --model "$model_path" \
        --task "$EVAL_TASK" \
        --split_type random \
        --disc-shots few \
        --gen-shots zero \
        --outputs-dir "$OUTPUTS_DIR" \
        --validator-log-odds \
        $TC_FLAGS \
        --save-scores-csv
}

START_TIME=$SECONDS

echo "============================================================"
echo "MEMBERSHIP (May-2) -> ROSCH extra eval refs"
echo "  train=${TRAIN_TASK}  eval=${EVAL_TASK}"
echo "============================================================"

for V in "${VARIANTS[@]}"; do
    MODEL_DIR="${MODELS[$V]}"
    TCS="${EVAL_TC[$V]}"
    echo ""
    echo "[$V]  $MODEL_DIR  on $EVAL_TASK"
    echo "    Missing TC types: $TCS"

    if [[ ! -d "$MODEL_DIR" ]]; then
        echo "  SKIP: model dir does not exist: $MODEL_DIR"
        continue
    fi

    for TC_TAG in $TCS; do
        run_one_eval "$MODEL_DIR" "$TC_TAG"
    done
done

ELAPSED=$((SECONDS - START_TIME))
echo ""
echo "============================================================"
echo "Done. eval=${EVAL_TASK}.  Elapsed: ${ELAPSED}s"
echo "Scores CSVs in: $OUTPUTS_DIR"
echo "============================================================"
