#!/bin/bash
# Eval the 9 membership-sans-rosch-v0 quick-iter models + base on a single
# rosch-* category. Designed to run as one short Slurm job per target rosch
# task; the launcher scripts/launch_membership_to_rosch_evals.sh fans out
# across all 10 rosch categories.
#
# Args:
#   $1 = EVAL_TASK (e.g. rosch-bird, rosch-toy, ...)
#
# Combined matched + cross-comparable basetyp/basetypneg per variant, mirrors
# scripts/run_eval_rosch_crosstask_one.sh.
#
# Usage examples:
#   bash scripts/run_eval_one_rosch_for_membership.sh rosch-toy           # locally
#   run 1 1 "bash scripts/run_eval_one_rosch_for_membership.sh rosch-toy" # via slurm

set -e

EVAL_TASK="${1:?usage: $0 <eval_task>  (e.g. rosch-toy)}"

source /u/jdr/venvs/venv_lexcons/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/datastor1/jdr/.cache/huggingface

cd "$(dirname "$0")"

BASE_MODEL=google/gemma-2-2b
TRAIN_TASK=membership-sans-rosch-v0
EPOCH=2
MODELS_DIR=../models-quickiter
OUTPUTS_DIR=../outputs-quickiter

mkdir -p "$OUTPUTS_DIR"

PREFIX="v6-google--gemma-2-2b-delta0.15-epoch${EPOCH}--${TRAIN_TASK}-all--d2g--random--alpha1.0"

# NOTE: base model is NOT evaluated here. Base evals on rosch-{carpenters-tool,
# clothing, fruit, sport, toy, vegetable, vehicle, weapon} were already done
# in the rosch-furniture-and-bird cross-task launch (dated 20260511). Base
# evals on rosch-{bird, furniture} are submitted as separate jobs to avoid
# duplicate CSVs that would crash the build scripts.
declare -A MODELS
MODELS[1_baseline]="${MODELS_DIR}/${PREFIX}--full-completion--force-same-x"
MODELS[2_offSelfTC]="${MODELS_DIR}/${PREFIX}--tc-self--full-completion--force-same-x"
MODELS[3_onSelfTC]="${MODELS_DIR}/${PREFIX}--tc-self--full-completion--force-same-x--online-tc"
MODELS[4_onPairs]="${MODELS_DIR}/${PREFIX}--full-completion--force-same-x--online-pairs"
MODELS[5_bothOnSelf]="${MODELS_DIR}/${PREFIX}--tc-self--full-completion--force-same-x--online-pairs--online-tc"
MODELS[6_SFT]="${MODELS_DIR}/${PREFIX}--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x"
MODELS[7_offNegTC]="${MODELS_DIR}/${PREFIX}--tc-neg--full-completion--force-same-x"
MODELS[8_onNegTC]="${MODELS_DIR}/${PREFIX}--tc-neg--full-completion--force-same-x--online-tc"
MODELS[9_bothOnNeg]="${MODELS_DIR}/${PREFIX}--tc-neg--full-completion--force-same-x--online-pairs--online-tc"

declare -A EVAL_TC
EVAL_TC[1_baseline]="self neg basetyp basetypneg"
EVAL_TC[2_offSelfTC]="self basetyp"
EVAL_TC[3_onSelfTC]="self basetyp"
EVAL_TC[4_onPairs]="self neg basetyp basetypneg"
EVAL_TC[5_bothOnSelf]="self basetyp"
EVAL_TC[6_SFT]="self neg basetyp basetypneg"
EVAL_TC[7_offNegTC]="neg basetypneg"
EVAL_TC[8_onNegTC]="neg basetypneg"
EVAL_TC[9_bothOnNeg]="neg basetypneg"

VARIANTS=(1_baseline 2_offSelfTC 3_onSelfTC 4_onPairs 5_bothOnSelf 6_SFT 7_offNegTC 8_onNegTC 9_bothOnNeg)

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
echo "MEMBERSHIP -> ROSCH eval: train=${TRAIN_TASK}  eval=${EVAL_TASK}"
echo "============================================================"

for V in "${VARIANTS[@]}"; do
    MODEL_DIR="${MODELS[$V]}"
    TCS="${EVAL_TC[$V]}"
    echo ""
    echo "[$V]  $MODEL_DIR  on $EVAL_TASK"
    echo "    TC types: $TCS"

    if [[ ! -d "$MODEL_DIR" ]]; then
        echo "  SKIP: model dir does not exist (training not done?)"
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
