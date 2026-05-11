#!/bin/bash
# Cross-task eval: take the 9 quick-iter rosch-furniture-and-bird models +
# the bare base model, and evaluate them on a *different* rosch category.
# Designed to run as one short Slurm job per target task; the launcher
# scripts/launch_rosch_crosstask_evals.sh fans out across the 8 non-train
# rosch categories.
#
# Args:
#   $1 = EVAL_TASK (e.g. rosch-toy, rosch-vegetable, ...)
#
# This script does BOTH the matched evals (mirrors run_eval_rosch_online_quickiter.sh)
# AND the cross-comparable basetyp/basetypneg extras (mirrors the *_basetyp_extra.sh)
# in a single pass, so each Slurm job only loads each model the necessary
# number of times.
#
# Model paths are unchanged — the models WERE trained on rosch-furniture-and-bird,
# only --task for the eval call differs.
#
# Usage examples:
#   bash scripts/run_eval_rosch_crosstask_one.sh rosch-toy            # locally
#   run 1 1 "bash scripts/run_eval_rosch_crosstask_one.sh rosch-toy"  # via slurm

set -e

EVAL_TASK="${1:?usage: $0 <eval_task>  (e.g. rosch-toy)}"

source /u/jdr/venvs/venv_lexcons/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/datastor1/jdr/.cache/huggingface

cd "$(dirname "$0")"

BASE_MODEL=google/gemma-2-2b
TRAIN_TASK=rosch-furniture-and-bird
EPOCH=2
MODELS_DIR=../models-quickiter
OUTPUTS_DIR=../outputs-quickiter

mkdir -p "$OUTPUTS_DIR"

PREFIX="v6-google--gemma-2-2b-delta0.15-epoch${EPOCH}--${TRAIN_TASK}-all--d2g--random--alpha1.0"

declare -A MODELS
MODELS[base]="${BASE_MODEL}"
MODELS[1_baseline]="${MODELS_DIR}/${PREFIX}--full-completion--force-same-x"
MODELS[2_offSelfTC]="${MODELS_DIR}/${PREFIX}--tc-self--full-completion--force-same-x"
MODELS[3_onSelfTC]="${MODELS_DIR}/${PREFIX}--tc-self--full-completion--force-same-x--online-tc"
MODELS[4_onPairs]="${MODELS_DIR}/${PREFIX}--full-completion--force-same-x--online-pairs"
MODELS[5_bothOnSelf]="${MODELS_DIR}/${PREFIX}--tc-self--full-completion--force-same-x--online-pairs--online-tc"
MODELS[6_SFT]="${MODELS_DIR}/${PREFIX}--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x"
MODELS[7_offNegTC]="${MODELS_DIR}/${PREFIX}--tc-neg--full-completion--force-same-x"
MODELS[8_onNegTC]="${MODELS_DIR}/${PREFIX}--tc-neg--full-completion--force-same-x--online-tc"
MODELS[9_bothOnNeg]="${MODELS_DIR}/${PREFIX}--tc-neg--full-completion--force-same-x--online-pairs--online-tc"

# Combined matched + cross-comparable basetyp/basetypneg per variant.
# (For the base model, self == basetyp and neg == basetypneg by definition,
#  so we skip the basetyp duplicates.)
declare -A EVAL_TC
EVAL_TC[base]="self neg"
EVAL_TC[1_baseline]="self neg basetyp basetypneg"
EVAL_TC[2_offSelfTC]="self basetyp"
EVAL_TC[3_onSelfTC]="self basetyp"
EVAL_TC[4_onPairs]="self neg basetyp basetypneg"
EVAL_TC[5_bothOnSelf]="self basetyp"
EVAL_TC[6_SFT]="self neg basetyp basetypneg"
EVAL_TC[7_offNegTC]="neg basetypneg"
EVAL_TC[8_onNegTC]="neg basetypneg"
EVAL_TC[9_bothOnNeg]="neg basetypneg"

VARIANTS=(base 1_baseline 2_offSelfTC 3_onSelfTC 4_onPairs 5_bothOnSelf 6_SFT 7_offNegTC 8_onNegTC 9_bothOnNeg)

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
echo "CROSS-TASK EVAL: train=${TRAIN_TASK}  eval=${EVAL_TASK}"
echo "============================================================"

for V in "${VARIANTS[@]}"; do
    MODEL_DIR="${MODELS[$V]}"
    TCS="${EVAL_TC[$V]}"
    echo ""
    echo "[$V]  $MODEL_DIR  on $EVAL_TASK"
    echo "    TC types: $TCS"

    if [[ "$V" != "base" && ! -d "$MODEL_DIR" ]]; then
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
