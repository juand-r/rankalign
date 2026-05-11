#!/bin/bash
# Extra basetyp / basetypneg evals for the gemma-2-2b ambigqa-train-as-test
# quick-iter models. The matched-eval script restricted basetyp to
# OFFLINE-trained models. This follow-up adds basetyp/basetypneg to:
#   - the no-TC-at-train models (1_baseline, 4_onPairs, 6_SFT)
#   - the online-TC-at-train models (3_onSelfTC, 5_bothOnSelf, 8_onNegTC, 9_bothOnNeg)
# so we get a fully cross-comparable basetyp column across all 9 fine-tuned
# variants (P_base is fixed; --self-typicality drifts per fine-tune).
#
# All eval calls pass --validator-log-odds.
#
# Usage:
#   run 1 2 bash scripts/run_eval_ambigqa_train_as_test_quickiter_basetyp_extra.sh

set -e

source /u/jdr/venvs/venv_lexcons/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/datastor1/jdr/.cache/huggingface

cd "$(dirname "$0")"

BASE_MODEL=google/gemma-2-2b
TASK=ambigqa-train-as-test
EPOCH=2
MODELS_DIR=../models-quickiter
OUTPUTS_DIR=../outputs-quickiter

mkdir -p "$OUTPUTS_DIR"

PREFIX="v6-google--gemma-2-2b-delta0.15-epoch${EPOCH}--${TASK}-all--d2g--random--alpha1.0"

declare -A MODELS
MODELS[1_baseline]="${MODELS_DIR}/${PREFIX}--full-completion--force-same-x"
MODELS[3_onSelfTC]="${MODELS_DIR}/${PREFIX}--tc-self--full-completion--force-same-x--online-tc"
MODELS[4_onPairs]="${MODELS_DIR}/${PREFIX}--full-completion--force-same-x--online-pairs"
MODELS[5_bothOnSelf]="${MODELS_DIR}/${PREFIX}--tc-self--full-completion--force-same-x--online-pairs--online-tc"
MODELS[6_SFT]="${MODELS_DIR}/${PREFIX}--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x"
MODELS[8_onNegTC]="${MODELS_DIR}/${PREFIX}--tc-neg--full-completion--force-same-x--online-tc"
MODELS[9_bothOnNeg]="${MODELS_DIR}/${PREFIX}--tc-neg--full-completion--force-same-x--online-pairs--online-tc"

declare -A EVAL_TC
EVAL_TC[1_baseline]="basetyp basetypneg"
EVAL_TC[3_onSelfTC]="basetyp"
EVAL_TC[4_onPairs]="basetyp basetypneg"
EVAL_TC[5_bothOnSelf]="basetyp"
EVAL_TC[6_SFT]="basetyp basetypneg"
EVAL_TC[8_onNegTC]="basetypneg"
EVAL_TC[9_bothOnNeg]="basetypneg"

VARIANTS=(1_baseline 3_onSelfTC 4_onPairs 5_bothOnSelf 6_SFT 8_onNegTC 9_bothOnNeg)

run_one_eval () {
    local model_path="$1"
    local tc_tag="$2"
    case "$tc_tag" in
        basetyp)     TC_FLAGS="--base-typicality --base-model-name $BASE_MODEL" ;;
        basetypneg)  TC_FLAGS="--base-typicality --neg-typicality --base-model-name $BASE_MODEL" ;;
        *) echo "Unknown TC tag: $tc_tag (this script handles basetyp/basetypneg only)"; return 1 ;;
    esac
    echo "  -- Eval [$tc_tag]  ($TC_FLAGS) + --validator-log-odds"
    python eval_by_claude.py \
        --model "$model_path" \
        --task "$TASK" \
        --split_type random \
        --disc-shots few \
        --gen-shots zero \
        --outputs-dir "$OUTPUTS_DIR" \
        --validator-log-odds \
        $TC_FLAGS \
        --save-scores-csv
}

START_TIME=$SECONDS

for V in "${VARIANTS[@]}"; do
    MODEL_DIR="${MODELS[$V]}"
    TCS="${EVAL_TC[$V]}"
    echo ""
    echo "============================================================"
    echo "[$V]  $MODEL_DIR"
    echo "    TC types: $TCS"
    echo "============================================================"

    if [[ ! -d "$MODEL_DIR" ]]; then
        echo "  SKIP: model dir does not exist"
        continue
    fi

    for TC_TAG in $TCS; do
        run_one_eval "$MODEL_DIR" "$TC_TAG"
    done
done

ELAPSED=$((SECONDS - START_TIME))
echo ""
echo "============================================================"
echo "Done. Elapsed: ${ELAPSED}s"
echo "Scores CSVs in: $OUTPUTS_DIR"
echo "============================================================"
