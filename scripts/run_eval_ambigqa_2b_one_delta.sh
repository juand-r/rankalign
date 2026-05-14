#!/bin/bash
# Eval one ambigqa-train-as-test RankAlign-baseline checkpoint trained at a
# given delta. Mirrors scripts/run_eval_ambigqa_2b_delta1.sh but parameterized
# over delta (and any TC-ref subset, default = full self/neg/basetyp/basetypneg
# matrix).
#
# Args:
#   $1 = delta value (matches the value used at training time, e.g. 0.3)
#
# Companion launcher: scripts/launch_ambigqa_deltasweep_evals.sh.

set -e

DELTA="${1:?usage: $0 <delta>  (e.g. 0.3)}"

source /u/jdr/venvs/venv_lexcons/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/datastor1/jdr/.cache/huggingface

cd "$(dirname "$0")"

BASE_MODEL=google/gemma-2-2b
TASK=ambigqa-train-as-test
EPOCH=2
MODELS_DIR=../models-quickiter
OUTPUTS_DIR=../outputs-quickiter

PREFIX="v6-google--gemma-2-2b-delta${DELTA}-epoch${EPOCH}--${TASK}-all--d2g--random--alpha1.0"
MODEL_DIR="${MODELS_DIR}/${PREFIX}--full-completion--force-same-x"

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: $MODEL_DIR does not exist"
    exit 1
fi

run_one_eval () {
    local tc_tag="$1"
    case "$tc_tag" in
        self)        TC_FLAGS="--self-typicality" ;;
        neg)         TC_FLAGS="--neg-typicality" ;;
        basetyp)     TC_FLAGS="--base-typicality --base-model-name $BASE_MODEL" ;;
        basetypneg)  TC_FLAGS="--base-typicality --neg-typicality --base-model-name $BASE_MODEL" ;;
        *) echo "Unknown TC tag: $tc_tag"; return 1 ;;
    esac
    echo "  -- Eval [$tc_tag]  ($TC_FLAGS) + --validator-log-odds  delta=$DELTA"
    python eval_by_claude.py \
        --model "$MODEL_DIR" \
        --task "$TASK" \
        --split_type random \
        --disc-shots few \
        --gen-shots zero \
        --outputs-dir "$OUTPUTS_DIR" \
        --validator-log-odds \
        $TC_FLAGS \
        --save-scores-csv
}

echo "============================================================"
echo "ambigqa delta=${DELTA} RankAlign baseline eval (matched matrix)"
echo "Model: $MODEL_DIR"
echo "============================================================"

for TC in self neg basetyp basetypneg; do
    run_one_eval "$TC"
done

echo "Done (delta=${DELTA})."
