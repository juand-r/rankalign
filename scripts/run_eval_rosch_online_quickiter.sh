#!/bin/bash
# Evaluate the 9 quick-iter rosch-furniture-and-bird models + the bare base
# model. Each model is evaluated with multiple TC reference choices.
#
# Eval-time TC convention (from docs/IMPORTANT-RESEARCH-PLAN.md):
#   - Models trained with self-TC -> matched eval is --self-typicality.
#     Also report --base-typicality (frozen P_base under null context),
#     which is also the "matched" eval for OFFLINE self-TC.
#   - Models trained with neg-TC  -> matched eval is --neg-typicality.
#     Also report --base-typicality + --neg-typicality (i.e. basetypneg-,
#     P_base under negated prompt), which is the "matched" eval for
#     OFFLINE neg-TC.
#   - Models trained without TC (1, 4, 6) and the bare base model: eval
#     with all four reference choices for cross-comparison. The base
#     model is special-cased (self == base, neg == basetypneg, by
#     definition).
#
# A single eval call with --{self,neg,base}-typicality emits a scores_*.csv
# containing raw, tc, lenorm, and tc+lenorm columns.
#
# Output: scores_*.csv in ../outputs-quickiter/. Use scripts/summarize_scores_file.py
# to aggregate metrics afterwards.
#
# Usage:
#   run 1 3 scripts/run_eval_rosch_online_quickiter.sh
#   # (single GPU; ~2.5 h for 9 models * ~2 evals each)

set -e

source /u/jdr/venvs/venv_lexcons/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/datastor1/jdr/.cache/huggingface

cd "$(dirname "$0")"

BASE_MODEL=google/gemma-2-2b
TASK=rosch-furniture-and-bird
EPOCH=2  # Last saved epoch for num_epochs=3
MODELS_DIR=../models-quickiter
OUTPUTS_DIR=../outputs-quickiter

mkdir -p "$OUTPUTS_DIR"

PREFIX="v6-google--gemma-2-2b-delta0.15-epoch${EPOCH}--${TASK}-all--d2g--random--alpha1.0"

# Map: variant_label -> model_dir
declare -A MODELS
MODELS[base]="${BASE_MODEL}"
MODELS[1_baseline]="${MODELS_DIR}/${PREFIX}--full-completion--force-same-x"
MODELS[2_offSelfTC]="${MODELS_DIR}/${PREFIX}--tc-self--full-completion--force-same-x"
MODELS[3_onSelfTC]="${MODELS_DIR}/${PREFIX}--tc-self--full-completion--force-same-x--online-tc"
MODELS[4_onPairs]="${MODELS_DIR}/${PREFIX}--full-completion--force-same-x--online-pairs"
MODELS[5_bothOnSelf]="${MODELS_DIR}/${PREFIX}--tc-self--full-completion--force-same-x--online-pairs--online-tc"
MODELS[6_SFT]="${MODELS_DIR}/${PREFIX}--full-completion--pref0--nllv1--nllg1--force-same-x"
MODELS[7_offNegTC]="${MODELS_DIR}/${PREFIX}--tc-neg--full-completion--force-same-x"
MODELS[8_onNegTC]="${MODELS_DIR}/${PREFIX}--tc-neg--full-completion--force-same-x--online-tc"
MODELS[9_bothOnNeg]="${MODELS_DIR}/${PREFIX}--tc-neg--full-completion--force-same-x--online-pairs--online-tc"

# Per-variant eval TC types. Format: space-separated tags from
# {self, neg, basetyp, basetypneg}.
declare -A EVAL_TC
EVAL_TC[base]="self neg"                    # self == basetyp, neg == basetypneg for the bare base
EVAL_TC[1_baseline]="self neg basetyp basetypneg"   # no TC at train -> all 4
EVAL_TC[2_offSelfTC]="self basetyp"          # offline self-TC: matched is basetyp; self for cross
EVAL_TC[3_onSelfTC]="self basetyp"           # online self-TC: matched is self; basetyp for cross
EVAL_TC[4_onPairs]="self neg basetyp basetypneg"    # no TC at train
EVAL_TC[5_bothOnSelf]="self basetyp"
EVAL_TC[6_SFT]="self neg basetyp basetypneg"        # no TC at train
EVAL_TC[7_offNegTC]="neg basetypneg"         # offline neg-TC: matched is basetypneg; neg for cross
EVAL_TC[8_onNegTC]="neg basetypneg"          # online neg-TC: matched is neg; basetypneg for cross
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
    echo "  -- Eval [$tc_tag]  ($TC_FLAGS)"
    python eval_by_claude.py \
        --model "$model_path" \
        --task "$TASK" \
        --split_type random \
        --disc-shots few \
        --gen-shots zero \
        --outputs-dir "$OUTPUTS_DIR" \
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
echo "Done. Elapsed: ${ELAPSED}s"
echo "Scores CSVs in: $OUTPUTS_DIR"
echo "============================================================"
