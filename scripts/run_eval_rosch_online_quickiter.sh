#!/bin/bash
# Evaluate the 9 quick-iter rosch-furniture-and-bird models + the bare base
# model. All evals are on the same combined furniture+bird test set.
#
# Eval-time TC convention (from docs/IMPORTANT-RESEARCH-PLAN.md, refined
# 2026-05-10 evening):
#   - Models trained with self-TC -> eval with --self-typicality (matched).
#   - Models trained with neg-TC  -> eval with --neg-typicality (matched).
#   - --base-typicality is used ONLY with offline-trained models, where it
#     matches the actual P_base reference precomputed at training time:
#       * offline self-TC (#2)  -> also report --base-typicality.
#       * offline neg-TC  (#7)  -> also report --base-typicality + --neg-typicality
#                                  (i.e. "basetypneg-": P_base under negated prompt).
#   - Models trained without TC at all (1, 4, 6, base) -> report under
#     both --self-typicality and --neg-typicality for cross-comparison.
#     We do NOT add --base-typicality for these (basetyp is reserved for
#     models whose offline training reference IS P_base).
#
# All eval calls pass --validator-log-odds (consistent with the train-time
# log-odds flag and the eval convention agreed for this round).
#
# A single eval call with --{self,neg,base}-typicality emits a scores_*.csv
# containing raw, tc, lenorm, and tc+lenorm columns. Aggregate afterwards
# with scripts/summarize_scores_file.py.
#
# Usage:
#   run 1 3 scripts/run_eval_rosch_online_quickiter.sh
#   # (single GPU; 10 models * 1-2 evals each, ~2 h)

set -e

source /u/jdr/venvs/venv_lexcons/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/datastor1/jdr/.cache/huggingface

cd "$(dirname "$0")"

BASE_MODEL=google/gemma-2-2b
TASK=rosch-furniture-and-bird
EPOCH=2
MODELS_DIR=../models-quickiter
OUTPUTS_DIR=../outputs-quickiter

mkdir -p "$OUTPUTS_DIR"

PREFIX="v6-google--gemma-2-2b-delta0.15-epoch${EPOCH}--${TASK}-all--d2g--random--alpha1.0"

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

# Per-variant eval TC types. Tags from {self, neg, basetyp, basetypneg}.
# basetyp[neg] is reserved for OFFLINE-trained models only.
declare -A EVAL_TC
EVAL_TC[base]="self neg"                         # self==basetyp, neg==basetypneg by definition
EVAL_TC[1_baseline]="self neg"                   # no TC at train
EVAL_TC[2_offSelfTC]="self basetyp"              # offline self-TC: matched is basetyp
EVAL_TC[3_onSelfTC]="self"                       # online self-TC: matched is self
EVAL_TC[4_onPairs]="self neg"                    # no TC at train
EVAL_TC[5_bothOnSelf]="self"                     # online self-TC
EVAL_TC[6_SFT]="self neg"                        # no TC at train
EVAL_TC[7_offNegTC]="neg basetypneg"             # offline neg-TC: matched is basetypneg
EVAL_TC[8_onNegTC]="neg"                         # online neg-TC: matched is neg
EVAL_TC[9_bothOnNeg]="neg"                       # online neg-TC

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
