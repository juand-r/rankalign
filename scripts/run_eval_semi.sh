#!/bin/bash

# Evaluate a trained model on test tasks using eval_by_claude.py.
# Adapted from run_eval_plausibleqa_v0_model_self.sh with configurable flags.
#
# Usage: run 1 <hours> scripts/run_eval_semi.sh <MODEL_DIR> [options] -- <TASK1> [TASK2] ...
#
# Options (before --):
#   --self-typcorr         Enable self-typicality correction
#   --typcorr              Enable GPT-2 typicality correction
#   --lenorm               Enable length normalization
#   --log-odds             Enable validator log-odds
#   --disc-shots-zero      Use zero-shot discriminator (default: few-shot)
#   --include-eos          Include EOS in completion scoring
#
# Example:
#   run 1 2 scripts/run_eval_semi.sh ../models/v6-...-semi0.1 --self-typcorr --log-odds -- plausibleqa-nq_1109 plausibleqa-nq_1114

source /u/jdr/venvs/venv_lexcons/bin/activate

MODEL="$1"
shift

EVAL_FLAGS=""
DISC_SHOTS="few"
OUTPUTS_DIR="../outputs"
while [[ $# -gt 0 ]]; do
    case $1 in
        --self-typcorr) EVAL_FLAGS="$EVAL_FLAGS --self-typicality"; shift ;;
        --neg-typcorr) EVAL_FLAGS="$EVAL_FLAGS --neg-typicality"; shift ;;
        --base-typcorr) EVAL_FLAGS="$EVAL_FLAGS --base-typicality"; shift ;;
        --base-model) EVAL_FLAGS="$EVAL_FLAGS --base-model-name $2"; shift 2 ;;
        --typcorr) EVAL_FLAGS="$EVAL_FLAGS --typicality-correction"; shift ;;
        --lenorm) EVAL_FLAGS="$EVAL_FLAGS --length-normalize"; shift ;;
        --log-odds) EVAL_FLAGS="$EVAL_FLAGS --validator-log-odds"; shift ;;
        --include-eos) EVAL_FLAGS="$EVAL_FLAGS --include-eos"; shift ;;
        --disc-shots-zero) DISC_SHOTS="zero"; shift ;;
        --outputs-dir) OUTPUTS_DIR="$2"; shift 2 ;;
        --) shift; break ;;
        *) break ;;
    esac
done

if [ -z "$MODEL" ] || [ -z "$1" ]; then
    echo "Usage: $0 <MODEL_DIR> [options] -- <TASK1> [TASK2] ..."
    echo ""
    echo "  Options: --self-typcorr --neg-typcorr --base-typcorr --base-model <name> --typcorr --lenorm --log-odds --include-eos --outputs-dir <dir>"
    exit 1
fi

cd "$(dirname "$0")"

# Build the score-file prefix so we can skip already-completed tasks.
# Mirrors the filename logic in eval_by_claude.py (two orthogonal axes).
SELF_PFX=""
if [[ "$EVAL_FLAGS" == *"--base-typicality"* && "$EVAL_FLAGS" == *"--neg-typicality"* ]]; then
    SELF_PFX="basetypneg-"
elif [[ "$EVAL_FLAGS" == *"--base-typicality"* ]]; then
    SELF_PFX="basetyp-"
elif [[ "$EVAL_FLAGS" == *"--neg-typicality"* ]]; then
    SELF_PFX="neg-"
elif [[ "$EVAL_FLAGS" == *"--self-typicality"* ]]; then
    SELF_PFX="self-"
fi

MODEL_SHORT=$(basename "$MODEL" | sed 's/--/_/g')

METRIC_SUF="_log-probs"
[[ "$EVAL_FLAGS" == *"--validator-log-odds"* ]] && METRIC_SUF="_log-odds"

TC_SUF=""
[[ "$EVAL_FLAGS" == *"--typicality-correction"* || "$EVAL_FLAGS" == *"--self-typicality"* || "$EVAL_FLAGS" == *"--neg-typicality"* || "$EVAL_FLAGS" == *"--base-typicality"* ]] && TC_SUF="_tc"

LENORM_SUF=""
[[ "$EVAL_FLAGS" == *"--length-normalize"* ]] && LENORM_SUF="_evallenorm"

EOS_SUF=""
[[ "$EVAL_FLAGS" == *"--include-eos"* ]] && EOS_SUF="_eos"

echo "Model: $MODEL"
echo "Tasks: $#"
echo "Flags: $EVAL_FLAGS"
echo "========================================"

DONE=0
SKIPPED=0
TOTAL=$#
START_TIME=$SECONDS

for TASK in "$@"; do
    DONE=$((DONE + 1))

    # Hypernym tasks include a _v2 suffix by default
    V2_SUF=""
    [[ "$TASK" == hypernym-* ]] && V2_SUF="_v2"

    PATTERN="${OUTPUTS_DIR}/scores_${SELF_PFX}${MODEL_SHORT}_${TASK}_test${V2_SUF}${METRIC_SUF}${TC_SUF}${LENORM_SUF}${EOS_SUF}_*.csv"
    if ls $PATTERN 1>/dev/null 2>&1; then
        echo "[$DONE/$TOTAL] $TASK  -- SKIP (score file exists)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    ELAPSED=$((SECONDS - START_TIME))
    RAN=$((DONE - SKIPPED))
    if [ $RAN -gt 1 ] && [ $ELAPSED -gt 0 ]; then
        PER_TASK=$((ELAPSED / (RAN - 1)))
        REMAINING=$(( PER_TASK * (TOTAL - DONE + 1) ))
        echo "[$DONE/$TOTAL] $TASK  (elapsed ${ELAPSED}s, ~${REMAINING}s remaining)"
    else
        echo "[$DONE/$TOTAL] $TASK"
    fi

    HF_HOME=/datastor1/jdr/.cache/huggingface \
    python eval_by_claude.py \
        --model "$MODEL" \
        --task "$TASK" \
        --split_type random \
        --disc-shots $DISC_SHOTS \
        --gen-shots zero \
        --outputs-dir "$OUTPUTS_DIR" \
        $EVAL_FLAGS \
        --save-scores-csv

    echo ""
done

TOTAL_TIME=$((SECONDS - START_TIME))
echo "========================================"
echo "Finished $TOTAL tasks in ${TOTAL_TIME}s (skipped $SKIPPED already-done)"
