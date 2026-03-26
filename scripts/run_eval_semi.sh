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
#
# Example:
#   run 1 2 scripts/run_eval_semi.sh ../models/v6-...-semi0.1 --self-typcorr --log-odds -- plausibleqa-nq_1109 plausibleqa-nq_1114

source /u/jdr/venvs/venv_lexcons/bin/activate

MODEL="$1"
shift

EVAL_FLAGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --self-typcorr) EVAL_FLAGS="$EVAL_FLAGS --self-typicality"; shift ;;
        --typcorr) EVAL_FLAGS="$EVAL_FLAGS --typicality-correction"; shift ;;
        --lenorm) EVAL_FLAGS="$EVAL_FLAGS --length-normalize"; shift ;;
        --log-odds) EVAL_FLAGS="$EVAL_FLAGS --validator-log-odds"; shift ;;
        --) shift; break ;;
        *) break ;;
    esac
done

if [ -z "$MODEL" ] || [ -z "$1" ]; then
    echo "Usage: $0 <MODEL_DIR> [options] -- <TASK1> [TASK2] ..."
    echo ""
    echo "  Options: --self-typcorr --typcorr --lenorm --log-odds"
    exit 1
fi

cd "$(dirname "$0")"

echo "Model: $MODEL"
echo "Tasks: $#"
echo "Flags: $EVAL_FLAGS"
echo "========================================"

DONE=0
TOTAL=$#
START_TIME=$SECONDS

for TASK in "$@"; do
    DONE=$((DONE + 1))
    ELAPSED=$((SECONDS - START_TIME))
    if [ $DONE -gt 1 ] && [ $ELAPSED -gt 0 ]; then
        PER_TASK=$((ELAPSED / (DONE - 1)))
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
        --disc-shots few \
        --gen-shots zero \
        $EVAL_FLAGS \
        --save-scores-csv

    echo ""
done

TOTAL_TIME=$((SECONDS - START_TIME))
echo "========================================"
echo "Finished all $TOTAL tasks in ${TOTAL_TIME}s"
