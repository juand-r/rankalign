#!/bin/bash

# Evaluate a trained model on specified PlausibleQA test tasks using
# eval_by_claude.py with SELF-typicality correction.
#
# Usage: run 1 <hours> scripts/run_eval_plausibleqa_v0_model_self.sh <MODEL_DIR> <TASK1> [TASK2] ...
#
# Example:
#   run 1 2 scripts/run_eval_plausibleqa_v0_model_self.sh ../models/v6-google--gemma-2-2b-delta0.15-epoch2--plausibleqa-all--d2g--random--alpha1.0--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds plausibleqa-nq_1109 plausibleqa-nq_1114

source /u/jdr/venvs/venv_lexcons/bin/activate

MODEL="$1"
shift

if [ -z "$MODEL" ] || [ -z "$1" ]; then
    echo "Usage: $0 <MODEL_DIR> <TASK1> [TASK2] ..."
    exit 1
fi

cd "$(dirname "$0")"

echo "Model: $MODEL"
echo "Tasks: $#"
echo "Self-typicality correction"
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
        --validator-log-odds \
        --self-typicality \
        --save-scores-csv

    echo ""
done

TOTAL_TIME=$((SECONDS - START_TIME))
echo "========================================"
echo "Finished all $TOTAL tasks in ${TOTAL_TIME}s"
