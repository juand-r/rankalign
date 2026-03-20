#!/bin/bash

# Evaluate base model (google/gemma-2-2b) on one or more PlausibleQA tasks.
#
# Usage: run 1 1 scripts/run_eval_plausibleqa_v0.sh <TASK1> [TASK2] ...
# Example: run 1 1 scripts/run_eval_plausibleqa_v0.sh plausibleqa-nq_1109 plausibleqa-nq_1114

source /u/jdr/venvs/venv_lexcons/bin/activate

MODEL="google/gemma-2-2b"

if [ -z "$1" ]; then
    echo "Usage: $0 <TASK1> [TASK2] ..."
    exit 1
fi

cd "$(dirname "$0")"

for TASK in "$@"; do
    echo "========================================"
    echo "Task:  $TASK"
    echo "Model: $MODEL"
    echo "========================================"

    python eval.py \
        --model "$MODEL" \
        --task "$TASK" \
        --split_type random \
        --disc-shots few \
        --gen-shots zero \
        --validator-log-odds \
        --typicality-correction \
        --save-scores-csv

    echo ""
done
