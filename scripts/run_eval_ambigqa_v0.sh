#!/bin/bash

# Evaluate base model (google/gemma-2-2b) on a single AmbigQA v0 task.
#
# Usage: run run_eval_ambigqa_v0.sh <TASK>
# Example: run run_eval_ambigqa_v0.sh ambigqa-v0-cubs-win
# Example: run run_eval_ambigqa_v0.sh ambigqa-plausibleqa-combined-v0
#
# Task list: data/ambigqa/v0/task-list.txt

source /u/jdr/venvs/venv_lexcons/bin/activate

TASK=$1
MODEL="google/gemma-2-2b"

if [ -z "$TASK" ]; then
    echo "Usage: $0 <TASK>"
    echo ""
    echo "  TASK: an AmbigQA v0 task name, e.g.:"
    echo "    ambigqa-plausibleqa-combined-v0   (all questions combined)"
    echo "    ambigqa-v0-cubs-win               (single question)"
    echo ""
    echo "  See data/ambigqa/v0/task-list.txt for full list of per-question tasks."
    exit 1
fi

echo "Task:  $TASK"
echo "Model: $MODEL"
echo ""

cd "$(dirname "$0")"

python eval.py \
    --model "$MODEL" \
    --task "$TASK" \
    --split_type random \
    --disc-shots few \
    --gen-shots zero \
    --validator-log-odds \
    --save-scores-csv
