#!/bin/bash

# Evaluate base model (google/gemma-2-2b) on one or more AmbigQA v1 tasks.
#
# Usage: run 1 1 scripts/run_eval_ambigqa_v1.sh <TASK1> [TASK2] ...
# Example: run 1 1 scripts/run_eval_ambigqa_v1.sh ambigqa-won ambigqa-year
#
# All 17 test tasks:
#   ambigqa-american ambigqa-danube ambigqa-executed ambigqa-gives
#   ambigqa-harry ambigqa-involved ambigqa-jack ambigqa-plays
#   ambigqa-received ambigqa-sang ambigqa-soccer ambigqa-used
#   ambigqa-voice ambigqa-winter ambigqa-won ambigqa-world ambigqa-year

source /u/jdr/venvs/venv_lexcons/bin/activate

MODEL="google/gemma-2-2b"

if [ -z "$1" ]; then
    echo "Usage: $0 <TASK1> [TASK2] ..."
    echo ""
    echo "  All 17 test tasks:"
    echo "    ambigqa-american ambigqa-danube ambigqa-executed ambigqa-gives"
    echo "    ambigqa-harry ambigqa-involved ambigqa-jack ambigqa-plays"
    echo "    ambigqa-received ambigqa-sang ambigqa-soccer ambigqa-used"
    echo "    ambigqa-voice ambigqa-winter ambigqa-won ambigqa-world ambigqa-year"
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
