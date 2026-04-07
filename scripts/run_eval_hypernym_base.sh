#!/bin/bash

# Evaluate base model (google/gemma-2-2b) on hypernym tasks.
# Produces non-self score files: scores_v6-google_gemma-2-2b_hypernym-*
#
# Usage: run 1 1 scripts/run_eval_hypernym_base.sh hypernym-helmets hypernym-kayaks hypernym-mirrors

source /u/jdr/venvs/venv_lexcons/bin/activate
export HF_HOME=/datastor1/jdr/.cache/huggingface

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
        --validator-log-odds \
        --typicality-correction \
        --save-scores-csv

    echo ""
done
