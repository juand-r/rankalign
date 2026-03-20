#!/bin/bash

# Evaluate base model (google/gemma-2-2b) on IFEval tasks using GPT-2 typicality.
#
# Usage: run 1 0 --min 5 --cpu 4 --mem 32G /datastor1/jdr/gv-gap/rankalign/scripts/run_eval_ifeval.sh ifeval-prompt_1 ifeval-prompt_2 ...

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
        --disc-shots zero \
        --gen-shots zero \
        --validator-log-odds \
        --typicality-correction \
        --save-scores-csv

    echo ""
done
