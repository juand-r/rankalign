#!/bin/bash

# Evaluate base model (google/gemma-2-2b) on hypernym tasks using SELF-typicality.
# Produces scores files with "self-" prefix.
#
# Usage: run 1 0 --min 5 --cpu 4 --mem 32G scripts/run_eval_hypernym_self.sh hypernym-bananas hypernym-cars ...

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
    echo "Self-typicality correction"
    echo "========================================"

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
