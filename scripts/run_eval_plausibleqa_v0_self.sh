#!/bin/bash

# Evaluate base model (google/gemma-2-2b) on PlausibleQA tasks using SELF-typicality.
# Produces scores files with "self-" prefix.
#
# Usage: run 1 0 --min 5 --cpu 4 --mem 32G scripts/run_eval_plausibleqa_v0_self.sh <TASK1> [TASK2] ...
# Example: run 1 0 --min 5 --cpu 4 --mem 32G scripts/run_eval_plausibleqa_v0_self.sh plausibleqa-nq_1109 plausibleqa-nq_1114

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
    echo "Self-typicality correction"
    echo "========================================"

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
