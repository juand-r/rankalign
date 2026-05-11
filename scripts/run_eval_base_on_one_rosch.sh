#!/bin/bash
# Eval base gemma-2-2b on a single rosch task, with self+neg eval refs only
# (matched to the membership->rosch report's base row).
#
# Args: $1 = EVAL_TASK (e.g. rosch-bird, rosch-furniture)

set -e

EVAL_TASK="${1:?usage: $0 <eval_task>}"

source /u/jdr/venvs/venv_lexcons/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/datastor1/jdr/.cache/huggingface

cd "$(dirname "$0")"

BASE_MODEL=google/gemma-2-2b
OUTPUTS_DIR=../outputs-quickiter

for TC_TAG in self neg; do
    if [[ "$TC_TAG" == "self" ]]; then
        TC_FLAGS="--self-typicality"
    else
        TC_FLAGS="--neg-typicality"
    fi
    echo "[base][$TC_TAG] on $EVAL_TASK"
    python eval_by_claude.py \
        --model "$BASE_MODEL" \
        --task "$EVAL_TASK" \
        --split_type random \
        --disc-shots few \
        --gen-shots zero \
        --outputs-dir "$OUTPUTS_DIR" \
        --validator-log-odds \
        $TC_FLAGS \
        --save-scores-csv
done

echo "Done. eval=$EVAL_TASK"
