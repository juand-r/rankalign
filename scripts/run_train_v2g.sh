#!/bin/bash
# Train RankAlign V2G baseline (no semi/labelonly, no force-same-x, no typcorr).
# Usage: run <gpus> <hours> scripts/run_train_v2g.sh <MODEL> <TASK> [--disc-shots zero]

set -e
source /u/jdr/venvs/venv_lexcons/bin/activate

MODEL=$1
TASK=$2
shift 2

DISC_SHOTS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --disc-shots) DISC_SHOTS="--disc-shots $2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

LORA_FLAG=""
if [[ "$MODEL" != *"-2b"* && "$MODEL" != *"-2b-"* ]]; then
    LORA_FLAG="--lora"
fi

cd "$(dirname "$0")"

echo "========================================"
echo "V2G Baseline Training"
echo "Task:  $TASK"
echo "Model: $MODEL"
echo "LoRA:  $LORA_FLAG"
echo "========================================"

python ranking_loss_ref.py \
    --model $MODEL \
    --task $TASK \
    --train_g_or_d g \
    --delta 0.15 \
    --all \
    $DISC_SHOTS \
    $LORA_FLAG
