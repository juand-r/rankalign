#!/bin/bash
# Sequential training: 3 variants (no-TC, self-TC, neg-TC) for one model on one GPU.
# Usage: bash train_seq_v21cu.sh <MODEL_HF_NAME> <GPU_INDEX>

set -uo pipefail
MODEL="$1"
GPU="$2"

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /workspace/.venv/bin/activate

cd /workspace/rankalign/scripts

TASK=humaneval-v2.1correct-upper
COMMON_FLAGS="--disc-shots zero --models-dir /workspace/models"

echo "=== GPU=$GPU  MODEL=$MODEL  start=$(date) ==="

# Variant 2: RankAlign baseline (pref-only, no TC, no force-same-x, no log-odds)
echo ""
echo "=== variant 2 (RankAlign, no TC) — $(date) ==="
bash run_train_semi.sh "$MODEL" "$TASK" pref-only semi 0.1 $COMMON_FLAGS --no-force-same-x

# Variant 6: RankAlign + self-TC (pref-only, --self-typcorr, --no-force-same-x)
echo ""
echo "=== variant 6 (RankAlign + self-TC) — $(date) ==="
bash run_train_semi.sh "$MODEL" "$TASK" pref-only semi 0.1 $COMMON_FLAGS --self-typcorr --no-force-same-x

# Variant 6': RankAlign + neg-TC (pref-only, --neg-typcorr, --no-force-same-x)
echo ""
echo "=== variant 6' (RankAlign + neg-TC) — $(date) ==="
bash run_train_semi.sh "$MODEL" "$TASK" pref-only semi 0.1 $COMMON_FLAGS --neg-typcorr --no-force-same-x

echo ""
echo "=== ALL VARIANTS DONE — $(date) ==="
