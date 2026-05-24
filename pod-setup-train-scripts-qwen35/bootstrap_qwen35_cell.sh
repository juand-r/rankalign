#!/bin/bash
# bootstrap_qwen35_cell.sh DATASET SETTING
#
# Entry point for a FRESH RunPod pod (TAUR account). SCP or clone to /tmp/ on the pod,
# then run it. Clones rankalign, runs setup, hands off to run_qwen35_cell.sh.
#
# Usage (run via nohup on pod):
#   export HF_TOKEN="hf_..."
#   nohup bash /tmp/bootstrap_qwen35_cell.sh DATASET SETTING \
#       > /workspace/logs/bootstrap_DATASET_SETTING.log 2>&1 < /dev/null &
#
# DATASET: persona | membership | ifeval
# SETTING: s1 | s2 | s4 | s7
#
# IMPORTANT: Uses requirements-gemma4.txt (transformers 5.8.1), NOT requirements.txt.
set -euo pipefail

DATASET="${1:?DATASET required (persona|membership|ifeval)}"
SETTING="${2:?SETTING required (s1|s2|s4|s7)}"

: "${HF_TOKEN:?HF_TOKEN must be set before running}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p /workspace/logs /workspace/models_q35 /workspace/outputs
LOG=/workspace/logs/bootstrap_${DATASET}_${SETTING}.log
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -u +%FT%TZ)] === bootstrap_qwen35_cell.sh: $DATASET $SETTING ==="

cd /workspace

# Clone or update rankalign (longform branch).
if [ ! -d /workspace/rankalign ]; then
    git clone -b longform https://github.com/juand-r/rankalign.git /workspace/rankalign
fi
git -C /workspace/rankalign fetch origin longform
git -C /workspace/rankalign reset --hard origin/longform
echo "[$(date -u +%FT%TZ)] rankalign at: $(git -C /workspace/rankalign log --oneline -1)"

# Sanity: scripts must exist in repo.
test -f /workspace/rankalign/setup-runpod-qwen35.sh || {
    echo "FATAL: setup-runpod-qwen35.sh not in repo"
    exit 1
}
test -f /workspace/rankalign/scripts/run_qwen35_cell.sh || {
    echo "FATAL: scripts/run_qwen35_cell.sh not in repo"
    exit 1
}

# Setup: venv (requirements-gemma4.txt) + bitsandbytes fix + model download.
bash /workspace/rankalign/setup-runpod-qwen35.sh

echo "[$(date -u +%FT%TZ)] === handing off to run_qwen35_cell.sh $DATASET $SETTING ==="
exec bash /workspace/rankalign/scripts/run_qwen35_cell.sh "$DATASET" "$SETTING"
