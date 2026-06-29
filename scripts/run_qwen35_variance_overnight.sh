#!/bin/bash
# run_qwen35_variance_overnight.sh RUN_ID
#
# Pod-side runner for the qwen IFEval RankAlign (s2) variance-vs-environment test.
# Replicates the ORIGINAL pod recipe EXACTLY by chaining the committed scripts that
# produced the 83.2 checkpoint (setup-runpod-qwen35.sh + run_qwen35_cell.sh), then adds
# the NO_BASE eval pass so we get the no-base self-/neg- CSVs (the mode 83.2 lives in).
#
# Each of the 3 pods runs this with a distinct RUN_ID (1/2/3). Runs differ only by the
# unseeded pair shuffle + GPU nondeterminism -> measures run-to-run spread in the pod env.
# NOTHING about the recipe is changed: same image (runpod/pytorch:2.4.0), requirements-gemma4
# (transformers 5.8.1, torch 2.4.1), --delta 0.15 --delta-bins 10, 3 epochs, LoRA.
#
# Usage on a fresh pod:
#   export HF_TOKEN="hf_..."
#   nohup bash /tmp/run_qwen35_variance_overnight.sh 1 > /workspace/logs/variance_run.log 2>&1 < /dev/null &
set -euo pipefail
RUN_ID="${1:?RUN_ID required (1|2|3)}"
: "${HF_TOKEN:?HF_TOKEN must be set}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p /workspace/logs
LOG=/workspace/logs/variance_run${RUN_ID}.log
exec > >(tee -a "$LOG") 2>&1
echo "[$(date -u +%FT%TZ)] === variance run ${RUN_ID} START ==="

# 1. clone/update repo (longform) — same as bootstrap_qwen35_cell.sh
if [ ! -d /workspace/rankalign ]; then
    git clone -b longform https://github.com/juand-r/rankalign.git /workspace/rankalign
fi
git -C /workspace/rankalign fetch origin longform
git -C /workspace/rankalign reset --hard origin/longform
echo "[$(date -u +%FT%TZ)] rankalign at: $(git -C /workspace/rankalign log --oneline -1)"

# 2. setup (idempotent: venv + requirements-gemma4 + moe patch + Qwen3.5-9B download)
bash /workspace/rankalign/setup-runpod-qwen35.sh

# 3. train + default eval (produces scores_basetyp-/basetypneg- CSVs)
bash /workspace/rankalign/scripts/run_qwen35_cell.sh ifeval s2

# 4. NO_BASE eval pass (model dir now exists -> skips train; produces scores_self-/neg- CSVs)
NO_BASE=1 bash /workspace/rankalign/scripts/run_qwen35_cell.sh ifeval s2

# 5. done marker (monitoring waits on this before upload/backup/stop)
echo "$RUN_ID" > /workspace/VARIANCE_S2_RUN${RUN_ID}_DONE
echo "[$(date -u +%FT%TZ)] === variance run ${RUN_ID} DONE ==="
echo "  models: /workspace/models_q35   scores: /workspace/outputs"
ls /workspace/outputs/scores_*ifeval*_test* 2>/dev/null | wc -l | sed 's/^/  score CSVs: /'
