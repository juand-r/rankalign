#!/bin/bash
# bootstrap_v21correct_multi.sh <SETTING_A> [SETTING_B ...]
#
# One committed entry point for a FRESH RunPod pod running
# humaneval-v2.1correct-multi settings. Clones rankalign, runs setup,
# then hands off to run_settings_v21correct_multi.sh.
#
# Usage (run as nohup on pod):
#   export HF_TOKEN="hf_..."
#   nohup bash /workspace/rankalign/pod-setup-train-scripts-gemma-4/bootstrap_v21correct_multi.sh 3 \
#       > /workspace/logs/bootstrap_s3.log 2>&1 &
#
# HF_TOKEN must be set in the environment (gated gemma-4 download).
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN not set — required for gated gemma-4 download}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p /workspace/logs
cd /workspace

# Full clone or update to latest longform.
if [ ! -d /workspace/rankalign ]; then
    git clone -b longform https://github.com/juand-r/rankalign.git /workspace/rankalign
fi
git -C /workspace/rankalign fetch origin longform
git -C /workspace/rankalign checkout longform
git -C /workspace/rankalign reset --hard origin/longform
echo "rankalign at $(git -C /workspace/rankalign log --oneline -1 | cut -c1-9)"

# Proven idempotent setup (venv, requirements, transformers-from-main,
# pre-download gemma-4-31B-it). Fails loud on any error.
bash /workspace/rankalign/setup-runpod-gemma4.sh

# Sanity: correct-multi task data must be present (82 CSVs).
ntasks=$(ls /workspace/rankalign/data/humaneval/v2.1correct-multi/humaneval_*.csv | wc -l)
echo "correct-multi task csvs present: $ntasks"
[ "$ntasks" -ge 80 ] || { echo "FATAL: expected >=80 task csvs, got $ntasks"; exit 1; }

# Sanity: wrapper script must be present.
test -f /workspace/rankalign/pod-setup-train-scripts-gemma-4/run_settings_v21correct_multi.sh

echo "=== handing off to run_settings_v21correct_multi.sh $* ==="
exec bash /workspace/rankalign/pod-setup-train-scripts-gemma-4/run_settings_v21correct_multi.sh "$@"
