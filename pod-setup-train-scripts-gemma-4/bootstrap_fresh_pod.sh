#!/bin/bash
# bootstrap_fresh_pod.sh <no_tc|self_tc|neg_tc>
#
# One committed entry point for a FRESH RunPod pod: clone rankalign (longform),
# run the proven gemma-4 setup recipe (venv + deps + transformers-from-main +
# pre-download the 62 GB model), then hand off to the reproducible
# run_arm_3epoch.sh for that arm. No ad-hoc steps.
#
# HF_TOKEN must be set in the environment (gated gemma-4 download). Never
# hardcoded here.
set -euo pipefail

ARM="${1:?usage: bootstrap_fresh_pod.sh <no_tc|self_tc|neg_tc>}"
: "${HF_TOKEN:?HF_TOKEN not set — required for gated gemma-4 download}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p /workspace/logs
cd /workspace

# Full clone (not shallow) so a later pin/reset is clean.
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

# Sanity: the reproducible runner and task data must be present.
test -f /workspace/rankalign/pod-setup-train-scripts-gemma-4/run_arm_3epoch.sh
ntasks=$(ls /workspace/rankalign/data/humaneval/v2.1correct-upper/humaneval_*.csv | wc -l)
echo "task csvs present: $ntasks"
[ "$ntasks" -ge 80 ] || { echo "FATAL: expected >=80 task csvs, got $ntasks"; exit 1; }

echo "=== handing off to run_arm_3epoch.sh $ARM ==="
exec bash /workspace/rankalign/pod-setup-train-scripts-gemma-4/run_arm_3epoch.sh "$ARM"
