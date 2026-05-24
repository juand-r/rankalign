#!/bin/bash
# setup-runpod-gemma4.sh — one-shot setup for a fresh OR resumed RunPod pod.
#
# Installs requirements-gemma4.txt inside a venv that inherits the pod image's
# torch 2.5.1+cu124 via --system-site-packages.  torch is NOT in the
# requirements file — it must come from the base image.
#
# Fails loud (set -e). Idempotent — safe to re-run after a pod resume.
#
# Usage:
#   cd /workspace && bash rankalign/setup-runpod-gemma4.sh
#
# Run scripts as: /workspace/.venv/bin/python <script>.py
# Or: source /workspace/.venv/bin/activate && python ...

set -euo pipefail

VENV=/workspace/.venv

export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p $HF_HUB_CACHE /workspace/logs
cd /workspace

# Persist env for future login shells.
cat > /etc/profile.d/hf_env.sh <<EOF
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=\$HF_HOME/hub
export TRANSFORMERS_CACHE=\$HF_HUB_CACHE
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1
[ -f $VENV/bin/activate ] && source $VENV/bin/activate
EOF

if [ ! -d /workspace/rankalign ]; then
    echo "=== clone rankalign ==="
    git clone -b longform --depth 1 https://github.com/juand-r/rankalign.git /workspace/rankalign
fi

if [ ! -f $VENV/bin/python ]; then
    echo "=== create venv at $VENV (with --system-site-packages for torch/CUDA) ==="
    python3 -m venv $VENV --system-site-packages
fi

# Use venv from this point onward.
source $VENV/bin/activate
echo "python: $(which python)"
python --version

echo "=== upgrade pip + install hf_transfer ==="
pip install --quiet --upgrade pip
pip install --quiet hf_transfer

echo "=== install requirements-gemma4.txt (pinned deps; torch comes from pod image) ==="
# --ignore-installed sidesteps the blinker distutils issue on some images.
# torch is NOT in this file — inherited via --system-site-packages.
pip install --quiet --ignore-installed -r /workspace/rankalign/requirements-gemma4.txt

echo "=== verify full import chain ==="
python <<'PYEOF'
import transformers, huggingface_hub, tokenizers
import sklearn, scipy, pandas, numpy, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
assert torch.cuda.is_available(), "CUDA not visible from venv"
print(f"  transformers: {transformers.__version__}")
print(f"  huggingface_hub: {huggingface_hub.__version__}")
print(f"  tokenizers: {tokenizers.__version__}")
print(f"  sklearn: {sklearn.__version__}")
print(f"  scipy: {scipy.__version__}")
print(f"  pandas: {pandas.__version__}")
print(f"  numpy: {numpy.__version__}")
print(f"  torch: {torch.__version__} cuda={torch.cuda.is_available()}")
print("  ALL IMPORTS OK")
PYEOF

if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "=== pre-download gemma-4-31B-it if not cached ==="
python <<'PYEOF'
import os, time
from huggingface_hub import snapshot_download
model = "google/gemma-4-31B-it"
cache_dir = os.environ.get("HF_HUB_CACHE", "/workspace/.cache/huggingface/hub")
expected_dir = os.path.join(cache_dir, f"models--{model.replace('/', '--')}")
snapshots = os.path.join(expected_dir, "snapshots")
if os.path.isdir(snapshots) and os.listdir(snapshots):
    print(f"  already cached at {expected_dir}")
else:
    t0 = time.time()
    snapshot_download(repo_id=model, cache_dir=cache_dir)
    print(f"  downloaded in {time.time()-t0:.0f}s")
PYEOF

echo ""
echo "=== disk ==="
df -h /workspace | tail -2
du -sh /workspace/.cache/huggingface 2>/dev/null || true

echo ""
echo "=== SETUP COMPLETE ==="
echo "To run scripts: $VENV/bin/python <script>.py"
echo "Or (in a login shell): source $VENV/bin/activate && python <script>.py"
date
