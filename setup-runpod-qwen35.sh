#!/bin/bash
# setup-runpod-qwen35.sh — one-shot setup for a fresh OR resumed RunPod pod running Qwen3.5-9B.
#
# Uses requirements-gemma4.txt (transformers 5.8.1, torch inherited from pod image 2.4.0).
# Idempotent — safe to re-run after a pod resume. Fails loud (set -e).
#
# Usage:
#   export HF_TOKEN="hf_..."
#   bash /workspace/rankalign/setup-runpod-qwen35.sh
set -euo pipefail

VENV=/workspace/.venv

export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$HF_HUB_CACHE" /workspace/logs /workspace/models_q35 /workspace/outputs
cd /workspace

# Persist env vars for future login shells.
cat > /etc/profile.d/hf_env.sh <<'EOF'
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f /workspace/.venv/bin/activate ] && source /workspace/.venv/bin/activate
EOF

echo "=== create venv at $VENV (--system-site-packages inherits pod-image torch) ==="
if [ ! -f "$VENV/bin/python" ]; then
    python3 -m venv "$VENV" --system-site-packages
fi
source "$VENV/bin/activate"
echo "python: $(which python) — $(python --version)"

echo "=== upgrade pip + hf_transfer ==="
pip install --quiet --upgrade pip
pip install --quiet hf_transfer

echo "=== install requirements-gemma4.txt (torch excluded — comes from pod image) ==="
pip install --quiet --ignore-installed -r /workspace/rankalign/requirements-gemma4.txt

# bitsandbytes 0.45.x imports triton.ops which was removed in triton 3.x (torch 2.4+).
# See: memory feedback_runpod_pod_setup_gotchas.md item 12.
echo "=== upgrade bitsandbytes for triton 3.x compatibility ==="
pip install --quiet "bitsandbytes>=0.49.2"

echo "=== verify imports ==="
python - <<'PYEOF'
import torch, transformers, peft, accelerate
import sklearn, scipy, pandas, numpy
assert torch.cuda.is_available(), "CUDA not visible — check GPU allocation"
print(f"  torch={torch.__version__} cuda={torch.cuda.is_available()}")
print(f"  transformers={transformers.__version__}")
print(f"  peft={peft.__version__}")
print(f"  sklearn={sklearn.__version__}  numpy={numpy.__version__}")
print("ALL IMPORTS OK")
PYEOF

echo "=== pre-download Qwen/Qwen3.5-9B ==="
if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

python - <<'PYEOF'
import os, time
from huggingface_hub import snapshot_download
model = "Qwen/Qwen3.5-9B"
cache_dir = os.environ.get("HF_HUB_CACHE", "/workspace/.cache/huggingface/hub")
snapshots = os.path.join(cache_dir, f"models--{model.replace('/', '--')}", "snapshots")
if os.path.isdir(snapshots) and os.listdir(snapshots):
    print(f"  already cached: {snapshots}")
else:
    print(f"  downloading {model}...")
    t0 = time.time()
    for attempt in range(5):
        try:
            snapshot_download(
                repo_id=model,
                cache_dir=cache_dir,
                token=os.environ.get("HF_TOKEN"),
            )
            print(f"  downloaded in {time.time()-t0:.0f}s")
            break
        except Exception as e:
            print(f"  attempt {attempt+1} failed: {e}")
            time.sleep(15)
    else:
        raise RuntimeError(f"Failed to download {model} after 5 attempts")
PYEOF

echo ""
echo "=== disk ==="
df -h /workspace | tail -2
du -sh /workspace/.cache/huggingface 2>/dev/null || true
echo ""
echo "=== SETUP COMPLETE ==="
date
