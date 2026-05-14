#!/bin/bash
# Initial setup for RunPod pod running gemma-4 inference / scoring.
#
# Why this exists:
#   The default requirements.txt is pinned to transformers==4.46.2, which
#   cannot load gemma-4. Running it then upgrading transformers separately
#   breaks scipy/sklearn binary compat (we ate ~10 minutes of debugging
#   this every time). This script does the right thing in one shot.
#
# Usage (on a fresh or resumed pod):
#   cd /workspace
#   bash setup-runpod-gemma4.sh
#
# Idempotent — safe to re-run after a pod resume.

set -uo pipefail  # no -e: keep going if any non-critical step fails

export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1
# HF_TOKEN must be set in the pod environment (env var or pod template). Do NOT hardcode here.
if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN not set — gated model downloads will fail." >&2
fi
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

mkdir -p /workspace/.cache/huggingface /workspace/logs
cd /workspace

# Persist env for future SSH sessions
cat > /etc/profile.d/hf_env.sh <<EOF
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1
EOF

echo "=== install deps (gemma-4 compatible) ==="
pip install --quiet hf_transfer 2>&1 | tail -2
# Clone rankalign if needed (we use its requirements-runpod-gemma4.txt)
if [ ! -d /workspace/rankalign ]; then
    echo "=== clone rankalign ==="
    git clone -b longform --depth 1 https://github.com/juand-r/rankalign.git /workspace/rankalign 2>&1 | tail -2
fi

# Install the runpod-friendly requirements in one shot.
echo "=== install requirements-runpod-gemma4 (compatible versions) ==="
pip install --quiet --upgrade -r /workspace/rankalign/requirements-runpod-gemma4.txt 2>&1 | tail -3

# Then install transformers from main (separately — git+ doesn't play nicely
# in a constraints file with other pins).
echo "=== install transformers from main ==="
pip install --quiet --upgrade "git+https://github.com/huggingface/transformers.git" 2>&1 | tail -3

# Verify the import chain works (this is the chain that breaks if
# sklearn/scipy got clobbered by stale wheels — fail fast here).
python -c "
import transformers, huggingface_hub, tokenizers, sklearn, scipy
from transformers import AutoModelForCausalLM, AutoTokenizer
print(f'transformers: {transformers.__version__}')
print(f'huggingface_hub: {huggingface_hub.__version__}')
print(f'tokenizers: {tokenizers.__version__}')
print(f'sklearn: {sklearn.__version__}')
print(f'scipy: {scipy.__version__}')
print('all imports ok')
" || {
    echo "*** import chain broken — try: pip install --force-reinstall --no-deps scikit-learn scipy" >&2
    exit 1
}

# Verify CUDA / torch
python -c "
import torch
print(f'torch: {torch.__version__}, cuda: {torch.cuda.is_available()}')
"

# Pre-download gemma-4-31B-it if not already cached
echo "=== pre-download gemma-4-31B-it (if not cached) ==="
python <<'PYEOF'
import os, time
from huggingface_hub import snapshot_download

model = "google/gemma-4-31B-it"
cache_dir = os.environ.get("HF_HUB_CACHE", "/workspace/.cache/huggingface/hub")
expected_dir = os.path.join(cache_dir, f"models--{model.replace('/', '--')}")
if os.path.exists(expected_dir):
    print(f"  already cached at {expected_dir}")
else:
    t0 = time.time()
    snapshot_download(repo_id=model, cache_dir=cache_dir)
    print(f"  downloaded in {time.time()-t0:.0f}s")
PYEOF

echo ""
echo "=== disk usage ==="
df -h /workspace 2>&1 | tail -2
du -sh /workspace/.cache/huggingface 2>&1

echo ""
echo "=== SETUP COMPLETE ==="
date
