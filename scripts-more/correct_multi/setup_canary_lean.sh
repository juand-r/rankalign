#!/bin/bash
# setup_canary_lean.sh — fast, MINIMAL pod env for the correct-multi canary.
#
# The canary's full 3rd-party surface is verified small: torch (base image),
# transformers (main, for gemma-4), libcst, pandas, huggingface_hub,
# hf_transfer, tokenizers — everything else stdlib. It does NOT run rankalign
# training code, so the heavy full requirements.txt (the ~50-min install) is
# unnecessary here. (setup-runpod-gemma4.sh's "don't simplify" caution is for
# the TRAINING pipeline; this is justified for the canary only.)
#
# Idempotent, fail-loud. HF_TOKEN read from /workspace/.hftoken or env.
#   bash setup_canary_lean.sh
set -euo pipefail

VENV=/workspace/.venv
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HUB_CACHE" /workspace/logs
cd /workspace
[ -f /workspace/.hftoken ] && export HF_TOKEN=$(cat /workspace/.hftoken)
[ -n "${HF_TOKEN:-}" ] && export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

cat > /etc/profile.d/hf_env.sh <<EOF
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=\$HF_HOME/hub
export TRANSFORMERS_CACHE=\$HF_HUB_CACHE
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1
[ -f $VENV/bin/activate ] && source $VENV/bin/activate
EOF

[ -d /workspace/rankalign ] || \
    git clone -q -b longform https://github.com/juand-r/rankalign.git /workspace/rankalign
git -C /workspace/rankalign fetch -q origin longform && \
    git -C /workspace/rankalign reset -q --hard origin/longform
echo "rankalign @ $(git -C /workspace/rankalign log --oneline -1 | cut -c1-9)"

[ -f $VENV/bin/python ] || python3 -m venv $VENV --system-site-packages
source $VENV/bin/activate
pip install --quiet --upgrade pip
echo "=== lean deps: PINNED, verified-working (NEVER git+main) ==="
# git+main transformers is a MOVING TARGET that broke 2026-05-18 (main
# 5.8.0.dev0 needs torch>=2.5 device_mesh; image is torch 2.4.1). The pinned
# release in requirements-canary.txt supports gemma-4 AND works on torch 2.4.
pip install --quiet -r \
    /workspace/rankalign/scripts-more/correct_multi/requirements-canary.txt

echo "=== verify the ACTUAL canary modules import (not just libs) ==="
python <<'PYEOF'
import sys
sys.path.insert(0, "/workspace/rankalign/scripts-more/correct_multi")
sys.path.insert(0, "/workspace/rankalign/scripts")
import torch, transformers, libcst, pandas  # noqa: F401
import transforms, build_canary, analyze_canary, build_v2_1_correct_multi  # noqa: F401
from dataset_builder.build_humaneval_v2_1_correct_upper import validate  # noqa: F401
assert torch.cuda.is_available(), "CUDA not visible from venv"
print(f"  torch {torch.__version__} cuda={torch.cuda.is_available()}")
print(f"  transformers {transformers.__version__}  libcst OK  pandas {pandas.__version__}")
print("  ALL CANARY MODULES IMPORT OK")
PYEOF

echo "=== pre-download gemma-4-31B-it ==="
python <<'PYEOF'
import os, time
from huggingface_hub import snapshot_download
m = "google/gemma-4-31B-it"
cd = os.environ.get("HF_HUB_CACHE", "/workspace/.cache/huggingface/hub")
exp = os.path.join(cd, f"models--{m.replace('/','--')}", "snapshots")
if os.path.isdir(exp) and os.listdir(exp):
    print("  already cached")
else:
    t=time.time(); snapshot_download(repo_id=m, cache_dir=cd)
    print(f"  downloaded in {time.time()-t:.0f}s")
PYEOF

echo "=== LEAN SETUP COMPLETE ==="
date
