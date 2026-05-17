#!/bin/bash
# setup-mll-gemma4.sh — one-shot venv for training gemma-4-31b-it (LoRA) on mll.
#
# mll counterpart of setup-runpod-gemma4.sh. Differences from the RunPod recipe:
#   - mll system python is 3.8 (too old for transformers-main / gemma-4). Build
#     the venv with /lusr python 3.10 (the interpreter venv_lexcons uses).
#   - NO --system-site-packages (clean venv; we pip-install torch==2.5.1, which
#     from PyPI is the cu124 build — A40-compatible, same as venv_lexcons).
#   - Do NOT assert torch.cuda at setup time: this runs on the login node which
#     has no GPU. The CUDA check happens inside the Slurm job instead.
#
# Fails loud (set -e). Idempotent — safe to re-run.
#
# Usage (login node — this is pip/env setup, not GPU compute):
#   bash /datastor1/jdr/gv-gap/rankalign/setup-mll-gemma4.sh
set -euo pipefail

PYBIN=/lusr/opt/python-3.10.16/bin/python3.10
VENV=/datastor2/jdr/venvs/gemma4
RANKALIGN=/datastor1/jdr/gv-gap/rankalign

export HF_HOME=/datastor2/jdr/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p "$HF_HUB_CACHE" /datastor2/jdr/venvs /datastor2/jdr/logs

[ -x "$PYBIN" ] || { echo "FATAL: $PYBIN not found"; exit 1; }
[ -d "$RANKALIGN" ] || { echo "FATAL: rankalign clone not at $RANKALIGN"; exit 1; }
echo "base interpreter: $("$PYBIN" --version 2>&1)"

if [ ! -f "$VENV/bin/python" ]; then
    echo "=== create venv at $VENV (clean, no system-site-packages) ==="
    "$PYBIN" -m venv "$VENV"
fi

source "$VENV/bin/activate"
echo "venv python: $(python --version 2>&1) @ $(which python)"

echo "=== upgrade pip + hf_transfer ==="
pip install --quiet --upgrade pip
pip install --quiet hf_transfer

echo "=== install rankalign requirements.txt (pins torch==2.5.1 -> PyPI cu124) ==="
pip install --quiet -r "$RANKALIGN/requirements.txt"

echo "=== override transformers/hub/tokenizers to git main (gemma-4 support) ==="
pip install --quiet --upgrade huggingface_hub tokenizers \
    "git+https://github.com/huggingface/transformers.git"

echo "=== verify import chain (CUDA check deferred to the Slurm job) ==="
python - <<'PYEOF'
import torch, transformers, tokenizers, huggingface_hub, peft, accelerate
import numpy, scipy, sklearn
from transformers import AutoModelForCausalLM, AutoTokenizer
print(f"  python      : {__import__('sys').version.split()[0]}")
print(f"  torch       : {torch.__version__}")
print(f"  transformers: {transformers.__version__}")
print(f"  tokenizers  : {tokenizers.__version__}")
print(f"  hub         : {huggingface_hub.__version__}")
print(f"  peft        : {peft.__version__}  accelerate: {accelerate.__version__}")
print(f"  numpy {numpy.__version__}  scipy {scipy.__version__}  sklearn {sklearn.__version__}")
print("  ALL IMPORTS OK (torch.cuda.is_available() is checked in the Slurm job)")
PYEOF

echo ""
echo "=== SETUP COMPLETE: $VENV ==="
echo "Use in the Slurm job as: source $VENV/bin/activate"
date
