#!/bin/bash
# setup-mll-qwen35-fla.sh
# ---------------------------------------------------------------------------
# Builds /datastor2/jdr/venvs/qwen35 = a clone of the working gemma4 venv PLUS
# the flash-linear-attention fast kernel for Qwen3.5's gated-delta-rule attention.
#
# WHY: training Qwen3.5-9B on the gemma4 venv falls back to the *torch* gated-delta-rule
# implementation (the modeling code warns "fast path is not available"), giving ~11.5 s/it
# on 2x A40. With fla the same run is ~4 s/it (~2.9x faster) -> a full 3-epoch run drops
# from ~49 h to ~17 h. Verified 2026-06-06 on mll (2x A40), canary job 43931.
#
# VERSION RECIPE (deltas from the gemma4 base — do NOT touch torch or CUDA):
#   fla-core==0.5.0       the REAL fla implementation. NB: PyPI `flash-linear-attention`
#                         0.5.0 is an empty namespace shell (fla.__file__ is None); the code
#                         lives in `fla-core`.
#   triton==3.3.0         fla-core 0.5.0 needs triton>=3.3 (gemma4's triton 3.1.0 dies at
#                         import: ValueError "'BT' is not in list" in the autotuner).
#   bitsandbytes==0.49.2  0.45.0 imports the removed `triton.ops.matmul_perf_model`, which
#                         crashes under triton 3.3 (ModuleNotFoundError: No module 'triton.ops').
#   torch 2.5.1+cu124 / transformers 5.8.0.dev0 / peft 0.14.0  stay UNCHANGED.
#
# is_fast_path_available stays False (no causal_conv1d — needs nvcc, which mll lacks), so the
# "fast path not available" warning still prints. That is HARMLESS: modeling_qwen3_5.py uses
# `self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_...` (line ~422), so the
# expensive linear-attention kernel uses fla regardless; causal_conv1d is only a cheap 1D conv
# with a torch fallback.
#
# Idempotent. Run on the mll login node (no GPU needed to install).
set -uo pipefail
SRC=/datastor2/jdr/venvs/gemma4
DST=/datastor2/jdr/venvs/qwen35

if [ ! -d "$DST" ]; then
    echo "[$(date)] cloning $SRC -> $DST (~6.7G)"
    cp -a "$SRC" "${DST}.tmp" && mv "${DST}.tmp" "$DST"
fi
echo "[$(date)] rehoming venv paths"
grep -rlI "$SRC" "$DST/bin" 2>/dev/null | xargs -r sed -i "s|$SRC|$DST|g"

source "$DST/bin/activate"

# moe.py patch (torch custom_op infer_schema vs transformers string annotations) — idempotent
MOE=$(python -c "import transformers,os;print(os.path.join(os.path.dirname(transformers.__file__),'integrations','moe.py'))")
if grep -q '^from __future__ import annotations' "$MOE"; then
    sed -i 's/^from __future__ import annotations$/# from __future__ import annotations (patched: torch infer_schema compat)/' "$MOE"
    echo "[$(date)] patched moe.py"
fi

echo "[$(date)] installing fast-attn kernels (--no-deps to leave torch/cuda untouched)"
pip install --no-deps fla-core==0.5.0 'triton==3.3.0' 'bitsandbytes==0.49.2'

echo "[$(date)] verifying"
python - <<'PY'
import torch, triton, bitsandbytes
import fla.ops.gated_delta_rule as g
from fla.modules import FusedRMSNormGated
assert hasattr(g, "chunk_gated_delta_rule")
print("OK: torch", torch.__version__, "triton", triton.__version__,
      "bnb", bitsandbytes.__version__, "fla-core + FusedRMSNormGated import OK")
PY
echo "[$(date)] qwen35 venv ready."
echo "Canary:  sbatch --time=24:00:00 scripts/run_qwen35_cell_mll.sbatch ifeval s1"
