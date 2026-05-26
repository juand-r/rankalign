#!/bin/bash
# setup_and_eval_s13ep0.sh <MODE> <USE_BT>
#
# s13 = SFT-lo + consistency-ft. gemma-4-31B-it, delta 2.14, EPOCH 0,
# trained on humaneval-v2.1correct-upper-all. Adapter (LoRA) lives at:
#   latkes/rankalign-v7-gemma-4-31B-it-d2.14-e0-humaneval-v2.1correct-upper-all-p0-nv1-ng1-cft-lo0.1-fix1
# base_model_name_or_path = google/gemma-4-31B-it (verified in adapter_config.json).
#
# MODE: self-typicality | neg-typicality
# USE_BT: 0 (no base-typicality) | 1 (with base-typicality)
#   self-typicality bt0 -> PMI self    | bt1 -> PMI base
#   neg-typicality  bt0 -> Neg self    | bt1 -> Neg base
#   (all four also contribute to the Raw column)
#
# Designed for a torch>=2.5 base image (e.g.
# runpod/pytorch:0.7.0-cu1241-torch251-ubuntu2204). torch comes from the image
# via --system-site-packages; requirements-gemma4.txt does NOT install torch.
#
# Idempotent: per-task done markers under outputs/.done make it fully resumable.
# Expects HF_TOKEN in the environment (gated gemma-4 download).
set -uo pipefail

MODE="${1:?usage: setup_and_eval_s13ep0.sh <self-typicality|neg-typicality> <0|1>}"
USE_BT="${2:?usage: setup_and_eval_s13ep0.sh <MODE> <0|1>}"

export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /workspace/.cache/huggingface /workspace/logs /workspace/models_g4it

# ---- STEP 1: clone rankalign (self-healing: re-clone if eval script absent) ----
# A migrated/partial volume can leave an empty rankalign dir; key on the actual
# script, not just the directory, so we always end up with a usable checkout.
if [ ! -f /workspace/rankalign/scripts/eval_by_claude.py ]; then
    echo "[$(date -u +%H:%M:%S)] Cloning rankalign (longform)..."
    rm -rf /workspace/rankalign
    git clone -b longform --depth 1 https://github.com/juand-r/rankalign.git /workspace/rankalign
fi

# ---- STEP 2: venv + pinned deps (SKIP entirely if venv already healthy) ----
# "Healthy" = core imports work AND CUDA is usable. Re-running the wrapper on a
# good venv must NOT re-run `pip install --ignore-installed`, which re-pulls the
# latest torch (cu130 = CUDA 13) and re-breaks a host whose driver is < CUDA 13.
# So gate the whole install behind a health check; otherwise (re)build cleanly.
VENV=/workspace/.venv
if "$VENV/bin/python" -c "import huggingface_hub, transformers, peft, torch; assert torch.cuda.is_available()" 2>/dev/null; then
    source "$VENV/bin/activate"
    echo "[$(date -u +%H:%M:%S)] venv already healthy (core imports + CUDA) — skipping deps install."
else
    echo "[$(date -u +%H:%M:%S)] (Re)building venv (--system-site-packages)..."
    "$VENV/bin/python" -c "import sys" 2>/dev/null || { rm -rf "$VENV"; python3 -m venv "$VENV" --system-site-packages; }
    source "$VENV/bin/activate"
    pip install --quiet --upgrade pip
    pip install --quiet hf_transfer
    pip install --quiet --ignore-installed -r /workspace/rankalign/requirements-gemma4.txt
    # Pin a matched torch+torchvision built for cu124. --ignore-installed pulls the
    # latest torch (2.12+cu130 = CUDA 13), which silently CPU-falls-back on hosts
    # whose driver only supports CUDA <13 (~150s/candidate, unusable). cu124 runs
    # on any driver >=12.4. Also fixes the torchvision ABI mismatch that breaks the
    # peft/transformers import. (Documented gemma-4 pair: torch 2.5.1 + tv 0.20.1.)
    pip install --quiet torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
    echo "[$(date -u +%H:%M:%S)] deps installed + torch 2.5.1/torchvision 0.20.1 (cu124) pinned."
fi

# ---- STEP 2b: fail loud if deps install was incomplete ----
# A silent `pip install` failure (e.g. a pin needing a newer Python) must abort
# here, NOT cascade into 82 model-less eval calls. set -e is unsafe in the eval
# loop (per-task failures are expected), so guard explicitly.
python -c "import huggingface_hub, transformers, peft, pandas, sklearn, accelerate, safetensors" \
    || { echo "[$(date -u +%H:%M:%S)] FATAL: requirements-gemma4.txt did not fully install (missing core deps). Check Python version vs pinned pandas/numpy/scipy." >&2; exit 1; }

# ---- STEP 2c: fail loud if torch is too old OR CUDA is not usable ----
# Must ABORT (not continue) if CUDA is unavailable, else the 31B model falls back
# to CPU and "runs" at ~150s/candidate while looking alive. set -e is unsafe in
# the eval loop, so guard explicitly here.
python - <<'PYEOF' || { echo "[$(date -u +%H:%M:%S)] FATAL: torch/CUDA check failed — aborting to avoid silent CPU fallback." >&2; exit 1; }
import torch
ver = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
assert ver >= (2, 5), f"torch {torch.__version__} too old for gemma-4 (need >=2.5)"
assert torch.cuda.is_available(), "CUDA not visible from venv (driver/cu build mismatch)"
print(f"[env] torch {torch.__version__} cuda OK")
PYEOF

# ---- STEP 3: pre-download base model (verify by listing, retry on failure) ----
python - <<'PYEOF'
import os, time
from huggingface_hub import snapshot_download
model = "google/gemma-4-31B-it"
cache = os.environ["HF_HUB_CACHE"]
snap = os.path.join(cache, "models--google--gemma-4-31B-it", "snapshots")
if os.path.isdir(snap) and os.listdir(snap):
    print("  base model already cached")
else:
    for attempt in range(5):
        try:
            t0 = time.time()
            snapshot_download(repo_id=model, cache_dir=cache,
                              token=os.environ.get("HF_TOKEN", ""),
                              allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt", "*.model"],
                              max_workers=8)
            print(f"  base model downloaded in {time.time()-t0:.0f}s")
            break
        except Exception as e:
            print(f"  attempt {attempt+1} failed: {e}")
            time.sleep(15)
    else:
        raise SystemExit("FATAL: base model download failed after 5 attempts")
PYEOF

# ---- STEP 4: download s13 epoch0 adapter ----
ADAPTER_REPO="latkes/rankalign-v7-gemma-4-31B-it-d2.14-e0-humaneval-v2.1correct-upper-all-p0-nv1-ng1-cft-lo0.1-fix1"
ADAPTER_DIR="/workspace/models_g4it/v7-google--gemma-4-31B-it-delta2.14-epoch0--humaneval-v2.1correct-upper-all--d2g--random--alpha1.0--full-completion--pref0.0--nllv1.0--nllg1.0--cft--labelonly0.1--fix1"
if [ ! -d "$ADAPTER_DIR" ] || [ -z "$(ls -A "$ADAPTER_DIR" 2>/dev/null)" ]; then
    echo "[$(date -u +%H:%M:%S)] Downloading s13 epoch0 adapter from HF..."
    python - <<PYEOF
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id="$ADAPTER_REPO",
    local_dir="$ADAPTER_DIR",
    token=os.environ.get("HF_TOKEN", ""),
    ignore_patterns=["*.gguf", "*.bin", "README.md"],
)
print("Adapter downloaded to $ADAPTER_DIR")
PYEOF
fi

# ---- STEP 5: eval loop over all humaneval-v2.1correct-upper tasks ----
HE_TASK="humaneval-v2.1correct-upper"
BASE="google/gemma-4-31B-it"
RANKALIGN_DIR="/workspace/rankalign"
OUTDIR="/workspace/outputs"
DONE_DIR="$OUTDIR/.done"
LOGDIR="/workspace/logs"
S="13"
EP="0"
MODE_ARG="--${MODE}"
MARKER_TAG="bt${USE_BT}"

if [ "$USE_BT" = "1" ]; then
    BT_ARGS="--base-typicality --base-model $BASE"
else
    BT_ARGS=""
fi

mkdir -p "$OUTDIR" "$DONE_DIR"
cd "$RANKALIGN_DIR/scripts"

TASKS=$(ls "$RANKALIGN_DIR/data/humaneval/v2.1correct-upper/humaneval_"*.csv \
    | xargs -n1 basename | sed 's/\.csv$//' \
    | sed "s/^/${HE_TASK}-/" | tr '\n' ' ')
NT=$(echo "$TASKS" | wc -w)

EVAL_LOG="$LOGDIR/eval_s${S}_ep${EP}_${MODE}_${MARKER_TAG}.log"
RUN_LOG="$LOGDIR/run_s13_epoch0.log"

echo "[$(date -u +%H:%M:%S)] ==== eval s${S} epoch${EP} ${MODE_ARG} bt=${USE_BT} ====" | tee "$EVAL_LOG"
echo "[$(date -u +%H:%M:%S)] adapter=${ADAPTER_DIR}" | tee -a "$EVAL_LOG"
echo "[$(date -u +%H:%M:%S)] tasks=$NT" | tee -a "$EVAL_LOG"

n=0
for T in $TASKS; do
    marker="${DONE_DIR}/s${S}_ep${EP}_${MODE}_${MARKER_TAG}_${T}.done"
    if [ -f "$marker" ]; then n=$((n+1)); continue; fi
    python eval_by_claude.py \
        --model "$ADAPTER_DIR" \
        --task "$T" \
        --split_type random \
        --gen-shots zero \
        --disc-shots zero \
        "$MODE_ARG" \
        $BT_ARGS \
        --validator-log-odds \
        --save-scores-csv \
        --outputs-dir "$OUTDIR" \
        >> "$EVAL_LOG" 2>&1 \
        && touch "$marker"
    n=$((n+1))
    [ $((n % 10)) -eq 0 ] && echo "[$(date -u +%H:%M:%S)] progress: $n/$NT" | tee -a "$EVAL_LOG"
done

echo "[$(date -u +%H:%M:%S)] SETTING13_EPOCH0_${MODE}_${MARKER_TAG}_COMPLETE ($n/$NT)" | tee -a "$EVAL_LOG"
echo "SETTING13_EPOCH0_${MODE}_${MARKER_TAG}_COMPLETE" >> "$RUN_LOG"
