#!/usr/bin/env bash
# run_ra9b_ifeval_id_eval.sh SETTING
#
# Runs eval on ID ifeval prompts (N > 21) for gemma-2-9b-it.
# Downloads the epoch2 merged model from HF (TAUR-dev) and the base model.
# Runs: basetyp, basetypneg, self, neg TC variants (as appropriate for setting).
#   s1/s2/s3: all four variants
#   s4: basetyp + self only
#   s7: basetypneg + neg only
#
# Usage (on pod, from /workspace/rankalign/scripts/):
#   bash run_ra9b_ifeval_id_eval.sh s1

set -euo pipefail
cd /workspace/rankalign/scripts

SETTING="${1:?SETTING required (s1|s2|s3|s4|s7)}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set}"

LOG_DIR="/workspace/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/ra9b_id_eval.log"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -u +%FT%TZ)] === run_ra9b_ifeval_id_eval.sh SETTING=$SETTING ==="

# ── HF cache setup ─────────────────────────────────────────────────────────
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"

# ── Models ──────────────────────────────────────────────────────────────────
HF_REPO="TAUR-dev/rankalign-v7-gemma2-9b-it-ifeval-${SETTING}-ep2"
BASE_MODEL_ID="google/gemma-2-9b-it"
EVAL_MODEL_DIR="/workspace/eval_model_${SETTING}"
OUTPUTS_DIR="/workspace/rankalign/outputs"
IFEVAL_DATA_DIR="/workspace/rankalign/data/fixed-prompts-ifeval"
mkdir -p "$OUTPUTS_DIR"

# ── Download fine-tuned model ───────────────────────────────────────────────
echo "[$(date -u +%FT%TZ)] Downloading fine-tuned model: $HF_REPO"
python3 - <<PYEOF
import os, time
from huggingface_hub import snapshot_download
for attempt in range(5):
    try:
        path = snapshot_download(
            repo_id="$HF_REPO",
            token=os.environ["HF_TOKEN"],
            max_workers=8,
        )
        print(f"Downloaded to: {path}")
        break
    except Exception as e:
        print(f"Attempt {attempt+1} failed: {e}")
        if attempt < 4: time.sleep(15)
        else: raise
PYEOF

# Create symlink
FT_MODEL_PATH=$(python3 -c "
import os
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id='$HF_REPO', token=os.environ['HF_TOKEN'], local_files_only=True)
print(p)
")
ln -sfn "$FT_MODEL_PATH" "$EVAL_MODEL_DIR"
echo "[$(date -u +%FT%TZ)] eval model -> $FT_MODEL_PATH"

# ── Download base model ──────────────────────────────────────────────────────
echo "[$(date -u +%FT%TZ)] Downloading base model: $BASE_MODEL_ID"
python3 - <<PYEOF
import os, time
from huggingface_hub import snapshot_download
for attempt in range(5):
    try:
        path = snapshot_download(
            repo_id="$BASE_MODEL_ID",
            token=os.environ["HF_TOKEN"],
            max_workers=8,
        )
        print(f"Downloaded to: {path}")
        break
    except Exception as e:
        print(f"Attempt {attempt+1} failed: {e}")
        if attempt < 4: time.sleep(15)
        else: raise
PYEOF
echo "[$(date -u +%FT%TZ)] Base model downloaded."

# ── Build ID task list (N > 21) ─────────────────────────────────────────────
mapfile -t _id_ns < <(
    for f in "$IFEVAL_DATA_DIR"/gpt_ifeval_results_prompt_*.jsonl; do
        n="${f##*_prompt_}"; n="${n%.jsonl}"
        [[ "$n" =~ ^[0-9]+$ ]] && (( n > 21 )) && echo "$n"
    done | sort -n
)
ID_TASKS=()
for n in "${_id_ns[@]}"; do ID_TASKS+=("ifeval-prompt_${n}"); done
echo "[$(date -u +%FT%TZ)] ${#ID_TASKS[@]} ID tasks: ${ID_TASKS[*]}"

# ── Activate venv ────────────────────────────────────────────────────────────
source /workspace/.venv/bin/activate 2>/dev/null || true

# ── Self TC (no base model) ──────────────────────────────────────────────────
echo "[$(date -u +%FT%TZ)] --- self TC (all ID tasks) ---"
for EVAL_TASK in "${ID_TASKS[@]}"; do
    echo "[$(date -u +%FT%TZ)] self: $EVAL_TASK"
    python eval_by_claude.py \
        --model "$EVAL_MODEL_DIR" \
        --task "$EVAL_TASK" \
        --split_type random \
        --disc-shots zero \
        --gen-shots zero \
        --outputs-dir "$OUTPUTS_DIR" \
        --validator-log-odds \
        --self-typicality \
        --save-scores-csv
done
echo "[$(date -u +%FT%TZ)] self TC done"

# ── Neg TC (no base model) — s1/s2/s3/s7 ─────────────────────────────────────
if [[ "$SETTING" != "s4" ]]; then
    echo "[$(date -u +%FT%TZ)] --- neg TC (all ID tasks) ---"
    for EVAL_TASK in "${ID_TASKS[@]}"; do
        echo "[$(date -u +%FT%TZ)] neg: $EVAL_TASK"
        python eval_by_claude.py \
            --model "$EVAL_MODEL_DIR" \
            --task "$EVAL_TASK" \
            --split_type random \
            --disc-shots zero \
            --gen-shots zero \
            --outputs-dir "$OUTPUTS_DIR" \
            --validator-log-odds \
            --neg-typicality \
            --save-scores-csv
    done
    echo "[$(date -u +%FT%TZ)] neg TC done"
fi

# ── Basetyp/PMI (--base-typicality --self-typicality) — s1/s2/s3/s4 ──────────
if [[ "$SETTING" != "s7" ]]; then
    echo "[$(date -u +%FT%TZ)] --- basetyp (PMI) on ID tasks ---"
    for EVAL_TASK in "${ID_TASKS[@]}"; do
        echo "[$(date -u +%FT%TZ)] basetyp: $EVAL_TASK"
        python eval_by_claude.py \
            --model "$EVAL_MODEL_DIR" \
            --task "$EVAL_TASK" \
            --split_type random \
            --disc-shots zero \
            --gen-shots zero \
            --outputs-dir "$OUTPUTS_DIR" \
            --validator-log-odds \
            --base-typicality --self-typicality \
            --base-model-name "$BASE_MODEL_ID" \
            --save-scores-csv
    done
    echo "[$(date -u +%FT%TZ)] basetyp done"
fi

# ── Basetypneg/Neg (--base-typicality --neg-typicality) — s1/s2/s3/s7 ────────
if [[ "$SETTING" != "s4" ]]; then
    echo "[$(date -u +%FT%TZ)] --- basetypneg (Neg) on ID tasks ---"
    for EVAL_TASK in "${ID_TASKS[@]}"; do
        echo "[$(date -u +%FT%TZ)] basetypneg: $EVAL_TASK"
        python eval_by_claude.py \
            --model "$EVAL_MODEL_DIR" \
            --task "$EVAL_TASK" \
            --split_type random \
            --disc-shots zero \
            --gen-shots zero \
            --outputs-dir "$OUTPUTS_DIR" \
            --validator-log-odds \
            --base-typicality --neg-typicality \
            --base-model-name "$BASE_MODEL_ID" \
            --save-scores-csv
    done
    echo "[$(date -u +%FT%TZ)] basetypneg done"
fi

touch /workspace/RA9B_ID_DONE
echo "[$(date -u +%FT%TZ)] === RA9B IFEVAL ID EVAL ALL DONE SETTING=$SETTING ==="
