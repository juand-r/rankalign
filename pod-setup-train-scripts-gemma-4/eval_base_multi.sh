#!/bin/bash
# eval_base_multi.sh <self_tc|neg_tc>
#
# Base model eval on humaneval-v2.1correct-multi. Meant to run on a dedicated
# pod while the sibling mode runs on a second pod in parallel.
#
#   bash eval_base_multi.sh self_tc   # pod A
#   bash eval_base_multi.sh neg_tc    # pod B
#
# Models evaluated (same set as the v2.1correct-upper baseline):
#   google/gemma-4-31B-it
#   google/gemma-2-27b-it
#   google/gemma-4-31B
#
# Idempotent: each (model, task) pair is skipped if its .done marker exists.
# Safe to re-run after a crash.
#
# Prerequisites:
#   - setup-runpod-gemma4.sh has been run (venv at /workspace/.venv)
#   - rankalign longform checkout at /workspace/rankalign
#   - HF_TOKEN set (all three models are gated)
set -uo pipefail

MODE="${1:?usage: eval_base_multi.sh <self_tc|neg_tc>}"
case "$MODE" in
  self_tc) TC_FLAG="--self-typicality" ;;
  neg_tc)  TC_FLAG="--neg-typicality" ;;
  *) echo "FATAL: unknown mode '$MODE' (self_tc|neg_tc)"; exit 2 ;;
esac

: "${HF_TOKEN:?HF_TOKEN not set — required for gated model downloads}"

# ---- Dataset ---------------------------------------------------------------
HE_TASK="humaneval-v2.1correct-multi"
DATASET_DIR="${HE_TASK#humaneval-}"   # v2.1correct-multi

# ---- Paths -----------------------------------------------------------------
RANKALIGN_DIR="${RANKALIGN_DIR:-/workspace/rankalign}"
VENV_DIR="${VENV_DIR:-/workspace/.venv}"
OUTDIR="${OUTDIR:-/workspace/outputs}"
LOGDIR="${LOGDIR:-/workspace/logs}"
DONE="${OUTDIR}/.done_base_${MODE}"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "$VENV_DIR/bin/activate"
cd "$RANKALIGN_DIR/scripts"

mkdir -p "$OUTDIR" "$DONE" "$LOGDIR"
LOG="$LOGDIR/eval_base_${MODE}.log"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

# All per-task task names for this dataset
TASKS=$(ls "$RANKALIGN_DIR/data/humaneval/${DATASET_DIR}/humaneval_"*.csv \
    | xargs -n1 basename | sed 's/\.csv$//' \
    | sed "s/^/${HE_TASK}-/" | tr '\n' ' ')
NT=$(echo "$TASKS" | wc -w)
log "mode=$MODE  TC_FLAG=$TC_FLAG  tasks=$NT  dataset=$HE_TASK"

# ---- Models to evaluate (same as v2.1correct-upper baseline) ---------------
MODELS=(
    "google/gemma-4-31B-it"
    "google/gemma-2-27b-it"
    "google/gemma-4-31B"
)

for MODEL in "${MODELS[@]}"; do
    MODEL_SLUG=$(echo "$MODEL" | sed 's|/|--|')
    log "======== $MODEL ========"

    # Download model if not cached
    CACHE_SIZE=$(du -sb "$HF_HUB_CACHE/models--${MODEL_SLUG}" 2>/dev/null | awk '{print $1}' || echo 0)
    if [ "${CACHE_SIZE:-0}" -lt 10000000000 ]; then
        log "  downloading $MODEL ..."
        python -c "
import os; from huggingface_hub import snapshot_download
snapshot_download(repo_id='$MODEL', cache_dir=os.environ['HF_HUB_CACHE'], max_workers=8)
" >> "$LOG" 2>&1
        CACHE_SIZE=$(du -sb "$HF_HUB_CACHE/models--${MODEL_SLUG}" 2>/dev/null | awk '{print $1}' || echo 0)
        if [ "${CACHE_SIZE:-0}" -lt 10000000000 ]; then
            log "  FATAL: $MODEL cache <10GB after download — auth or disk issue; skipping"
            continue
        fi
    else
        log "  already cached ($(( CACHE_SIZE/1024/1024/1024 ))GB)"
    fi

    # Eval all tasks (idempotent)
    n=0; skipped=0
    for T in $TASKS; do
        marker="${DONE}/${MODEL_SLUG}_${T}.done"
        if [ -f "$marker" ]; then skipped=$((skipped+1)); n=$((n+1)); continue; fi
        python eval_by_claude.py --model "$MODEL" --task "$T" --split_type random \
            --gen-shots zero --disc-shots zero $TC_FLAG \
            --validator-log-odds --save-scores-csv \
            --outputs-dir "$OUTDIR" \
            >> "$LOGDIR/eval_base_${MODE}_${MODEL_SLUG}.log" 2>&1 \
            && touch "$marker"
        n=$((n+1))
        [ $((n % 10)) -eq 0 ] && log "  $MODEL  $MODE: $n/$NT (skipped=$skipped)"
    done
    log "  $MODEL DONE: $n/$NT (skipped=$skipped)"

    # Delete model cache to free ~60GB for the next model
    log "  deleting $MODEL cache ..."
    rm -rf "$HF_HUB_CACHE/models--${MODEL_SLUG}"
    log "  disk after delete: $(df -h /workspace | tail -1)"

    log "======== $MODEL COMPLETE ========"
done

log "============ ALL MODELS DONE — mode=$MODE ============"
