#!/bin/bash
# Sequential model pipeline on h100-train-4 with 3 H100 NVL GPUs.
# For each of 3 models: download → train 3 variants in parallel → eval 4 variants → delete model.
#
# Disk peak per model: ~60GB model + ~16GB venv + ~1GB checkpoints ≈ 80GB (fits in 100GB quota).

set -uo pipefail

export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN not set — was hardcoded on the pod; redacted before commit}"
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /workspace/.venv/bin/activate
cd /workspace/rankalign/scripts

PIPELINE_LOG=/workspace/logs/sequential_pipeline.log
> $PIPELINE_LOG
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $PIPELINE_LOG; }

MODELS="google/gemma-4-31B-it google/gemma-2-27b-it google/gemma-4-31B"
TASKS=$(ls /workspace/rankalign/data/humaneval/v2.1correct-upper/humaneval_*.csv | xargs -n1 basename | sed 's/\.csv$//' | sed 's/^/humaneval-v2.1correct-upper-/' | tr '\n' ' ')

for MODEL in $MODELS; do
    MODEL_SLUG=$(echo $MODEL | sed 's|/|--|')
    log "================ PROCESSING $MODEL ================"

    # Phase A: download
    log "  [A] downloading $MODEL"
    python -c "
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$MODEL', cache_dir=os.environ['HF_HUB_CACHE'], token=os.environ['HF_TOKEN'], max_workers=8)
" >> $PIPELINE_LOG 2>&1
    # Verify download size (gated repos silently produce only README without auth)
    CACHE_SIZE=$(du -sb /workspace/.cache/huggingface/hub/models--${MODEL_SLUG} 2>/dev/null | awk '{print $1}')
    log "  [A] cache size: $((CACHE_SIZE/1024/1024/1024))GB"
    if [ "${CACHE_SIZE:-0}" -lt 10000000000 ]; then
        log "  [A] FATAL: cache <10GB — auth or download failed. Skipping $MODEL."
        continue
    fi

    # Phase B: train 3 variants in parallel on 3 GPUs
    log "  [B] launching 3 training variants in parallel"
    for IDX_VAR in "0:no-tc:" "1:self-tc:--self-typcorr" "2:neg-tc:--neg-typcorr"; do
        GPU=${IDX_VAR%%:*}
        REST=${IDX_VAR#*:}
        VARNAME=${REST%%:*}
        TC_FLAG=${REST#*:}
        log "    [B.$VARNAME] GPU=$GPU TC_FLAG='$TC_FLAG'"
        CUDA_VISIBLE_DEVICES=$GPU nohup bash run_train_semi.sh "$MODEL" humaneval-v2.1correct-upper pref-only semi 0.1 \
            --disc-shots zero $TC_FLAG --no-force-same-x --models-dir /workspace/models \
            > /workspace/logs/train_${MODEL_SLUG}_${VARNAME}.log 2>&1 &
    done
    wait
    log "  [B] all 3 training variants done for $MODEL"

    # Phase C: eval 4 variants on 3 GPUs
    log "  [C] launching trained-model evals"
    # Find the checkpoint dirs
    CKPT_BASE=$(ls -d /workspace/models/v6-${MODEL_SLUG}--*humaneval-v2.1correct-upper-all*semi0.1 2>/dev/null | head -1)
    CKPT_SELF=$(ls -d /workspace/models/v6-${MODEL_SLUG}--*humaneval-v2.1correct-upper-all*tc-self*semi0.1 2>/dev/null | head -1)
    CKPT_NEG=$(ls -d /workspace/models/v6-${MODEL_SLUG}--*humaneval-v2.1correct-upper-all*tc-neg*semi0.1 2>/dev/null | head -1)
    # Prefer merged versions if they exist
    for V in CKPT_BASE CKPT_SELF CKPT_NEG; do
        eval "ckpt=\$$V"
        if [ -n "$ckpt" ] && [ -d "${ckpt}_merged" ]; then
            eval "$V=${ckpt}_merged"
        fi
    done
    log "    [C] ckpt no-tc: $CKPT_BASE"
    log "    [C] ckpt self-tc: $CKPT_SELF"
    log "    [C] ckpt neg-tc: $CKPT_NEG"

    # Run 4 evals: var2 needs self AND neg; var6 needs self only; var6' needs neg only.
    # GPU 0: var2 self → var2 neg (sequential, 2 evals)
    # GPU 1: var6 self (1 eval)
    # GPU 2: var6' neg (1 eval)
    (
        export CUDA_VISIBLE_DEVICES=0
        for T in $TASKS; do
            python eval_by_claude.py --model "$CKPT_BASE" --task "$T" --split_type random \
                --gen-shots zero --disc-shots zero --self-typicality --base-typicality --base-model "$MODEL" \
                --validator-log-odds --save-scores-csv --outputs-dir /workspace/outputs 2>&1 | tail -2
        done
        for T in $TASKS; do
            python eval_by_claude.py --model "$CKPT_BASE" --task "$T" --split_type random \
                --gen-shots zero --disc-shots zero --neg-typicality --base-typicality --base-model "$MODEL" \
                --validator-log-odds --save-scores-csv --outputs-dir /workspace/outputs 2>&1 | tail -2
        done
    ) > /workspace/logs/eval_${MODEL_SLUG}_var2_gpu0.log 2>&1 &
    (
        export CUDA_VISIBLE_DEVICES=1
        for T in $TASKS; do
            python eval_by_claude.py --model "$CKPT_SELF" --task "$T" --split_type random \
                --gen-shots zero --disc-shots zero --self-typicality --base-typicality --base-model "$MODEL" \
                --validator-log-odds --save-scores-csv --outputs-dir /workspace/outputs 2>&1 | tail -2
        done
    ) > /workspace/logs/eval_${MODEL_SLUG}_var6_gpu1.log 2>&1 &
    (
        export CUDA_VISIBLE_DEVICES=2
        for T in $TASKS; do
            python eval_by_claude.py --model "$CKPT_NEG" --task "$T" --split_type random \
                --gen-shots zero --disc-shots zero --neg-typicality --base-typicality --base-model "$MODEL" \
                --validator-log-odds --save-scores-csv --outputs-dir /workspace/outputs 2>&1 | tail -2
        done
    ) > /workspace/logs/eval_${MODEL_SLUG}_var6prime_gpu2.log 2>&1 &
    wait
    log "  [C] all 4 trained-model evals done for $MODEL"

    # Phase D: delete model from cache to free ~60GB for next model. KEEP checkpoints + outputs.
    log "  [D] deleting $MODEL from cache"
    rm -rf /workspace/.cache/huggingface/hub/models--${MODEL_SLUG}
    DF_AFTER=$(df -h /workspace | tail -1)
    log "  [D] disk after delete: $DF_AFTER"
    log "================ DONE $MODEL ================"
done
log "============ ALL 3 MODELS DONE ============"
