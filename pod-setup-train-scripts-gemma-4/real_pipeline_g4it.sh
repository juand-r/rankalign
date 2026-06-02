#!/bin/bash
# Real training+eval pipeline for gemma-4-31B-it ONLY, per-variant cycle to fit 100GB disk.
# For each variant: train (3-GPU shard + grad ckpt) -> verify merged ckpt -> eval (base-typicality)
# -> mark scores -> delete merged ckpt (keep tiny adapter) -> next variant.
set -uo pipefail
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN not set — was hardcoded on the pod; redacted before commit}"
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
# WandB: log ONLINE so training curves sync to the cloud (juand-r/rankalign).
# (Was `export WANDB_MODE=offline` — that stranded gemma-4 curves on the pod, which were
#  lost when the pod stopped. 2026-06 lesson.) Requires WANDB_API_KEY in the env; falls back
#  to offline with a LOUD warning if it is missing (so it never silently strands curves).
if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=online
else
    echo "[WARN] WANDB_API_KEY not set -> wandb OFFLINE; curves will NOT sync. Export WANDB_API_KEY to fix." >&2
    export WANDB_MODE=offline
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /workspace/.venv/bin/activate
cd /workspace/rankalign/scripts

MODEL=google/gemma-4-31B-it
BASE=google/gemma-4-31B-it
MODELS_DIR=/workspace/models_g4it
OUTDIR=/workspace/outputs
mkdir -p $MODELS_DIR $OUTDIR
LOG=/workspace/logs/real_pipeline_g4it.log
> $LOG
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
TASKS=$(ls /workspace/rankalign/data/humaneval/v2.1correct-upper/humaneval_*.csv | xargs -n1 basename | sed 's/\.csv$//' | sed 's/^/humaneval-v2.1correct-upper-/' | tr '\n' ' ')
COMMON="--model $MODEL --num_epochs 3 --task humaneval-v2.1correct-upper --train_g_or_d g --split_type random --nll_validator_weight 0 --nll_generator_weight 0 --preference_loss_weight 1 --all --delta 0.15 --semi-supervised 0.1 --disc-shots zero --lora --gradient_checkpointing --models-dir $MODELS_DIR --total_samples 5110"

run_variant () {
    local VNAME="$1"; local TC_TRAIN="$2"; local EVAL_MODES="$3"
    log "==== VARIANT $VNAME : train ===="
    python ranking_loss_ref_gemma4.py $COMMON $TC_TRAIN > /workspace/logs/real_train_${VNAME}.log 2>&1
    local RC=$?
    log "  $VNAME training exit=$RC"
    # Find merged checkpoint
    local CKPT=$(ls -d ${MODELS_DIR}/v6-google--gemma-4-31B-it-*humaneval-v2.1correct-upper-all*_merged 2>/dev/null | tail -1)
    if [ $RC -ne 0 ] || [ -z "$CKPT" ] || [ ! -d "$CKPT" ]; then
        log "  FATAL: $VNAME training failed or no merged ckpt. Skipping eval. See real_train_${VNAME}.log"
        return 1
    fi
    log "  $VNAME merged ckpt: $CKPT"
    # Eval (sequential modes; each uses 3-GPU shard for trained+base)
    for MODE in $EVAL_MODES; do
        log "  ==== VARIANT $VNAME : eval $MODE ===="
        local done=0
        for T in $TASKS; do
            python eval_by_claude.py --model "$CKPT" --task "$T" --split_type random \
                --gen-shots zero --disc-shots zero $MODE --base-typicality --base-model "$BASE" \
                --validator-log-odds --save-scores-csv --outputs-dir $OUTDIR > /dev/null 2>&1
            done=$((done+1))
        done
        log "  $VNAME eval $MODE done ($done tasks)"
    done
    # Mark scores ready; keep adapter, delete merged to free ~21GB
    local NSCORES=$(ls ${OUTDIR}/scores_*v2.1correct-upper* 2>/dev/null | wc -l)
    log "  $VNAME total score files now: $NSCORES"
    log "  deleting merged ckpt $CKPT to free disk"
    rm -rf "$CKPT"
    df -h /workspace | tail -1 | sed "s/^/[disk] /" | tee -a $LOG
    log "==== VARIANT $VNAME DONE ===="
    echo "VARIANT_${VNAME}_COMPLETE" >> $LOG
}

# var2 (no TC): eval with both self and neg
run_variant "no_tc" "" "--self-typicality --neg-typicality" || true
# var6 (self-TC): eval self only
run_variant "self_tc" "--self-typicality" "--self-typicality" || true
# var6' (neg-TC): eval neg only
run_variant "neg_tc" "--neg-typicality" "--neg-typicality" || true

log "============ PIPELINE COMPLETE ============"
