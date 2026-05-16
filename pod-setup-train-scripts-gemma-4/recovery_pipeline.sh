#!/bin/bash
# Recovery pipeline: merge no_tc epoch0, eval no_tc, then train+eval self_tc and neg_tc (1 epoch each).
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

MODEL=google/gemma-4-31B-it
BASE=google/gemma-4-31B-it
MODELS_DIR=/workspace/models_g4it
OUTDIR=/workspace/outputs
mkdir -p "$MODELS_DIR" "$OUTDIR"
LOG=/workspace/logs/recovery_pipeline.log
> "$LOG"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

TASKS=$(ls /workspace/rankalign/data/humaneval/v2.1correct-upper/humaneval_*.csv \
    | xargs -n1 basename | sed 's/\.csv$//' | sed 's/^/humaneval-v2.1correct-upper-/' \
    | tr '\n' ' ')
NTASKS=$(echo "$TASKS" | wc -w)
log "Tasks: $NTASKS"

EPOCH0_ADAPTER='v6-google--gemma-4-31B-it-delta0.15-epoch0--humaneval-v2.1correct-upper-all--d2g--random--alpha1.0--full-completion--semi0.1'
MERGED_NO_TC="${MODELS_DIR}/${EPOCH0_ADAPTER}_merged"

# ===== STEP 1: Merge no_tc epoch0 =====
log "==== STEP 1: Merge no_tc epoch0 ===="
if [ -d "$MERGED_NO_TC" ]; then
    log "  Merged already exists, skipping merge."
else
    python /workspace/merge_no_tc.py >> "$LOG" 2>&1
    RC=$?
    if [ "$RC" -ne 0 ] || [ ! -d "$MERGED_NO_TC" ]; then
        log "  FATAL: merge failed (RC=$RC). Check log."
        exit 1
    fi
    log "  Merge done: $MERGED_NO_TC"
fi

# ===== STEP 2: Eval no_tc (both self and neg TC) =====
for MODE in --self-typicality --neg-typicality; do
    MNAME="${MODE#--}"
    log "  ==== no_tc eval $MODE ===="
    DONE=0
    for T in $TASKS; do
        python eval_by_claude.py --model "$MERGED_NO_TC" --task "$T" --split_type random \
            --gen-shots zero --disc-shots zero "$MODE" --base-typicality --base-model "$BASE" \
            --validator-log-odds --save-scores-csv --outputs-dir "$OUTDIR" \
            >> "/workspace/logs/eval_notc_${MNAME}.log" 2>&1
        DONE=$((DONE+1))
    done
    log "  no_tc eval $MODE done ($DONE tasks)"
done
NSCORES=$(ls "${OUTDIR}"/scores_*v2.1correct-upper* 2>/dev/null | wc -l)
log "  no_tc total score files: $NSCORES"
log "  Deleting merged to free disk: $MERGED_NO_TC"
rm -rf "$MERGED_NO_TC"
log "VARIANT no_tc COMPLETE"
echo "VARIANT_no_tc_COMPLETE" >> "$LOG"
df -h /workspace | tail -1 | tee -a "$LOG"

# ===== Helper: run_variant for remaining variants =====
COMMON="--model $MODEL --num_epochs 1 --task humaneval-v2.1correct-upper --train_g_or_d g --split_type random --nll_validator_weight 0 --nll_generator_weight 0 --preference_loss_weight 1 --all --delta 0.15 --semi-supervised 0.1 --disc-shots zero --lora --gradient_checkpointing --models-dir $MODELS_DIR --total_samples 5110"

run_variant() {
    local VNAME="$1"
    local TC_TRAIN="$2"
    local EVAL_MODE="$3"
    log "==== VARIANT $VNAME : train ===="
    # shellcheck disable=SC2086
    python ranking_loss_ref_gemma4.py $COMMON $TC_TRAIN \
        > "/workspace/logs/real_train_${VNAME}.log" 2>&1
    local RC=$?
    log "  $VNAME training exit=$RC"
    local CKPT
    CKPT=$(ls -d "${MODELS_DIR}"/v6-google--gemma-4-31B-it-*humaneval-v2.1correct-upper-all*_merged 2>/dev/null | tail -1)
    if [ "$RC" -ne 0 ] || [ -z "$CKPT" ] || [ ! -d "$CKPT" ]; then
        log "  FATAL: $VNAME training/merge failed. See real_train_${VNAME}.log"
        return 1
    fi
    log "  $VNAME merged ckpt: $CKPT"
    log "  ==== VARIANT $VNAME : eval $EVAL_MODE ===="
    DONE=0
    for T in $TASKS; do
        python eval_by_claude.py --model "$CKPT" --task "$T" --split_type random \
            --gen-shots zero --disc-shots zero "$EVAL_MODE" --base-typicality --base-model "$BASE" \
            --validator-log-odds --save-scores-csv --outputs-dir "$OUTDIR" \
            >> "/workspace/logs/eval_${VNAME}.log" 2>&1
        DONE=$((DONE+1))
    done
    log "  $VNAME eval $EVAL_MODE done ($DONE tasks)"
    local NSCORES
    NSCORES=$(ls "${OUTDIR}"/scores_*v2.1correct-upper* 2>/dev/null | wc -l)
    log "  $VNAME total score files: $NSCORES"
    log "  Deleting merged: $CKPT"
    rm -rf "$CKPT"
    df -h /workspace | tail -1 | tee -a "$LOG"
    log "==== VARIANT $VNAME DONE ===="
    echo "VARIANT_${VNAME}_COMPLETE" >> "$LOG"
}

run_variant "self_tc" "--self-typicality" "--self-typicality" || log "self_tc FAILED"
run_variant "neg_tc" "--neg-typicality" "--neg-typicality" || log "neg_tc FAILED"

log "============ PIPELINE COMPLETE ============"
echo "PIPELINE_COMPLETE" >> "$LOG"
