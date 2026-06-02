#!/bin/bash
# Recovery pipeline v2: eval no_tc using LoRA adapter directly (PEFT, no merge),
# then train+eval self_tc and neg_tc (1 epoch each, LoRA adapter passed to eval).
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
mkdir -p "$MODELS_DIR" "$OUTDIR"
LOG=/workspace/logs/recovery_v2.log
> "$LOG"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

TASKS=$(ls /workspace/rankalign/data/humaneval/v2.1correct-upper/humaneval_*.csv \
    | xargs -n1 basename | sed 's/\.csv$//' | sed 's/^/humaneval-v2.1correct-upper-/' \
    | tr '\n' ' ')
NTASKS=$(echo "$TASKS" | wc -w)
log "Tasks: $NTASKS"

EPOCH0_ADAPTER="${MODELS_DIR}/v6-google--gemma-4-31B-it-delta0.15-epoch0--humaneval-v2.1correct-upper-all--d2g--random--alpha1.0--full-completion--semi0.1"

if [ ! -d "$EPOCH0_ADAPTER" ]; then
    log "FATAL: no_tc epoch0 adapter missing: $EPOCH0_ADAPTER"
    exit 1
fi
log "no_tc adapter: $EPOCH0_ADAPTER"

# ===== STEP 1: Eval no_tc with PEFT adapter (no merge needed) =====
for MODE in --self-typicality --neg-typicality; do
    MNAME="${MODE#--}"
    log "  ==== no_tc eval $MODE ===="
    DONE=0
    for T in $TASKS; do
        python eval_by_claude.py --model "$EPOCH0_ADAPTER" --task "$T" --split_type random \
            --gen-shots zero --disc-shots zero "$MODE" --base-typicality --base-model "$BASE" \
            --validator-log-odds --save-scores-csv --outputs-dir "$OUTDIR" \
            >> "/workspace/logs/eval_notc_${MNAME}.log" 2>&1
        DONE=$((DONE+1))
    done
    log "  no_tc eval $MODE done ($DONE tasks)"
done
NSCORES=$(ls "${OUTDIR}"/scores_*v2.1correct-upper* 2>/dev/null | wc -l)
log "  no_tc total score files so far: $NSCORES"
log "VARIANT no_tc COMPLETE"
echo "VARIANT_no_tc_COMPLETE" >> "$LOG"
df -h /workspace | tail -1 | tee -a "$LOG"

# ===== STEP 2+3: Train self_tc and neg_tc (1 epoch), eval with PEFT adapter =====
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
    # Find the saved LoRA adapter (NOT merged — no merge needed)
    local ADAPTER
    ADAPTER=$(ls -d "${MODELS_DIR}"/v6-google--gemma-4-31B-it-*humaneval-v2.1correct-upper-all* 2>/dev/null \
        | grep -v '_merged' | tail -1)
    if [ "$RC" -ne 0 ] || [ -z "$ADAPTER" ] || [ ! -d "$ADAPTER" ]; then
        log "  FATAL: $VNAME training failed or no adapter. See real_train_${VNAME}.log"
        return 1
    fi
    log "  $VNAME adapter: $ADAPTER"
    log "  ==== VARIANT $VNAME : eval $EVAL_MODE ===="
    DONE=0
    for T in $TASKS; do
        python eval_by_claude.py --model "$ADAPTER" --task "$T" --split_type random \
            --gen-shots zero --disc-shots zero "$EVAL_MODE" --base-typicality --base-model "$BASE" \
            --validator-log-odds --save-scores-csv --outputs-dir "$OUTDIR" \
            >> "/workspace/logs/eval_${VNAME}.log" 2>&1
        DONE=$((DONE+1))
    done
    log "  $VNAME eval $EVAL_MODE done ($DONE tasks)"
    local NSCORES
    NSCORES=$(ls "${OUTDIR}"/scores_*v2.1correct-upper* 2>/dev/null | wc -l)
    log "  $VNAME total score files: $NSCORES"
    df -h /workspace | tail -1 | tee -a "$LOG"
    log "==== VARIANT $VNAME DONE ===="
    echo "VARIANT_${VNAME}_COMPLETE" >> "$LOG"
}

run_variant "self_tc" "--self-typicality" "--self-typicality" || log "self_tc FAILED"
run_variant "neg_tc" "--neg-typicality" "--neg-typicality" || log "neg_tc FAILED"

log "============ PIPELINE COMPLETE ============"
echo "PIPELINE_COMPLETE" >> "$LOG"
