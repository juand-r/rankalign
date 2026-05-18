#!/bin/bash
# Reproducible single-arm 3-epoch train+eval for gemma-4-31B-it.
# ONE committed entry point — no ad-hoc relaunch.
#
# ---- Dataset (change only this line to switch datasets) --------------------
HE_TASK="humaneval-v2.1correct-multi"
# HE_TASK="humaneval-v2.1correct-upper"   # previous dataset
# ---------------------------------------------------------------------------
#
#   bash run_arm_3epoch.sh <no_tc|self_tc|neg_tc>
#
# Runs ONE arm end-to-end on its own pod:
#   1. train google/gemma-4-31B-it, --num_epochs 3 (saves epoch0/1/2 adapters;
#      save_steps=1 so every epoch is checkpointed; merge disabled by patch 4)
#   2. eval each epoch checkpoint with the matching TC recipe
#      (--base-typicality --base-model google/gemma-4-31B-it, identical to the
#       verified recovery_pipeline_v2.sh STEP 2+3)
#
# Idempotent: training skipped if all 3 epoch adapters exist; each
# (epoch,mode,task) eval skipped if its done-marker exists. Safe to re-run
# after a crash — picks up where it stopped. (Training itself is NOT
# resumable mid-run; a training crash re-trains this arm from scratch, but
# only this arm — run one arm per pod for fault isolation.)
#
# HF_TOKEN is read from the environment and must be set (gated gemma-4
# download). It is NEVER hardcoded here.
set -uo pipefail

DATASET_DIR="${HE_TASK#humaneval-}"   # e.g. v2.1correct-multi
ARM="${1:?usage: run_arm_3epoch.sh <no_tc|self_tc|neg_tc>}"
case "$ARM" in
  no_tc)   TC_TRAIN="";                 EVAL_MODES="--self-typicality --neg-typicality" ;;
  self_tc) TC_TRAIN="--self-typicality"; EVAL_MODES="--self-typicality" ;;
  neg_tc)  TC_TRAIN="--neg-typicality";  EVAL_MODES="--neg-typicality" ;;
  *) echo "FATAL: unknown arm '$ARM' (no_tc|self_tc|neg_tc)"; exit 2 ;;
esac

# Which epoch checkpoints to eval. Default = ONLY the final epoch of a
# 3-epoch run (epoch2, 0-indexed) — that is the lab-standard comparison
# point. Override e.g. EVAL_EPOCHS="0 1 2" to also eval intermediate epochs.
EVAL_EPOCHS="${EVAL_EPOCHS:-2}"

# BASETYP=1 (default): typicality reference = frozen base model
#   (--base-typicality --base-model $BASE); correct for offline-trained
#   models. BASETYP=0: omit it, so --self/--neg-typicality use the
#   evaluated model's OWN log P(y) (filename prefix becomes self-/neg-
#   instead of basetyp-/basetypneg-). Marker/log names include bt<N> so a
#   BASETYP=0 re-run does not collide with / skip the BASETYP=1 results.
BASETYP="${BASETYP:-1}"

: "${HF_TOKEN:?HF_TOKEN not set in environment — required for gated gemma-4 download}"

# Paths are env-overridable; defaults are the RunPod pod layout, so existing
# pod usage is unchanged. The mll Slurm wrapper sets these to mll paths.
RANKALIGN_DIR="${RANKALIGN_DIR:-/workspace/rankalign}"
VENV_DIR="${VENV_DIR:-/workspace/.venv}"
MODELS_DIR="${MODELS_DIR:-/workspace/models_g4it}"
OUTDIR="${OUTDIR:-/workspace/outputs}"
LOGDIR="${LOGDIR:-/workspace/logs}"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$VENV_DIR/bin/activate"
cd "$RANKALIGN_DIR/scripts"

MODEL=google/gemma-4-31B-it
BASE=google/gemma-4-31B-it
DONE="$OUTDIR/.done"
mkdir -p "$MODELS_DIR" "$OUTDIR" "$DONE" "$LOGDIR"
LOG="$LOGDIR/run_${ARM}_3epoch.log"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

# Verified COMMON (= real_pipeline_g4it.sh, num_epochs 3). Do not edit casually.
COMMON="--model $MODEL --num_epochs 3 --task $HE_TASK \
--train_g_or_d g --split_type random --nll_validator_weight 0 \
--nll_generator_weight 0 --preference_loss_weight 1 --all --delta 0.15 \
--semi-supervised 0.1 --disc-shots zero --lora --gradient_checkpointing \
--models-dir $MODELS_DIR --total_samples 5110"

TASKS=$(ls "$RANKALIGN_DIR"/data/humaneval/${DATASET_DIR}/humaneval_*.csv \
    | xargs -n1 basename | sed 's/\.csv$//' \
    | sed "s/^/${HE_TASK}-/" | tr '\n' ' ')
NT=$(echo "$TASKS" | wc -w)

adapter_dir(){ # $1 = epoch
  local infix=""
  [ "$ARM" = self_tc ] && infix="tc-self--"
  [ "$ARM" = neg_tc ]  && infix="tc-neg--"
  echo "${MODELS_DIR}/v6-google--gemma-4-31B-it-delta0.15-epoch$1--${HE_TASK}-all--d2g--random--alpha1.0--${infix}full-completion--semi0.1"
}

log "HE_TASK=$HE_TASK  ARM=$ARM  tasks=$NT  TC_TRAIN='${TC_TRAIN:-<none>}'  EVAL_MODES='$EVAL_MODES'"

# ---- STEP 1: train (skip only if all 3 epoch adapters already present) ----
if [ -d "$(adapter_dir 0)" ] && [ -d "$(adapter_dir 1)" ] && [ -d "$(adapter_dir 2)" ]; then
    log "training: all 3 epoch adapters present, skipping train"
else
    log "training: launching ranking_loss_ref_gemma4.py (3 epochs)"
    # shellcheck disable=SC2086
    python ranking_loss_ref_gemma4.py $COMMON $TC_TRAIN \
        > "$LOGDIR/train_${ARM}_3epoch.log" 2>&1
    rc=$?
    log "training exit=$rc"
    if [ ! -d "$(adapter_dir 2)" ]; then
        log "FATAL: epoch2 adapter missing after training. See $LOGDIR/train_${ARM}_3epoch.log"
        exit 1
    fi
fi

# ---- STEP 2: eval the requested epoch checkpoint(s) (idempotent markers) ----
log "eval epochs: $EVAL_EPOCHS  BASETYP=$BASETYP"
if [ "$BASETYP" = "1" ]; then BT_ARGS="--base-typicality --base-model $BASE"; else BT_ARGS=""; fi
for EP in $EVAL_EPOCHS; do
    AD="$(adapter_dir "$EP")"
    if [ ! -d "$AD" ]; then log "WARN: epoch$EP adapter missing, skipping its eval"; continue; fi
    for MODE in $EVAL_MODES; do
        mname="${MODE#--}"
        log "  ==== eval $ARM epoch$EP $MODE ===="
        n=0
        for T in $TASKS; do
            marker="${DONE}/${ARM}_ep${EP}_${mname}_bt${BASETYP}_${T}.done"
            if [ -f "$marker" ]; then n=$((n+1)); continue; fi
            python eval_by_claude.py --model "$AD" --task "$T" --split_type random \
                --gen-shots zero --disc-shots zero "$MODE" $BT_ARGS \
                --validator-log-odds --save-scores-csv \
                --outputs-dir "$OUTDIR" \
                >> "$LOGDIR/eval_${ARM}_ep${EP}_${mname}_bt${BASETYP}.log" 2>&1 \
                && touch "$marker"
            n=$((n+1))
            [ $((n % 10)) -eq 0 ] && log "    $MODE epoch$EP: $n/$NT"
        done
        log "  eval $ARM epoch$EP $MODE done ($n/$NT)"
    done
    log "EPOCH${EP}_EVAL_COMPLETE ($ARM)"
    echo "ARM_${ARM}_EPOCH${EP}_COMPLETE" >> "$LOG"
done

log "============ ARM $ARM COMPLETE ============"
echo "ARM_${ARM}_COMPLETE" >> "$LOG"
