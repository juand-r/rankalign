#!/bin/bash
# run_settings_v21correct_upper.sh <SETTING_A> [SETTING_B ...]
#
# Runs one or more numbered training settings sequentially on humaneval-v2.1correct-upper.
# Adapted from run_settings_v21correct_multi.sh for the correct-upper task.
#
# Usage on each pod (per TRAINING_PLAN_v21correct_upper.md):
#   nohup bash /workspace/rankalign/pod-setup-train-scripts-gemma-4/run_settings_v21correct_upper.sh 1 4 \
#       > /workspace/logs/train_s1_s4.log 2>&1 &
#
# Settings available in this script:
#   1=SFT-lo  4=New+fsx+tc  5=RankAlign+fsx+tc  7=New+fsx+negtc  8=RankAlign+fsx+negtc
#
# Previously done on correct-upper: s2 (RankAlign), s6 (RankAlign+tc), s9 (RankAlign+negtc).
#
# Idempotent: training skipped if all 3 epoch adapters exist;
# eval skipped per done-marker. Safe to re-run after a crash.
#
# Bug fixes vs old gemma-2 runs:
#   - s5 and s8: no --validator-log-odds during training (vlo belongs with comb, not pref-only).
set -uo pipefail

HE_TASK="humaneval-v2.1correct-upper"
DATASET_DIR="${HE_TASK#humaneval-}"   # v2.1correct-upper
MODEL="google/gemma-4-31B-it"
BASE="google/gemma-4-31B-it"

: "${HF_TOKEN:?HF_TOKEN not set — required for gated gemma-4 download}"

RANKALIGN_DIR="${RANKALIGN_DIR:-/workspace/rankalign}"
VENV_DIR="${VENV_DIR:-/workspace/.venv}"
MODELS_DIR="${MODELS_DIR:-/workspace/models_g4it}"
OUTDIR="${OUTDIR:-/workspace/outputs}"
LOGDIR="${LOGDIR:-/workspace/logs}"
EVAL_EPOCHS="${EVAL_EPOCHS:-2}"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "$VENV_DIR/bin/activate"
cd "$RANKALIGN_DIR/scripts"

DONE_DIR="$OUTDIR/.done"
mkdir -p "$MODELS_DIR" "$OUTDIR" "$DONE_DIR" "$LOGDIR"

TASKS=$(ls "$RANKALIGN_DIR/data/humaneval/${DATASET_DIR}/humaneval_"*.csv \
    | xargs -n1 basename | sed 's/\.csv$//' \
    | sed "s/^/${HE_TASK}-/" | tr '\n' ' ')
NT=$(echo "$TASKS" | wc -w)

BT_ARGS="--base-typicality --base-model $BASE"

# Globals populated by configure_setting()
SETTING_NAME=""
ADAPTER_SUFFIX=""
EVAL_MODES=""
TRAIN_FLAGS=""

# Adapter suffix values are derived from ranking_loss_ref_fix.py (updated 2026-05-22;
# previously referenced ranking_loss_ref_gemma4.py line ~2771).
# (verified against TRAINING_PLAN_v21correct_upper.md and the correct-multi run).
configure_setting() {
    local S="$1"
    local pref_w=1 nll_v_w=0 nll_g_w=0
    local semi_flag="--semi-supervised 0.1"
    local fsx="" vlo="" tc_train=""

    case "$S" in
        1)
            SETTING_NAME="SFT-lo"
            pref_w=0; nll_v_w=1; nll_g_w=1; semi_flag="--labeled-only 0.1"
            ADAPTER_SUFFIX="-all--d2g--random--alpha1.0--full-completion--pref0.0--nllv1.0--nllg1.0--labelonly0.1"
            EVAL_MODES="--self-typicality --neg-typicality"
            ;;
        4)
            SETTING_NAME="New+fsx+tc"
            nll_v_w=1; nll_g_w=1; fsx="--force-same-x"; vlo="--validator-log-odds"; tc_train="--self-typicality"
            ADAPTER_SUFFIX="-all--d2g--random--alpha1.0--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1"
            EVAL_MODES="--self-typicality"
            ;;
        5)
            SETTING_NAME="RankAlign+fsx+tc"
            # Note: no --validator-log-odds here (bug fix vs old gemma-2 runs)
            fsx="--force-same-x"; tc_train="--self-typicality"
            ADAPTER_SUFFIX="-all--d2g--random--alpha1.0--tc-self--full-completion--force-same-x--semi0.1"
            EVAL_MODES="--self-typicality"
            ;;
        7)
            SETTING_NAME="New+fsx+negtc"
            nll_v_w=1; nll_g_w=1; fsx="--force-same-x"; vlo="--validator-log-odds"; tc_train="--neg-typicality"
            ADAPTER_SUFFIX="-all--d2g--random--alpha1.0--tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1"
            EVAL_MODES="--neg-typicality"
            ;;
        8)
            SETTING_NAME="RankAlign+fsx+negtc"
            # Note: no --validator-log-odds here (bug fix vs old gemma-2 runs)
            fsx="--force-same-x"; tc_train="--neg-typicality"
            ADAPTER_SUFFIX="-all--d2g--random--alpha1.0--tc-neg--full-completion--force-same-x--semi0.1"
            EVAL_MODES="--neg-typicality"
            ;;
        *)
            echo "FATAL: unknown setting '$S' — this script handles settings 1, 4, 5, 7, 8 on correct-upper"; exit 2 ;;
    esac

    TRAIN_FLAGS="--model $MODEL --num_epochs 3 --task $HE_TASK \
--train_g_or_d g --split_type random \
--nll_validator_weight $nll_v_w --nll_generator_weight $nll_g_w \
--preference_loss_weight $pref_w \
--all --delta 0.15 $semi_flag --disc-shots zero \
--lora --gradient_checkpointing \
--models-dir $MODELS_DIR --total_samples 5110 --no-upload-hf"
    [ -n "$fsx" ]      && TRAIN_FLAGS="$TRAIN_FLAGS $fsx"
    [ -n "$vlo" ]      && TRAIN_FLAGS="$TRAIN_FLAGS $vlo"
    [ -n "$tc_train" ] && TRAIN_FLAGS="$TRAIN_FLAGS $tc_train"
}

adapter_dir_for() {
    echo "${MODELS_DIR}/v6-google--gemma-4-31B-it-delta0.15-epoch${1}--${HE_TASK}${ADAPTER_SUFFIX}"
}

run_setting() {
    local S="$1"
    configure_setting "$S"

    local LOG="$LOGDIR/run_s${S}_${SETTING_NAME}.log"
    local TRAIN_LOG="$LOGDIR/train_s${S}.log"

    echo "[$(date -u +%H:%M:%S)] ======== SETTING $S: $SETTING_NAME ========" | tee -a "$LOG"
    echo "[$(date -u +%H:%M:%S)] tasks=$NT  EVAL_MODES='$EVAL_MODES'" | tee -a "$LOG"
    echo "[$(date -u +%H:%M:%S)] ADAPTER_SUFFIX=$ADAPTER_SUFFIX" | tee -a "$LOG"
    echo "[$(date -u +%H:%M:%S)] TRAIN_FLAGS=$TRAIN_FLAGS" | tee -a "$LOG"

    local ad0; ad0=$(adapter_dir_for 0)
    local ad1; ad1=$(adapter_dir_for 1)
    local ad2; ad2=$(adapter_dir_for 2)

    # STEP 1: train (skip if all 3 epoch adapters already present)
    if [ -d "$ad0" ] && [ -d "$ad1" ] && [ -d "$ad2" ]; then
        echo "[$(date -u +%H:%M:%S)] training: all 3 epoch adapters present, skipping" | tee -a "$LOG"
    else
        echo "[$(date -u +%H:%M:%S)] training: starting (log: $TRAIN_LOG)" | tee -a "$LOG"
        # shellcheck disable=SC2086
        # Updated 2026-05-22: use ranking_loss_ref_fix.py (bug fixes; handles gemma-4/gemma-2/qwen-3)
        # with --delta-bins 10 (new default for fix1/log-odds runs) and --gemma4-lora (required for gemma-4 LoRA).
        # Old: python ranking_loss_ref_gemma4.py $TRAIN_FLAGS
        python ranking_loss_ref_fix.py $TRAIN_FLAGS --delta-bins 10 --gemma4-lora \
            > "$TRAIN_LOG" 2>&1
        local rc=$?
        echo "[$(date -u +%H:%M:%S)] training exit=$rc" | tee -a "$LOG"
        if [ ! -d "$ad2" ]; then
            echo "[$(date -u +%H:%M:%S)] FATAL: epoch2 adapter missing after training (setting $S). See $TRAIN_LOG" | tee -a "$LOG"
            exit 1
        fi
        echo "[$(date -u +%H:%M:%S)] training DONE (epoch2 adapter confirmed)" | tee -a "$LOG"
    fi

    # STEP 2: eval (idempotent per done-marker, one call per task per mode)
    local EP
    for EP in $EVAL_EPOCHS; do
        local ad; ad=$(adapter_dir_for "$EP")
        if [ ! -d "$ad" ]; then
            echo "[$(date -u +%H:%M:%S)] WARN: epoch$EP adapter missing, skipping eval" | tee -a "$LOG"
            continue
        fi
        local MODE
        for MODE in $EVAL_MODES; do
            local mname="${MODE#--}"
            local EVAL_LOG="$LOGDIR/eval_s${S}_ep${EP}_${mname}.log"
            echo "[$(date -u +%H:%M:%S)] ==== eval s${S} epoch${EP} $MODE ====" | tee -a "$LOG"
            local n=0
            local T
            for T in $TASKS; do
                local marker="${DONE_DIR}/s${S}_ep${EP}_${mname}_bt1_${T}.done"
                if [ -f "$marker" ]; then n=$((n+1)); continue; fi
                python eval_by_claude.py --model "$ad" --task "$T" --split_type random \
                    --gen-shots zero --disc-shots zero "$MODE" $BT_ARGS \
                    --validator-log-odds --save-scores-csv \
                    --outputs-dir "$OUTDIR" \
                    >> "$EVAL_LOG" 2>&1 \
                    && touch "$marker"
                n=$((n+1))
                [ $((n % 10)) -eq 0 ] && echo "[$(date -u +%H:%M:%S)]   s${S} $MODE ep${EP}: $n/$NT" | tee -a "$LOG"
            done
            echo "[$(date -u +%H:%M:%S)] eval s${S} ep${EP} $MODE done ($n/$NT)" | tee -a "$LOG"
        done
        echo "[$(date -u +%H:%M:%S)] SETTING${S}_EPOCH${EP}_EVAL_COMPLETE" | tee -a "$LOG"
    done

    echo "[$(date -u +%H:%M:%S)] ======== SETTING $S ($SETTING_NAME) COMPLETE ========" | tee -a "$LOG"
    echo "SETTING${S}_COMPLETE" >> "$LOG"
}

# ---- main -------------------------------------------------------------------
if [ $# -eq 0 ]; then
    echo "usage: run_settings_v21correct_upper.sh <SETTING_1> [SETTING_2 ...]"
    echo "settings: 1=SFT-lo  4=New+fsx+tc  5=RankAlign+fsx+tc  7=New+fsx+negtc  8=RankAlign+fsx+negtc"
    exit 1
fi

echo "[$(date -u +%H:%M:%S)] Starting settings: $*  (tasks=$NT)"
for S in "$@"; do
    run_setting "$S"
done
echo "[$(date -u +%H:%M:%S)] ALL SETTINGS COMPLETE: $*"
