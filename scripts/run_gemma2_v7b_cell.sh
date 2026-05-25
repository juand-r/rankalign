#!/bin/bash
# run_gemma2_v7b_cell.sh DATASET SETTING
#
# v7b variant: same as run_gemma2_cell.sh but WITHOUT per-prompt-delta,
# shape-budget-mode global, or delta-bins 10. Fixed delta=0.15 throughout.
# Uses ranking_loss_ref_fix.py (same as v7).
#
# DATASET: persona | membership | ifeval
# SETTING: s1 | s2 | s3 | s4 | s7

set -euo pipefail

DATASET="${1:?DATASET required (persona|membership|ifeval)}"
SETTING="${2:?SETTING required (s1|s2|s3|s4|s7)}"
MODEL="google/gemma-2-9b-it"

VENV=/workspace/.venv
REPO=/workspace/rankalign
MODELS_DIR=/workspace/models2
OUTPUTS_DIR=/workspace/outputs
LOG_DIR=/workspace/logs
mkdir -p "$MODELS_DIR" "$OUTPUTS_DIR" "$LOG_DIR"

LOG="$LOG_DIR/cell_${DATASET}_${SETTING}.log"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -u +%FT%TZ)] === run_gemma2_v7b_cell.sh DATASET=$DATASET SETTING=$SETTING ==="
echo "[$(date -u +%FT%TZ)] model=$MODEL (v7b: fixed delta=0.15, no ppd, no delta-bins)"

source "$VENV/bin/activate"
cd "$REPO/scripts"

export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Dataset config ---
case "$DATASET" in
    persona)
        TASK="persona-v1"
        EVAL_TASKS=(
            persona-v1-psychopathy persona-v1-machiavellianism
            persona-v1-narcissism persona-v1-desire-to-create-allies
            persona-v1-interest-in-music persona-v1-interest-in-science
        )
        MAX_SEQ=""
        GRAD_CKP=""
        DISC_SHOTS_TRAIN="--disc-shots few"
        DISC_SHOTS_EVAL="--disc-shots few"
        ;;
    membership)
        TASK="membership-sans-rosch-v0"
        EVAL_TASKS=(
            rosch-bird rosch-carpenters-tool rosch-clothing rosch-fruit
            rosch-furniture rosch-sport rosch-toy rosch-vehicle
            rosch-vegetable rosch-weapon
        )
        MAX_SEQ=""
        GRAD_CKP=""
        DISC_SHOTS_TRAIN="--disc-shots few"
        DISC_SHOTS_EVAL="--disc-shots few"
        ;;
    ifeval)
        TASK="ifeval-concat"
        EVAL_TASKS=()
        IFEVAL_DATA_DIR="/workspace/rankalign/data/fixed-prompts-ifeval"
        while IFS= read -r n; do
            EVAL_TASKS+=("ifeval-prompt_$n")
        done < <(
            for f in "$IFEVAL_DATA_DIR"/gpt_ifeval_results_prompt_*.jsonl; do
                basename "$f" | grep -oE '[0-9]+' | head -1
            done | awk '$1+0 <= 21' | sort -n
        )
        MAX_SEQ="--max-seq-len 1024"
        GRAD_CKP="--gradient_checkpointing"
        DISC_SHOTS_TRAIN="--disc-shots zero"
        DISC_SHOTS_EVAL="--disc-shots zero"
        ;;
    *)
        echo "Unknown DATASET: $DATASET (persona|membership|ifeval)"; exit 1 ;;
esac

# --- Per-setting flags ---
# v7b: PPD_FLAGS="" for ALL settings (no --per-prompt-delta, no --shape-budget-mode global)
case "$SETTING" in
    s1)
        LOSS_FLAGS="--nll_validator_weight 1 --nll_generator_weight 1 --preference_loss_weight 0"
        SEMI_FLAG="--labeled-only 0.1"
        FSX_FLAG=""
        TC_FLAG=""
        VLO_FLAG="--validator-log-odds"
        PPD_FLAGS=""
        EVAL_MODES="--self-typicality --neg-typicality"
        DIR_SUFFIX="--full-completion--pref0.0--nllv1.0--nllg1.0--vallogodds--labelonly0.1--fix1"
        ;;
    s2)
        LOSS_FLAGS="--nll_validator_weight 0 --nll_generator_weight 0 --preference_loss_weight 1"
        SEMI_FLAG="--semi-supervised 0.1"
        FSX_FLAG=""
        TC_FLAG=""
        VLO_FLAG="--validator-log-odds"
        PPD_FLAGS=""
        EVAL_MODES="--self-typicality --neg-typicality"
        DIR_SUFFIX="--full-completion--vallogodds--semi0.1--fix1"
        ;;
    s3)
        # New+fsx, no TC — v7b: no --per-prompt-delta, no --shape-budget-mode global
        LOSS_FLAGS="--nll_validator_weight 1 --nll_generator_weight 1 --preference_loss_weight 1"
        SEMI_FLAG="--semi-supervised 0.1"
        FSX_FLAG="--force-same-x"
        TC_FLAG=""
        VLO_FLAG="--validator-log-odds"
        PPD_FLAGS=""
        EVAL_MODES="--self-typicality --neg-typicality"
        DIR_SUFFIX="--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1--fix1"
        ;;
    s4)
        # New+fsx+selfTC — v7b: no ppd
        LOSS_FLAGS="--nll_validator_weight 1 --nll_generator_weight 1 --preference_loss_weight 1"
        SEMI_FLAG="--semi-supervised 0.1"
        FSX_FLAG="--force-same-x"
        TC_FLAG="--self-typicality"
        VLO_FLAG="--validator-log-odds"
        PPD_FLAGS=""
        EVAL_MODES="--self-typicality"
        DIR_SUFFIX="--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1--fix1"
        ;;
    s7)
        # New+fsx+negTC — v7b: no ppd
        LOSS_FLAGS="--nll_validator_weight 1 --nll_generator_weight 1 --preference_loss_weight 1"
        SEMI_FLAG="--semi-supervised 0.1"
        FSX_FLAG="--force-same-x"
        TC_FLAG="--neg-typicality"
        VLO_FLAG="--validator-log-odds"
        PPD_FLAGS=""
        EVAL_MODES="--neg-typicality"
        DIR_SUFFIX="--tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1--fix1"
        ;;
    *)
        echo "Unknown SETTING: $SETTING (s1|s2|s3|s4|s7)"; exit 1 ;;
esac

LORA_FLAG="--lora"
MODEL_REPL=$(echo "$MODEL" | sed 's|/|--|g')
# ranking_loss_ref_fix.py always writes v7- prefix; delta=0.15 fixed; no --ppd in suffix
GLOB_PATTERN="${MODELS_DIR}/v7-${MODEL_REPL}-delta0.15-epoch[012]--${TASK}-all--d2g--random--alpha1.0${DIR_SUFFIX}_merged"

echo "[$(date -u +%FT%TZ)] model dir glob: $GLOB_PATTERN"

# ============================================================
# TRAIN
# ============================================================
echo "[$(date -u +%FT%TZ)] === TRAIN: $SETTING x $TASK ==="

EXISTING=$(ls -dt $GLOB_PATTERN 2>/dev/null | head -1 || true)
if [ -n "$EXISTING" ] && [ -d "$EXISTING" ]; then
    echo "[$(date -u +%FT%TZ)] Model dir exists, skipping train: $EXISTING"
    MODEL_DIR="$EXISTING"
else
    # shellcheck disable=SC2086
    python ranking_loss_ref_fix.py \
        --model "$MODEL" \
        --num_epochs 3 \
        --task "$TASK" \
        --train_g_or_d g \
        --split_type random \
        --all \
        --delta 0.15 \
        $DISC_SHOTS_TRAIN \
        --models-dir "$MODELS_DIR" \
        $LOSS_FLAGS \
        $SEMI_FLAG \
        $FSX_FLAG \
        $TC_FLAG \
        $VLO_FLAG \
        $LORA_FLAG \
        $MAX_SEQ \
        $GRAD_CKP \
        --no-upload-hf --no-wandb

    MODEL_DIR=$(ls -dt $GLOB_PATTERN 2>/dev/null | head -1 || true)
    if [ -z "$MODEL_DIR" ] || [ ! -d "$MODEL_DIR" ]; then
        echo "[$(date -u +%FT%TZ)] ERROR: no model dir found matching: $GLOB_PATTERN"
        exit 1
    fi
    echo "[$(date -u +%FT%TZ)] Train complete. Model dir: $MODEL_DIR"
fi

# ============================================================
# EVAL
# ============================================================
echo "[$(date -u +%FT%TZ)] === EVAL: ${#EVAL_TASKS[@]} tasks, modes='$EVAL_MODES' ==="

EVAL_MODEL_DIR="/workspace/eval_model_${SETTING}"
ln -sfn "$MODEL_DIR" "$EVAL_MODEL_DIR"
echo "[$(date -u +%FT%TZ)] eval model symlink: $EVAL_MODEL_DIR -> $MODEL_DIR"

for MODE in $EVAL_MODES; do
    echo "[$(date -u +%FT%TZ)] --- eval mode: $MODE ---"
    for EVAL_TASK in "${EVAL_TASKS[@]}"; do
        echo "[$(date -u +%FT%TZ)] eval: $EVAL_TASK"
        python eval_by_claude.py \
            --model "$EVAL_MODEL_DIR" \
            --task "$EVAL_TASK" \
            --split_type random \
            $DISC_SHOTS_EVAL \
            --gen-shots zero \
            --outputs-dir "$OUTPUTS_DIR" \
            --validator-log-odds \
            $MODE \
            --base-typicality \
            --base-model-name "$MODEL" \
            --save-scores-csv
        echo "[$(date -u +%FT%TZ)] done: $EVAL_TASK ($MODE)"
    done
    echo "[$(date -u +%FT%TZ)] mode $MODE complete"
done

echo "[$(date -u +%FT%TZ)] === ALL DONE: DATASET=$DATASET SETTING=$SETTING ==="
