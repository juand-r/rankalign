#!/bin/bash
# run_qwen35_cell.sh DATASET SETTING
#
# Trains Qwen/Qwen3.5-9B on one cell (dataset x setting) then evals.
# Designed for RunPod TAUR pods — calls Python scripts directly.
#
# DATASET: persona | membership | ifeval
# SETTING: s1 | s2 | s4 | s7
#
# Idempotent: skips train if a model dir already exists; eval skips per-task
# if the score CSV exists (handled by eval_by_claude.py itself).
#
# Key differences from run_gemma2_cell.sh:
#   - Uses requirements-gemma4.txt (transformers 5.8.1) NOT requirements.txt
#   - disc-shots: NOT passed for training (auto-detect sets zero for Qwen3+)
#   - disc-shots: zero for eval (Qwen3.5 is a chat model)
#   - MODELS_DIR: /workspace/models_q35
#   - --lora only (no --gemma4-lora)

set -euo pipefail

DATASET="${1:?DATASET required (persona|membership|ifeval)}"
SETTING="${2:?SETTING required (s1|s2|s4|s7)}"
MODEL="Qwen/Qwen3.5-9B"

VENV=/workspace/.venv
REPO=/workspace/rankalign
MODELS_DIR=/workspace/models_q35
OUTPUTS_DIR=/workspace/outputs
LOG_DIR=/workspace/logs
mkdir -p "$MODELS_DIR" "$OUTPUTS_DIR" "$LOG_DIR"

LOG="$LOG_DIR/cell_${DATASET}_${SETTING}.log"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -u +%FT%TZ)] === run_qwen35_cell.sh DATASET=$DATASET SETTING=$SETTING ==="
echo "[$(date -u +%FT%TZ)] model=$MODEL"

source "$VENV/bin/activate"
cd "$REPO/scripts"

export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Guard: ensure moe.py is patched (idempotent — setup does this but may have been skipped)
MOE_FILE=$(python -c "import transformers, os; print(os.path.join(os.path.dirname(transformers.__file__), 'integrations', 'moe.py'))" 2>/dev/null || true)
if [ -n "$MOE_FILE" ] && [ -f "$MOE_FILE" ]; then
    if grep -q "^from __future__ import annotations" "$MOE_FILE"; then
        sed -i "s/^from __future__ import annotations$/# from __future__ import annotations (patched: torch 2.4.1 compat)/" "$MOE_FILE"
        echo "[$(date -u +%FT%TZ)] Applied moe.py patch"
    fi
fi

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
        ;;
    membership)
        TASK="membership-sans-rosch-v0"
        EVAL_TASKS=(
            rosch-bird rosch-carpenters-tool rosch-clothing rosch-fruit
            rosch-furniture rosch-sport rosch-toy rosch-vehicle
            rosch-vegetable rosch-weapon
        )
        MAX_SEQ=""
        ;;
    ifeval)
        TASK="ifeval-concat"
        EVAL_TASKS=()
        for n in $(seq 1 21); do EVAL_TASKS+=("ifeval-prompt_$n"); done
        MAX_SEQ="--max-seq-len 1024"
        ;;
    *)
        echo "Unknown DATASET: $DATASET (persona|membership|ifeval)"; exit 1 ;;
esac

# --- Per-setting flags ---
# Qwen3.5-9B uses --lora (NOT --gemma4-lora).
# disc-shots: NOT passed to training (auto-detect = zero for Qwen3+ chat models).
# validator-log-odds: used for all settings.
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
    s4)
        LOSS_FLAGS="--nll_validator_weight 1 --nll_generator_weight 1 --preference_loss_weight 1"
        SEMI_FLAG="--semi-supervised 0.1"
        FSX_FLAG="--force-same-x"
        TC_FLAG="--self-typicality"
        VLO_FLAG="--validator-log-odds"
        PPD_FLAGS="--per-prompt-delta --shape-budget-mode global"
        EVAL_MODES="--self-typicality"
        DIR_SUFFIX="--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1"
        ;;
    s7)
        LOSS_FLAGS="--nll_validator_weight 1 --nll_generator_weight 1 --preference_loss_weight 1"
        SEMI_FLAG="--semi-supervised 0.1"
        FSX_FLAG="--force-same-x"
        TC_FLAG="--neg-typicality"
        VLO_FLAG="--validator-log-odds"
        PPD_FLAGS="--per-prompt-delta --shape-budget-mode global"
        EVAL_MODES="--neg-typicality"
        DIR_SUFFIX="--tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1"
        ;;
    *)
        echo "Unknown SETTING: $SETTING (s1|s2|s4|s7)"; exit 1 ;;
esac

# Qwen3.5-9B uses LoRA; eval reads the _merged dir
LORA_FLAG="--lora"
MODEL_REPL=$(echo "$MODEL" | sed 's|/|--|g')
GLOB_PATTERN="${MODELS_DIR}/v7-${MODEL_REPL}-delta*-epoch[012]--${TASK}-all--d2g--random--alpha1.0${DIR_SUFFIX}_merged"

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
    # disc-shots intentionally NOT passed — training script auto-detects Qwen3+
    # and sets disc_shots=zero (appropriate for chat-template models).
    # shellcheck disable=SC2086
    python ranking_loss_ref_fix.py \
        --model "$MODEL" \
        --num_epochs 3 \
        --task "$TASK" \
        --train_g_or_d g \
        --split_type random \
        --all \
        --delta 0.15 \
        --delta-bins 10 \
        --models-dir "$MODELS_DIR" \
        $LOSS_FLAGS \
        $SEMI_FLAG \
        $FSX_FLAG \
        $TC_FLAG \
        $VLO_FLAG \
        $PPD_FLAGS \
        $LORA_FLAG \
        $MAX_SEQ \
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
echo "[$(date -u +%FT%TZ)] model dir: $MODEL_DIR"

for MODE in $EVAL_MODES; do
    echo "[$(date -u +%FT%TZ)] --- eval mode: $MODE ---"
    for EVAL_TASK in "${EVAL_TASKS[@]}"; do
        echo "[$(date -u +%FT%TZ)] eval: $EVAL_TASK"
        python eval_by_claude.py \
            --model "$MODEL_DIR" \
            --task "$EVAL_TASK" \
            --split_type random \
            --disc-shots zero \
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
