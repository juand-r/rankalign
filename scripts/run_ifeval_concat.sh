#!/bin/bash
set -euo pipefail

# Train and evaluate models on the ifeval-concat task.
# Trains 4 variants:
#   - SFT (pref=0, nllv=1, nllg=1)
#   - Pref only (pref=1, nllv=0, nllg=0)
#   - All-terms (pref=1, nllv=1, nllg=1)
#   - All-terms + tc (pref=1, nllv=1, nllg=1, typicality correction)
#
# Evaluates each variant on:
#   - ifeval-concat
#   - each ifeval-prompt_* task
#
# Usage:
#   ./run_ifeval_concat.sh <GPU_LIST> [--train-only|--eval-only]
# Example:
#   ./run_ifeval_concat.sh 0,1,2,3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GPU_LIST=${1:-}
MODE="both"
shift || true

while [[ $# -gt 0 ]]; do
    case $1 in
        --train-only)
            MODE="train"
            shift
            ;;
        --eval-only)
            MODE="eval"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$GPU_LIST" ]; then
    echo "Usage: $0 <GPU_LIST> [--train-only|--eval-only]"
    echo "  GPU_LIST: comma-separated GPU IDs (e.g., 0,1,2,3)"
    exit 1
fi

IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}
TRAIN_GPU=${GPUS[0]}

MODEL="google/gemma-2-9b-it"
NUM_EPOCHS=2
TOTAL_SAMPLES=5110
DELTA=0.15
ALPHA=1.0
TASK_CONCAT="ifeval-concat"

DATA_DIR="../data/fixed-prompts-ifeval"

mkdir -p logs

TASKS=()
for f in "$DATA_DIR"/gpt_ifeval_results_*.jsonl; do
    [ -e "$f" ] || continue
    base=$(basename "$f")
    name="${base#gpt_ifeval_results_}"
    name="${name%.jsonl}"
    TASKS+=("ifeval-$name")
done

if [ ${#TASKS[@]} -eq 0 ]; then
    echo "No ifeval prompt datasets found in $DATA_DIR"
    exit 1
fi

# Always include concat task for evaluation
EVAL_TASKS=("$TASK_CONCAT")
EVAL_TASKS+=("${TASKS[@]}")

run_train_variant() {
    local NAME=$1
    local PREF_W=$2
    local NLLV_W=$3
    local NLLG_W=$4
    local USE_TC=$5
    local USE_VALIDATOR_LOG_ODDS=${6:-0}
    local LOG_FILE="logs/train_${TASK_CONCAT}_${NAME}_gpu${TRAIN_GPU}.log"

    echo "[GPU $TRAIN_GPU] Training ${NAME} (log: $LOG_FILE)"
    {
        echo "========================================"
        echo "Task: $TASK_CONCAT | Variant: $NAME"
        echo "pref=$PREF_W nllv=$NLLV_W nllg=$NLLG_W tc=$USE_TC validator_log_odds=$USE_VALIDATOR_LOG_ODDS"
        echo "========================================"

        CUDA_VISIBLE_DEVICES="$TRAIN_GPU" python ranking_loss_ref.py \
            --model "$MODEL" \
            --num_epochs "$NUM_EPOCHS" \
            --task "$TASK_CONCAT" \
            --train_g_or_d g \
            --nll_validator_weight "$NLLV_W" \
            --nll_generator_weight "$NLLG_W" \
            --preference_loss_weight "$PREF_W" \
            --all \
            --delta "$DELTA" \
            --split_type random \
            --alpha "$ALPHA" \
            --total_samples "$TOTAL_SAMPLES" \
            --save_steps 1 \
            --lora \
            $([ "$USE_TC" = "1" ] && echo "--typicality-correction") \
            $([ "$USE_VALIDATOR_LOG_ODDS" = "1" ] && echo "--validator-log-odds")
    } >> "$LOG_FILE" 2>&1
}

build_model_dir() {
    local PREF_W=$1
    local NLLV_W=$2
    local NLLG_W=$3
    local USE_TC=$4

    local model_tag="${MODEL//\//--}"
    local typcorr_str=""
    local pref_str=""
    local nllv_str=""
    local nllg_str=""

    if [ "$USE_TC" = "1" ]; then
        typcorr_str="--tc-online"
    fi
    if [ "$PREF_W" != "1" ]; then
        pref_str="--pref${PREF_W}"
    fi
    if [ "$NLLV_W" != "0" ]; then
        nllv_str="--nllv${NLLV_W}"
    fi
    if [ "$NLLG_W" != "0" ]; then
        nllg_str="--nllg${NLLG_W}"
    fi

    echo "../models/v6-${model_tag}-delta${DELTA}-epoch${NUM_EPOCHS}--${TASK_CONCAT}-all--d2g--random--alpha${ALPHA}${typcorr_str}--full-completion${pref_str}${nllv_str}${nllg_str}_merged"
}

run_eval_variant() {
    local NAME=$1
    local PREF_W=$2
    local NLLV_W=$3
    local NLLG_W=$4
    local USE_TC=$5
    local TASK=$6
    local GPU=$7

    local MODEL_DIR
    MODEL_DIR=$(build_model_dir "$PREF_W" "$NLLV_W" "$NLLG_W" "$USE_TC")
    local LOG_FILE="logs/eval_${TASK}_${NAME}_gpu${GPU}.log"

    if [ ! -d "$MODEL_DIR" ]; then
        echo "  [SKIP] Model not found: $MODEL_DIR" >> "$LOG_FILE"
        return
    fi

    CUDA_VISIBLE_DEVICES="$GPU" python eval.py \
        --model "$MODEL_DIR" \
        --task "$TASK" \
        --split_type random \
        --validator-log-odds \
        --save-scores-csv \
        --disc-shots zero \
        --viz \
        --length-normalize \
        --typicality-correction \
        >> "$LOG_FILE" 2>&1
}

if [ "$MODE" = "train" ] || [ "$MODE" = "both" ]; then
    echo "========================================"
    echo "Training variants on $TASK_CONCAT (GPU $TRAIN_GPU)"
    echo "========================================"

    run_train_variant "sft" 0.0 1 1 0 0
    run_train_variant "pref_only" 1 0 0 0 0
    run_train_variant "all_terms" 1 1 1 0 1
    run_train_variant "all_terms_tc" 1 1 1 1 1
fi

if [ "$MODE" = "eval" ] || [ "$MODE" = "both" ]; then
    echo "========================================"
    echo "Evaluating variants on concat + per-prompt tasks"
    echo "========================================"

    VARIANTS=(
        "sft:0.0:1:1:0"
        "pref_only:1:0:0:0"
        "all_terms:1:1:1:0"
        "all_terms_tc:1:1:1:1"
    )

    idx=0
    for TASK in "${EVAL_TASKS[@]}"; do
        for V in "${VARIANTS[@]}"; do
            IFS=':' read -r NAME PREF_W NLLV_W NLLG_W USE_TC <<< "$V"
            GPU=${GPUS[$((idx % NUM_GPUS))]}
            run_eval_variant "$NAME" "$PREF_W" "$NLLV_W" "$NLLG_W" "$USE_TC" "$TASK" "$GPU" &
            idx=$((idx + 1))
        done
    done

    wait
    echo "All evaluations completed."
fi
