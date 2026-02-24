#!/bin/bash
set -euo pipefail

# Train and evaluate models on the ifeval-concat task.
# Trains 5 variants:
#   - SFT (pref=0, nllv=1, nllg=1)
#   - Pref only (pref=1, nllv=0, nllg=0)
#   - All-terms (pref=1, nllv=1, nllg=1, validator-log-odds)
#   - All-terms + tc (pref=1, nllv=1, nllg=1, typicality correction, validator-log-odds)
#   - All-terms + tc, no vallogodds (pref=1, nllv=1, nllg=1, typicality correction only)
#
# Evaluates each variant on:
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

echo "EVAL_TASKS: ${EVAL_TASKS[@]}"

run_train_variant() {
    local NAME=$1
    local PREF_W=$2
    local NLLV_W=$3
    local NLLG_W=$4
    local USE_TC=$5
    local USE_VALIDATOR_LOG_ODDS=${6:-0}
    local GPU=${7:-$TRAIN_GPU}
    local LOG_FILE="logs/train_${TASK_CONCAT}_${NAME}_gpu${GPU}.log"

    echo "[GPU $GPU] Training ${NAME} (log: $LOG_FILE)"
    {
        echo "========================================"
        echo "Task: $TASK_CONCAT | Variant: $NAME"
        echo "pref=$PREF_W nllv=$NLLV_W nllg=$NLLG_W tc=$USE_TC validator_log_odds=$USE_VALIDATOR_LOG_ODDS"
        echo "========================================"

        CUDA_VISIBLE_DEVICES="$GPU" python ranking_loss_ref.py \
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
            --force-same-x \
            $([ "$USE_TC" = "1" ] && echo "--typicality-correction") \
            $([ "$USE_VALIDATOR_LOG_ODDS" = "1" ] && echo "--validator-log-odds")
    } >> "$LOG_FILE" 2>&1
}

build_model_dir() {
    local PREF_W=$1
    local NLLV_W=$2
    local NLLG_W=$3
    local USE_TC=$4
    local USE_VALLOGODDS=${5:-0}

    local model_tag="${MODEL//\//--}"
    local typcorr_str=""
    local pref_str=""
    local nllv_str=""
    local nllg_str=""
    local vallogodds_str=""
    # Training saves with 0-based epoch index; NUM_EPOCHS=2 -> last save at epoch 1
    local LAST_EPOCH=$((NUM_EPOCHS - 1))

    if [ "$USE_TC" = "1" ]; then
        typcorr_str="--tc-online"
    fi
    if [ "$PREF_W" != "1" ]; then
        pref_str="--pref${PREF_W}"
    fi
    # Training script (Python) formats weights as float, so 1 -> "1.0" in dir name
    if [ "$NLLV_W" != "0" ]; then
        if [ "$NLLV_W" = "1" ] || [ "$NLLV_W" = "1.0" ]; then
            nllv_str="--nllv1.0"
        else
            nllv_str="--nllv${NLLV_W}"
        fi
    fi
    if [ "$NLLG_W" != "0" ]; then
        if [ "$NLLG_W" = "1" ] || [ "$NLLG_W" = "1.0" ]; then
            nllg_str="--nllg1.0"
        else
            nllg_str="--nllg${NLLG_W}"
        fi
    fi
    if [ "$USE_VALLOGODDS" = "1" ]; then
        vallogodds_str="--vallogodds"
    fi
    # Training uses --force-same-x; model save path includes it, so we must too for eval to find the dir
    local force_same_x_str="--force-same-x"

    echo "../models/v6-${model_tag}-delta${DELTA}-epoch${LAST_EPOCH}--${TASK_CONCAT}-all--d2g--random--alpha${ALPHA}${typcorr_str}--full-completion${pref_str}${nllv_str}${nllg_str}${force_same_x_str}${vallogodds_str}_merged"
}

run_eval_variant() {
    local NAME=$1
    local PREF_W=$2
    local NLLV_W=$3
    local NLLG_W=$4
    local USE_TC=$5
    local USE_VALLOGODDS=$6
    local TASK=$7
    local GPU=$8

    local MODEL_DIR
    local LOG_FILE="logs/eval_${TASK}_${NAME}_gpu${GPU}.log"

    if [ "$NAME" = "base" ]; then
        MODEL_DIR="$MODEL"
    else
        MODEL_DIR=$(build_model_dir "$PREF_W" "$NLLV_W" "$NLLG_W" "$USE_TC" "$USE_VALLOGODDS")
        if [ ! -d "$MODEL_DIR" ]; then
            echo "  [SKIP] Model not found: $MODEL_DIR" >> "$LOG_FILE"
            return
        fi
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

# Training task specs: NAME:PREF_W:NLLV_W:NLLG_W:USE_TC:USE_VALLOGODDS
TRAIN_SPECS=(
    "sft:0.0:1:1:0:0"
    "pref_only:1:0:0:0:0"
    "all_terms:1:1:1:0:1"
    "all_terms_tc:1:1:1:1:1"
)

# Run one GPU worker: execute all tasks whose index % NUM_GPUS == gpu_idx, sequentially.
run_train_worker() {
    local gpu_idx=$1
    local gpu_id=${GPUS[$gpu_idx]}
    local idx=0
    for spec in "${TRAIN_SPECS[@]}"; do
        if [ $((idx % NUM_GPUS)) -eq "$gpu_idx" ]; then
            IFS=':' read -r NAME PREF_W NLLV_W NLLG_W USE_TC USE_VALLOGODDS <<< "$spec"
            run_train_variant "$NAME" "$PREF_W" "$NLLV_W" "$NLLG_W" "$USE_TC" "$USE_VALLOGODDS" "$gpu_id"
        fi
        idx=$((idx + 1))
    done
}

if [ "$MODE" = "train" ] || [ "$MODE" = "both" ]; then
    echo "========================================"
    echo "Training variants on $TASK_CONCAT (GPUs: $GPU_LIST, one task per GPU at a time)"
    echo "========================================"

    for gpu_idx in $(seq 0 $((NUM_GPUS - 1))); do
        run_train_worker "$gpu_idx" &
    done
    wait
    echo "All training runs completed."
fi

# Eval variant specs: NAME:PREF_W:NLLV_W:NLLG_W:USE_TC:USE_VALLOGODDS
VARIANTS=(
    "base:0:0:0:0:0"
    "sft:0.0:1:1:0:0"
    "pref_only:1:0:0:0:0"
    "all_terms:1:1:1:0:1"
    "all_terms_tc:1:1:1:1:1"
)

# Build flat list of eval jobs: each element is "TASK|NAME:PREF_W:NLLV_W:NLLG_W:USE_TC:USE_VALLOGODDS"
EVAL_JOBS=()
for TASK in "${EVAL_TASKS[@]}"; do
    for V in "${VARIANTS[@]}"; do
        EVAL_JOBS+=("${TASK}|${V}")
    done
done

# Run one GPU worker: execute all eval jobs whose index % NUM_GPUS == gpu_idx, sequentially.
run_eval_worker() {
    local gpu_idx=$1
    local gpu_id=${GPUS[$gpu_idx]}
    local idx=0
    for job in "${EVAL_JOBS[@]}"; do
        if [ $((idx % NUM_GPUS)) -eq "$gpu_idx" ]; then
            TASK="${job%%|*}"
            V="${job#*|}"
            IFS=':' read -r NAME PREF_W NLLV_W NLLG_W USE_TC USE_VALLOGODDS <<< "$V"
            run_eval_variant "$NAME" "$PREF_W" "$NLLV_W" "$NLLG_W" "$USE_TC" "$USE_VALLOGODDS" "$TASK" "$gpu_id"
        fi
        idx=$((idx + 1))
    done
}

if [ "$MODE" = "eval" ] || [ "$MODE" = "both" ]; then
    echo "========================================"
    echo "Evaluating variants on concat + per-prompt tasks (one task per GPU at a time)"
    echo "========================================"
    echo "  Eval logs:  $SCRIPT_DIR/logs/eval_<TASK>_<VARIANT>_gpu<N>.log"
    echo "  CSV files:  $SCRIPT_DIR/../outputs/scores_*.csv"
    mkdir -p "$SCRIPT_DIR/../outputs"

    for gpu_idx in $(seq 0 $((NUM_GPUS - 1))); do
        run_eval_worker "$gpu_idx" &
    done
    wait
    echo "All evaluations completed."
fi
