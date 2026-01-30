#!/bin/bash

# Run eval.py on ifeval-X tasks for all model variants
# Evaluates multiple model variants per task:
#   - Base model (google/gemma-2-9b-it)
#   - d2g vanilla (no corrections)
#   - d2g with typcorr
# Uses --train flag to evaluate on training set
#
# Usage: ./run_eval_hypernym_tasks.sh <GPU_LIST>
# Example: ./run_eval_hypernym_tasks.sh 0,1,2,3

GPU_LIST=$1

if [ -z "$GPU_LIST" ]; then
    echo "Usage: $0 <GPU_LIST>"
    echo "  GPU_LIST: comma-separated GPU IDs (e.g., 0,1,2,3)"
    echo ""
    echo "This script evaluates all model variants on ifeval-X tasks."
    echo "Models evaluated per task:"
    echo "  - Base model (google/gemma-2-9b-it)"
    echo "  - d2g (delta=0.15) vanilla (no corrections)"
    echo "  - d2g (delta=0.15) with typcorr"
    exit 1
fi

# Parse GPU list into array
IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}

echo "Using $NUM_GPUS GPUs: ${GPUS[*]}"

EPOCH=1

# ifeval tasks
TASKS=(
    "ifeval-prompt_1"
    "ifeval-prompt_2"
    "ifeval-prompt_3"
    "ifeval-prompt_4"
    "ifeval-prompt_5"
)

mkdir -p logs

# Function to run eval for a single model
run_eval() {
    local GPU=$1
    local MODEL=$2
    local TASK=$3
    local EXTRA_FLAGS=$4
    local LOG_FILE=$5
    
    echo "[GPU $GPU] Evaluating: $(basename $MODEL)"

    echo "$MODEL $EXTRA_FLAGS" >> "$LOG_FILE"
    
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model "$MODEL" \
        --task "$TASK" \
        --split_type random \
        --validator-log-odds \
        --save-scores-csv \
        --disc-shots zero \
        --viz \
        --typicality-correction \
        --length-normalize \
        $EXTRA_FLAGS \
        >> "$LOG_FILE" 2>&1
}

# Function to run all model variants for a task
run_task_evals() {
    local TASK=$1
    local GPU=$2
    local LOG_FILE="logs/eval_${TASK}_all_gpu${GPU}.log"
    
    echo "========================================"
    echo "[GPU $GPU] Starting evals for: $TASK"
    echo "========================================"
    
    # ========================================
    # BASE MODEL (google/gemma-2-9b-it)
    # ========================================
    # #Base model
    # MODEL="google/gemma-2-9b-it"
    # run_eval "$GPU" "$MODEL" "$TASK" "" "$LOG_FILE"

    # #No preference loss
    # MODEL="../models/v5-google--gemma-2-9b-it-delta0.15-epoch${EPOCH}--${TASK}-all--d2g--random--alpha1.0--full-completion--pref0.0--nllv1.0--nllg1.0_merged"
    # run_eval "$GPU" "$MODEL" "$TASK" "" "$LOG_FILE"

    # ========================================
    # d2g (delta=0.15) variants
    # ========================================
    
    # d2g vanilla (no corrections)
    # MODEL="../models/v5-google--gemma-2-9b-it-delta0.15-epoch${EPOCH}--${TASK}-all--d2g--random--alpha1.0--full-completion--nllv1.0--nllg1.0_merged"
    # if [ -d "$MODEL" ]; then
    #     run_eval "$GPU" "$MODEL" "$TASK" "" "$LOG_FILE"
    # else
    #     echo "  [SKIP] Model not found: $MODEL" >> "$LOG_FILE"
    # fi
    
    # d2g with typcorr
    MODEL="../models/v5-google--gemma-2-9b-it-delta0.15-epoch${EPOCH}--${TASK}-all--d2g--random--alpha1.0--tc-online--full-completion--nllv1.0--nllg1.0_merged"
    if [ -d "$MODEL" ]; then
        run_eval "$GPU" "$MODEL" "$TASK" "" "$LOG_FILE"
    else
        echo "  [SKIP] Model not found: $MODEL" >> "$LOG_FILE"
    fi
    
    
    echo "[GPU $GPU] Finished evals for: $TASK"
}

# Distribute tasks across GPUs
echo ""
echo "Distributing ${#TASKS[@]} tasks across $NUM_GPUS GPUs..."
echo ""

for i in "${!TASKS[@]}"; do
    GPU_IDX=$((i % NUM_GPUS))
    GPU=${GPUS[$GPU_IDX]}
    TASK=${TASKS[$i]}
    
    run_task_evals "$TASK" "$GPU" &
done

echo "All tasks launched. Waiting for completion..."
echo "Check logs/ directory for progress."
echo ""

wait

echo ""
echo "========================================"
echo "All evaluations completed!"
echo "========================================"
