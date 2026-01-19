#!/bin/bash

# Run eval.py on hypernym-X tasks for lenorm models
# Evaluates 4 model variants per task (d2g/g2d with lenorm, with/without typcorr)
# Uses --train flag to evaluate on training set and --fp32-model for continuous log-odds
#
# Usage: ./run_eval_hypernym_tasks.sh <GPU_LIST>
# Example: ./run_eval_hypernym_tasks.sh 0,1,2,3

GPU_LIST=$1

if [ -z "$GPU_LIST" ]; then
    echo "Usage: $0 <GPU_LIST>"
    echo "  GPU_LIST: comma-separated GPU IDs (e.g., 0,1,2,3)"
    echo ""
    echo "This script evaluates lenorm models on hypernym-X tasks."
    echo "Models evaluated per task:"
    echo "  - d2g (delta=0.15) with lenorm"
    echo "  - d2g (delta=0.15) with typcorr + lenorm"
    echo "  - g2d (delta=2.5) with lenorm"
    echo "  - g2d (delta=2.5) with typcorr + lenorm"
    exit 1
fi

# Parse GPU list into array
IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}

echo "Using $NUM_GPUS GPUs: ${GPUS[*]}"

EPOCH=2

# Hypernym tasks
TASKS=(
    "hypernym-bananas"
    "hypernym-bazookas"
    "hypernym-cabinets"
    "hypernym-cars"
    "hypernym-chairs"
    "hypernym-crows"
    "hypernym-diapers"
    "hypernym-dogs"
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
    
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model "$MODEL" \
        --task "$TASK" \
        --split_type random \
        --validator-log-odds \
        --typicality-correction \
        --save-scores-csv \
        --viz \
        --use_full_completion_logprobs \
        --train \
        --fp32-model \
        $EXTRA_FLAGS \
        >> "$LOG_FILE" 2>&1
}

# Function to run all 4 model variants for a task
run_task_evals() {
    local TASK=$1
    local GPU=$2
    local LOG_FILE="logs/eval_${TASK}_lenorm_gpu${GPU}.log"
    
    echo "========================================"
    echo "[GPU $GPU] Starting evals for: $TASK"
    echo "========================================"
    
    # d2g (delta=0.15) with lenorm only
    MODEL="../models/v5-google--gemma-2-2b-delta0.15-epoch${EPOCH}--${TASK}-all--d2g--random--alpha1.0--lenorm--full-completion--nllv1.0--nllg1.0"
    if [ -d "$MODEL" ]; then
        run_eval "$GPU" "$MODEL" "$TASK" "" "$LOG_FILE"
    else
        echo "  [SKIP] Model not found: $MODEL" >> "$LOG_FILE"
    fi
    
    # d2g (delta=0.15) with typcorr + lenorm
    MODEL="../models/v5-google--gemma-2-2b-delta0.15-epoch${EPOCH}--${TASK}-all--d2g--random--alpha1.0--typcorr--lenorm--full-completion--nllv1.0--nllg1.0"
    if [ -d "$MODEL" ]; then
        run_eval "$GPU" "$MODEL" "$TASK" "" "$LOG_FILE"
    else
        echo "  [SKIP] Model not found: $MODEL" >> "$LOG_FILE"
    fi
    
    # g2d (delta=2.5) with lenorm only
    MODEL="../models/v5-google--gemma-2-2b-delta2.5-epoch${EPOCH}--${TASK}-all--g2d--random--alpha1.0--lenorm--full-completion--nllv1.0--nllg1.0"
    if [ -d "$MODEL" ]; then
        run_eval "$GPU" "$MODEL" "$TASK" "" "$LOG_FILE"
    else
        echo "  [SKIP] Model not found: $MODEL" >> "$LOG_FILE"
    fi
    
    # g2d (delta=2.5) with typcorr + lenorm
    MODEL="../models/v5-google--gemma-2-2b-delta2.5-epoch${EPOCH}--${TASK}-all--g2d--random--alpha1.0--typcorr--lenorm--full-completion--nllv1.0--nllg1.0"
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
