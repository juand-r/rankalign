#!/bin/bash

# Run eval.py on hypernym-X tasks for concat models
# Evaluates two concat models (trained on hypernym-concat-bananas-to-dogs-v2) on individual hypernym tasks
#
# Usage: ./run_eval_concat_model.sh <GPU_LIST>
# Example: ./run_eval_concat_model.sh 0,1,2,3

GPU_LIST=$1

if [ -z "$GPU_LIST" ]; then
    echo "Usage: $0 <GPU_LIST>"
    echo "  GPU_LIST: comma-separated GPU IDs (e.g., 0,1,2,3)"
    echo ""
    echo "This script evaluates concat models on hypernym-X tasks."
    echo "Models evaluated:"
    echo "  - d2g with typcorr + force-same-x"
    echo "  - d2g vanilla + force-same-x"
    exit 1
fi

# Parse GPU list into array
IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}

echo "Using $NUM_GPUS GPUs: ${GPUS[*]}"

# Fixed model paths (trained on concat task)
MODEL_TYPCORR="../models/v5-google--gemma-2-2b-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-v2-all--d2g--random--alpha1.0--typcorr--full-completion--nllv1.0--nllg1.0--force-same-x"
MODEL_VANILLA="../models/v5-google--gemma-2-2b-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-v2-all--d2g--random--alpha1.0--full-completion--nllv1.0--nllg1.0--force-same-x"

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
    "hypernym-kites"
    "hypernym-jackets"
    "hypernym-elephants"
    "hypernym-ducklings"
    "hypernym-dolls"
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
        $EXTRA_FLAGS \
        >> "$LOG_FILE" 2>&1
}

# Function to run eval for concat models on a task
run_task_evals() {
    local TASK=$1
    local GPU=$2
    local LOG_FILE="logs/eval_concat_${TASK}_gpu${GPU}.log"
    
    echo "========================================"
    echo "[GPU $GPU] Starting evals for: $TASK"
    echo "========================================"
    
    # Concat model with typcorr
    if [ -d "$MODEL_TYPCORR" ]; then
        run_eval "$GPU" "$MODEL_TYPCORR" "$TASK" "" "$LOG_FILE"
    else
        echo "  [SKIP] Model not found: $MODEL_TYPCORR" >> "$LOG_FILE"
    fi
    
    # Concat model vanilla (no typcorr)
    if [ -d "$MODEL_VANILLA" ]; then
        run_eval "$GPU" "$MODEL_VANILLA" "$TASK" "" "$LOG_FILE"
    else
        echo "  [SKIP] Model not found: $MODEL_VANILLA" >> "$LOG_FILE"
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
