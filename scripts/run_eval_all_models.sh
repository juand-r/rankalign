#!/bin/bash

# Run eval.py on all models for tasks with epoch 1-9
# Tasks: elephants, ducklings, jackets, kites, dolls
# Non-typicality correction versions only
# Spread across GPUs 0, 1, 2, 3, 4 (one task per GPU)
#
# Usage: ./run_eval_all_models.sh

source /u/jdr/venvs/venv_lexcons/bin/activate
cd /datastor1/jdr/gv-gap/rankalign/scripts

# Tasks and their assigned GPUs
declare -A TASK_GPU
TASK_GPU["elephants"]=0
TASK_GPU["ducklings"]=1
TASK_GPU["jackets"]=2
TASK_GPU["kites"]=3
TASK_GPU["dolls"]=4

TASKS=("elephants" "ducklings" "jackets" "kites" "dolls")
EPOCHS=(1 2 3 4 5 6 7 8 9)

echo "========================================"
echo "Running eval on all models"
echo "  Tasks: ${TASKS[*]}"
echo "  Epochs: ${EPOCHS[*]}"
echo "  GPUs: 0, 1, 2, 3, 4"
echo "========================================"

# Function to run all epochs for a task on its assigned GPU
run_task() {
    local TASK=$1
    local GPU=${TASK_GPU[$TASK]}
    
    echo "[GPU $GPU] Starting evaluations for hypernym-$TASK"
    
    for EPOCH in "${EPOCHS[@]}"; do
        MODEL="../models/v5-google--gemma-2-2b-delta0.15-epoch${EPOCH}--hypernym-${TASK}-all--d2g--random--alpha1.0--full-completion--nllv1.0--nllg1.0"
        
        if [ -d "$MODEL" ]; then
            echo "[GPU $GPU] Evaluating epoch $EPOCH for hypernym-$TASK"
            CUDA_VISIBLE_DEVICES=$GPU python eval.py \
                --model "$MODEL" \
                --task "hypernym-${TASK}" \
                --split_type random \
                --save-scores-csv \
                --validator-log-odds \
                --viz
        else
            echo "[GPU $GPU] SKIPPING - Model not found: $MODEL"
        fi
    done
    
    echo "[GPU $GPU] Completed all epochs for hypernym-$TASK"
}

# Run all tasks in parallel (each task on its own GPU)
for TASK in "${TASKS[@]}"; do
    run_task "$TASK" &
done

# Wait for all background jobs to complete
echo ""
echo "Waiting for all evaluations to complete..."
wait

echo ""
echo "All evaluations completed!"
