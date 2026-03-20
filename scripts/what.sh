#!/bin/bash
WHO IWH
# Run ranking loss training for all 33 hypernym-X tasks
# Each task runs 4 configurations: g/d with/without typicality correction
# Distributes tasks across multiple GPUs
#
# Usage: ./run_all_hypernym_tasks.sh <GPU_LIST>
# Example: ./run_all_hypernym_tasks.sh 0,1,2,3,4,5,6,7
#
# With 33 tasks and 8 GPUs: GPU 0 gets 5 tasks, others get 4.
# Tasks are distributed round-robin across GPUs.

# CLI arguments
GPU_LIST=$1

if [ -z "$GPU_LIST" ]; then
    echo "Usage: $0 <GPU_LIST>"
    echo "  GPU_LIST: Comma-separated list of GPU numbers (e.g., 0,1,2,3,4,5,6,7)"
    echo ""
    echo "Example: $0 0,1,2,3,4,5,6,7"
    echo ""
    echo "This will train 33 hypernym-X tasks with 4 configs each."
    echo "Tasks are distributed 4 per GPU (needs 8 GPUs for 32 tasks)."
    exit 1
fi

# Parse GPU list into array
IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}

echo "Using $NUM_GPUS GPUs: ${GPUS[*]}"

MODEL="google/gemma-2-2b"
NUM_EPOCHS=5
TOTAL_SAMPLES=5110

# All 33 hypernym-X tasks (32 individual + 1 concat)
TASKS=(
    "hypernym-bananas"
    "hypernym-bazookas"
    "hypernym-cabinets"
    "hypernym-cars"
    "hypernym-chairs"
    "hypernym-crows"
    "hypernym-diapers"
    "hypernym-dogs"
    "hypernym-dolls"
    "hypernym-ducklings"
    "hypernym-elephants"
    "hypernym-guns"
    "hypernym-hammers"
    "hypernym-helmets"
    "hypernym-jackets"
    "hypernym-kayaks"
    "hypernym-kites"
#    "hypernym-magnifying glasses"
#    "hypernym-mirrors"
#    "hypernym-nuts"
#    "hypernym-olives"
#    "hypernym-oysters"
#    "hypernym-penguins"
#    "hypernym-puppies"
#    "hypernym-rocking horses"
#    "hypernym-scallions"
#    "hypernym-spatulas"
#    "hypernym-spinach"
#    "hypernym-strollers"
#    "hypernym-swords"
#    "hypernym-turkeys"
#    "hypernym-wagons"
#    "hypernym-concat"
#    "hypernym-concat-bananas-to-dogs"
)

NUM_TASKS=${#TASKS[@]}
TASKS_PER_GPU=$(( (NUM_TASKS + NUM_GPUS - 1) / NUM_GPUS ))

echo "Total tasks: $NUM_TASKS"
echo "Tasks per GPU: ~$TASKS_PER_GPU"
echo ""

# Function to run all 4 configs for a single task on a given GPU
run_task() {
    local DN=$1
    local TASK=$2
    local LOG_FILE="logs/train_${TASK//[ ]/_}_gpu${DN}.log"
    
    mkdir -p logs
    
    echo "[GPU $DN] Starting task: $TASK (logging to $LOG_FILE)"
    
    {
echo "========================================"
echo "Running task: $TASK on GPU $DN"
echo "========================================"

# Config 1: train_g_or_d=g, delta=0.15, no typicality correction
echo "--- Config 1: g, delta=0.15, no typcorr ---"
CUDA_VISIBLE_DEVICES=$DN python ranking_loss_ref.py \
    --model $MODEL \
    --num_epochs $NUM_EPOCHS \
            --task "$TASK" \
    --train_g_or_d g \
    --split_type random \
    --nll_validator_weight 1 \
    --nll_generator_weight 1 \
    --preference_loss_weight 0 \
    --all \
    --delta 0.15 \
    --total_samples $TOTAL_SAMPLES \
    --validator-log-odds

# Config 2: train_g_or_d=g, delta=0.15, with length normalization
echo "--- Config 2: g, delta=0.15, with typcorr ---"
CUDA_VISIBLE_DEVICES=$DN python ranking_loss_ref.py \
    --model $MODEL \
    --num_epochs $NUM_EPOCHS \
            --task "$TASK" \
    --train_g_or_d g \
    --split_type random \
    --nll_validator_weight 1 \
    --nll_generator_weight 1 \
    --preference_loss_weight 0 \
    --all \
    --delta 0.15 \
    --total_samples $TOTAL_SAMPLES \
    --typicality-correction \
    --validator-log-odds

# Config: with typcorr and length normalization
echo "--- Config 4: g, delta=0.15, with typcorr and length normalization ---"
CUDA_VISIBLE_DEVICES=$DN python ranking_loss_ref.py \
    --model $MODEL \
    --num_epochs $NUM_EPOCHS \
            --task "$TASK" \
    --train_g_or_d g \
    --split_type random \
    --nll_validator_weight 1 \
    --nll_generator_weight 1 \
    --preference_loss_weight 0 \
    --all \
    --delta 0.15 \
    --total_samples $TOTAL_SAMPLES \
    --typicality-correction \
    --length-normalize \
    --validator-log-odds

echo "--- Config 2: g, delta=0.15, with length normalization ---"
CUDA_VISIBLE_DEVICES=$DN python ranking_loss_ref.py \
    --model $MODEL \
    --num_epochs $NUM_EPOCHS \
            --task "$TASK" \
    --train_g_or_d g \
    --split_type random \
    --nll_validator_weight 1 \
    --nll_generator_weight 1 \
    --preference_loss_weight 0 \
    --all \
    --delta 0.15 \
    --total_samples $TOTAL_SAMPLES \
    --length-normalize \
    --validator-log-odds

# Config 3: train_g_or_d=d, delta=2.5, no typicality correction
#echo "--- Config 3: d, delta=2.5, no typcorr, with length normalization ---"
#CUDA_VISIBLE_DEVICES=$DN python ranking_loss_ref.py \
#    --model $MODEL \
#    --num_epochs $NUM_EPOCHS \
#            --task "$TASK" \
#    --train_g_or_d d \
#    --split_type random \
#    --nll_validator_weight 1 \
#    --nll_generator_weight 1 \
#    --all \
#    --delta 2.5 \
#    --total_samples $TOTAL_SAMPLES \
#    --length-normalize

# Config 4: train_g_or_d=d, delta=2.5, with length normalization
#echo "--- Config 4: d, delta=2.5, with typcorr ---"
#CUDA_VISIBLE_DEVICES=$DN python ranking_loss_ref.py \
#    --model $MODEL \
#    --num_epochs $NUM_EPOCHS \
#            --task "$TASK" \
#    --train_g_or_d d \
#    --split_type random \
#    --nll_validator_weight 1 \
#    --nll_generator_weight 1 \
#    --all \
#    --delta 2.5 \
#    --total_samples $TOTAL_SAMPLES \
#    --length-normalize


#echo "--- Config 4: d, delta=2.5, with typcorr and length normalization ---"
#CUDA_VISIBLE_DEVICES=$DN python ranking_loss_ref.py \
#    --model $MODEL \
#    --num_epochs $NUM_EPOCHS \
#            --task "$TASK" \
#    --train_g_or_d d \
#    --split_type random \
#    --nll_validator_weight 1 \
#    --nll_generator_weight 1 \
#    --all \
#    --delta 2.5 \
#    --total_samples $TOTAL_SAMPLES \
#    --typicality-correction \
#    --length-normalize



echo "Finished task: $TASK"
    } >> "$LOG_FILE" 2>&1
}

# Distribute tasks to GPUs (round-robin assignment)
declare -a GPU_TASK_LISTS

# Initialize empty task lists for each GPU
for i in "${!GPUS[@]}"; do
    GPU_TASK_LISTS[$i]=""
done

# Assign tasks round-robin to GPUs
for i in "${!TASKS[@]}"; do
    GPU_IDX=$((i % NUM_GPUS))
    if [ -z "${GPU_TASK_LISTS[$GPU_IDX]}" ]; then
        GPU_TASK_LISTS[$GPU_IDX]="${TASKS[$i]}"
    else
        GPU_TASK_LISTS[$GPU_IDX]+=$'\n'"${TASKS[$i]}"
    fi
done

# Launch one worker per GPU - each worker processes its tasks sequentially
echo "Launching training on all GPUs..."
echo ""

for i in "${!GPUS[@]}"; do
    GPU=${GPUS[$i]}
    TASK_LIST="${GPU_TASK_LISTS[$i]}"
    
    if [ -n "$TASK_LIST" ]; then
        # Convert newline-separated string to array
        IFS=$'\n' read -ra TASK_ARRAY <<< "$TASK_LIST"
        
        echo "GPU $GPU will train sequentially: ${TASK_ARRAY[*]}"
        
        # Launch a background worker for this GPU that processes tasks one at a time
        (
            for TASK in "${TASK_ARRAY[@]}"; do
                [[ -n "$TASK" ]] && run_task "$GPU" "$TASK"
            done
            echo "[GPU $GPU] All tasks completed!"
        ) &
    fi
done

echo ""
echo "All GPUs launched. Waiting for completion..."
echo "Check logs/ directory for progress."
echo ""

# Wait for all background jobs to complete
wait

echo ""
echo "========================================"
echo "All training completed!"
echo "========================================"
