 #!/bin/bash
set -euo pipefail

# Run ranking loss training for all ifeval per prompt tasks
# Each task runs: d2g with/without typicality correction
# Distributes tasks across multiple GPUs
#
# Usage: ./run_all_ifeval_tasks.sh <GPU_LIST>
# Example: ./run_all_ifeval_tasks.sh 0,1,2,3,4,5,6,7

# Always run from this script's directory so relative paths work.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# CLI arguments
GPU_LIST=$1

if [ -z "$GPU_LIST" ]; then
    echo "Usage: $0 <GPU_LIST>"
    echo "  GPU_LIST: Comma-separated list of GPU numbers (e.g., 0,1,2,3,4,5,6,7)"
    echo ""
    echo "Example: $0 0,1,2,3,4,5,6,7"
    echo ""
    echo "This will train ifeval-X tasks with 2 configs each."
    exit 1
fi

# Parse GPU list into array
IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}

echo "Using $NUM_GPUS GPUs: ${GPUS[*]}"

MODEL="google/gemma-2-9b-it"
NUM_EPOCHS=2
TOTAL_SAMPLES=5110

TASKS=(
    # "ifeval-prompt_1"
    # "ifeval-prompt_2"
    # "ifeval-prompt_3"
    # "ifeval-prompt_4"
    "ifeval-prompt_5"
)

NUM_TASKS=${#TASKS[@]}
TASKS_PER_GPU=$(( (NUM_TASKS + NUM_GPUS - 1) / NUM_GPUS ))

echo "Total tasks: $NUM_TASKS"
echo "Tasks per GPU: ~$TASKS_PER_GPU"
echo ""

# Function to run all configs for a single task on a given GPU
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

#  preference loss = 0.0 (no rankalign)
echo "--- Config 0: no preference loss ---"
CUDA_VISIBLE_DEVICES="$DN" python ranking_loss_ref.py \
    --model "$MODEL" \
    --num_epochs "$NUM_EPOCHS" \
    --task "$TASK" \
    --train_g_or_d g \
    --nll_validator_weight 1 \
    --nll_generator_weight 1 \
    --preference_loss_weight 0.0 \
    --all \
    --delta 0.15 \
    --total_samples "$TOTAL_SAMPLES" \
    --save_steps 1 \
    --lora

#  train_g_or_d=g, delta=0.15, no typ correction
echo "--- Config 1: g, delta=0.15, without typcorr ---"
CUDA_VISIBLE_DEVICES="$DN" python ranking_loss_ref.py \
    --model "$MODEL" \
    --num_epochs "$NUM_EPOCHS" \
    --task "$TASK" \
    --train_g_or_d g \
    --nll_validator_weight 1 \
    --nll_generator_weight 1 \
    --all \
    --delta 0.15 \
    --total_samples "$TOTAL_SAMPLES" \
    --save_steps 1 \
    --lora

#  train_g_or_d=g, delta=0.15, with typ correction
echo "--- Config 2: g, delta=0.15, with typcorr ---"
CUDA_VISIBLE_DEVICES="$DN" python ranking_loss_ref.py \
    --model "$MODEL" \
    --num_epochs "$NUM_EPOCHS" \
    --task "$TASK" \
    --train_g_or_d g \
    --nll_validator_weight 1 \
    --nll_generator_weight 1 \
    --all \
    --delta 0.15 \
    --total_samples "$TOTAL_SAMPLES" \
    --save_steps 1 \
    --lora \
    --typicality-correction

echo "Finished task: $TASK"
    } >> "$LOG_FILE" 2>&1
}

# Function to run multiple tasks sequentially on one GPU
run_gpu_tasks() {
    local DN=$1
    shift
    local GPU_TASKS=("$@")
    
    for TASK in "${GPU_TASKS[@]}"; do
        run_task "$DN" "$TASK"
    done
    
    echo "[GPU $DN] All tasks completed!"
}

# Distribute tasks to GPUs
declare -A GPU_TASK_LISTS

for i in "${!TASKS[@]}"; do
    GPU_IDX=$((i % NUM_GPUS))
    GPU_NUM=${GPUS[$GPU_IDX]}
    GPU_TASK_LISTS[$GPU_NUM]+="${TASKS[$i]}"$'\n'
done

# Launch all GPUs in parallel
echo "Launching training on all GPUs..."
echo ""

for GPU in "${GPUS[@]}"; do
    # Convert newline-separated string back to array
    # read only consumes the first line, so use mapfile to capture all tasks
    mapfile -t TASK_ARRAY <<< "${GPU_TASK_LISTS[$GPU]}"
    
    # Filter out empty entries
    FILTERED_TASKS=()
    for t in "${TASK_ARRAY[@]}"; do
        [[ -n "$t" ]] && FILTERED_TASKS+=("$t")
    done
    
    if [ ${#FILTERED_TASKS[@]} -gt 0 ]; then
        echo "GPU $GPU will train: ${FILTERED_TASKS[*]}"
        run_gpu_tasks "$GPU" "${FILTERED_TASKS[@]}" &
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
