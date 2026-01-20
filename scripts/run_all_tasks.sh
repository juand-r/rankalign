#!/bin/bash

# Run ranking loss training for a single task
# Each task runs 4 configurations: g/d with/without typicality correction
#
# Usage: ./run_all_tasks.sh <GPU_NUMBER> <TASK>
# Example: ./run_all_tasks.sh 5 hypernym

# CLI arguments
DN=$1
TASK=$2

if [ -z "$DN" ] || [ -z "$TASK" ]; then
    echo "Usage: $0 <GPU_NUMBER> <TASK>"
    echo "  GPU_NUMBER: GPU device number (e.g., 0, 1, 5)"
    echo "  TASK: one of {hypernym, trivia-qa, swords, lambada, ifeval, collie}"
    exit 1
fi

MODEL="google/gemma-2-2b"
NUM_EPOCHS=3
TOTAL_SAMPLES=5110

# TASKS=("hypernym" "trivia-qa" "swords" "lambada" "ifeval" "collie")

# for TASK in "${TASKS[@]}"; do
echo "========================================"
echo "Running task: $TASK on GPU $DN"
echo "========================================"

# Config 1: train_g_or_d=g, delta=0.15, no typicality correction
echo "--- Config 1: g, delta=0.15, no typcorr ---"
CUDA_VISIBLE_DEVICES=$DN python ranking_loss_ref.py \
    --model $MODEL \
    --num_epochs $NUM_EPOCHS \
    --task $TASK \
    --train_g_or_d g \
    --split_type random \
    --nll_validator_weight 1 \
    --nll_generator_weight 1 \
    --all \
    --delta 0.15 \
    --total_samples $TOTAL_SAMPLES \
    --force-same-x

# Config 2: train_g_or_d=g, delta=0.15, with typicality correction
echo "--- Config 2: g, delta=0.15, with typcorr ---"
CUDA_VISIBLE_DEVICES=$DN python ranking_loss_ref.py \
    --model $MODEL \
    --num_epochs $NUM_EPOCHS \
    --task $TASK \
    --train_g_or_d g \
    --split_type random \
    --nll_validator_weight 1 \
    --nll_generator_weight 1 \
    --all \
    --delta 0.15 \
    --total_samples $TOTAL_SAMPLES \
    --typicality-correction \
    --force-same-x

# Config 3: train_g_or_d=d, delta=2.5, no typicality correction
echo "--- Config 3: d, delta=2.5, no typcorr ---"
CUDA_VISIBLE_DEVICES=$DN python ranking_loss_ref.py \
    --model $MODEL \
    --num_epochs $NUM_EPOCHS \
    --task $TASK \
    --train_g_or_d d \
    --split_type random \
    --nll_validator_weight 1 \
    --nll_generator_weight 1 \
    --all \
    --delta 2.5 \
    --total_samples $TOTAL_SAMPLES \
    --force-same-x

# Config 4: train_g_or_d=d, delta=2.5, with typicality correction
echo "--- Config 4: d, delta=2.5, with typcorr ---"
CUDA_VISIBLE_DEVICES=$DN python ranking_loss_ref.py \
    --model $MODEL \
    --num_epochs $NUM_EPOCHS \
    --task $TASK \
    --train_g_or_d d \
    --split_type random \
    --nll_validator_weight 1 \
    --nll_generator_weight 1 \
    --all \
    --delta 2.5 \
    --total_samples $TOTAL_SAMPLES \
    --typicality-correction \
    --force-same-x

echo "Finished task: $TASK"
# done

# echo "All tasks completed!"
