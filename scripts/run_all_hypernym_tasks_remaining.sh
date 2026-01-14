#!/bin/bash

# Run the remaining 3 configs for hypernym-concat-bananas-to-dogs on GPUs 1, 2, 3
# (Config 1 is already running on GPU 0)
#
# GPU 1: Config 2 - g, delta=0.15, with typcorr
# GPU 2: Config 3 - d, delta=2.5, no typcorr
# GPU 3: Config 4 - d, delta=2.5, with typcorr

MODEL="google/gemma-2-2b"
NUM_EPOCHS=3
TOTAL_SAMPLES=5110
TASK="hypernym-concat-bananas-to-dogs"

mkdir -p logs

# Config 2: train_g_or_d=g, delta=0.15, with typicality correction (GPU 1)
echo "Starting Config 2 on GPU 1: g, delta=0.15, with typcorr"
(
CUDA_VISIBLE_DEVICES=1 python ranking_loss_ref.py \
    --model $MODEL \
    --num_epochs $NUM_EPOCHS \
    --task "$TASK" \
    --train_g_or_d g \
    --split_type random \
    --nll_validator_weight 1 \
    --nll_generator_weight 1 \
    --all \
    --delta 0.15 \
    --use-full-completion \
    --total_samples $TOTAL_SAMPLES \
    --typicality-correction
echo "Config 2 (GPU 1) finished!"
) >> logs/train_config2_gpu1.log 2>&1 &

# Config 3: train_g_or_d=d, delta=2.5, no typicality correction (GPU 2)
echo "Starting Config 3 on GPU 2: d, delta=2.5, no typcorr"
(
CUDA_VISIBLE_DEVICES=2 python ranking_loss_ref.py \
    --model $MODEL \
    --num_epochs $NUM_EPOCHS \
    --task "$TASK" \
    --train_g_or_d d \
    --split_type random \
    --nll_validator_weight 1 \
    --nll_generator_weight 1 \
    --all \
    --delta 2.5 \
    --use-full-completion \
    --total_samples $TOTAL_SAMPLES
echo "Config 3 (GPU 2) finished!"
) >> logs/train_config3_gpu2.log 2>&1 &

# Config 4: train_g_or_d=d, delta=2.5, with typicality correction (GPU 3)
echo "Starting Config 4 on GPU 3: d, delta=2.5, with typcorr"
(
CUDA_VISIBLE_DEVICES=3 python ranking_loss_ref.py \
    --model $MODEL \
    --num_epochs $NUM_EPOCHS \
    --task "$TASK" \
    --train_g_or_d d \
    --split_type random \
    --nll_validator_weight 1 \
    --nll_generator_weight 1 \
    --all \
    --delta 2.5 \
    --use-full-completion \
    --total_samples $TOTAL_SAMPLES \
    --typicality-correction
echo "Config 4 (GPU 3) finished!"
) >> logs/train_config4_gpu3.log 2>&1 &

echo ""
echo "All 3 configs launched on GPUs 1, 2, 3"
echo "Check logs/train_config{2,3,4}_gpu{1,2,3}.log for progress"
echo ""

wait
echo "All remaining configs completed!"
