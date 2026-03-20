# LOW CHES - should be SAFE (less displacement)
CUDA_VISIBLE_DEVICES=1 python ranking_loss_ref_explore.py \
    --model google/gemma-2-2b \
    --task hypernym \
    --ches_percentile 0 \
    --num_epochs 5 \
    --delta 2.5 \
    --total_samples 5000 \
    --pair_seed 42

# 25th percentile
CUDA_VISIBLE_DEVICES=1 python ranking_loss_ref_explore.py \
    --model google/gemma-2-2b \
    --task hypernym \
    --ches_percentile 25 \
    --num_epochs 5 \
    --delta 2.5 \
    --total_samples 5000 \
    --pair_seed 42

# 50th percentile (median)
CUDA_VISIBLE_DEVICES=1 python ranking_loss_ref_explore.py \
    --model google/gemma-2-2b \
    --task hypernym \
    --ches_percentile 50 \
    --num_epochs 5 \
    --delta 2.5 \
    --total_samples 5000 \
    --pair_seed 42

# 75th percentile
CUDA_VISIBLE_DEVICES=1 python ranking_loss_ref_explore.py \
    --model google/gemma-2-2b \
    --task hypernym \
    --ches_percentile 75 \
    --num_epochs 5 \
    --delta 2.5 \
    --total_samples 5000 \
    --pair_seed 42

# HIGH CHES - should FAIL (catastrophic displacement)
CUDA_VISIBLE_DEVICES=1  python ranking_loss_ref_explore.py \
    --model google/gemma-2-2b \
    --task hypernym \
    --ches_percentile 100 \
    --num_epochs 5 \
    --delta 2.5 \
    --total_samples 5000 \
    --pair_seed 42
