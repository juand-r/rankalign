#!/bin/bash
# Convenience script to run full typicality analysis pipeline
#
# Usage: ./run_typicality_analysis.sh [MODEL] [DEVICE] [SEED] [SPLIT_TYPE]
#   MODEL: e.g., google/gemma-2-2b (default)
#   DEVICE: GPU device number (default: 0)
#   SEED: Random seed (default: 0)
#   SPLIT_TYPE: random, hyper, or both (default: random)

set -e  # Exit on error

MODEL=${1:-google/gemma-2-2b}
DEVICE=${2:-0}
SEED=${3:-0}
SPLIT_TYPE=${4:-random}

# Create output directory if needed
mkdir -p analysis_results

echo "=========================================="
echo "Typicality Analysis Pipeline"
echo "=========================================="
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Seed: $SEED"
echo "Split type: $SPLIT_TYPE"
echo "=========================================="

# Get safe model name for files (replace / with -)
MODEL_SAFE=$(echo "$MODEL" | tr '/' '-')

# Step 1: Run eval.py to generate debug output (always run to ensure correct format)
EVAL_OUTPUT="../outputs/debug_values_hypernym_logprobs.csv"
echo ""
echo "Step 1/3: Running eval.py to generate debug output..."
cd ../scripts
source ~/venvs/venv_lexcons/bin/activate
CUDA_VISIBLE_DEVICES=$DEVICE python eval.py \
    --model "$MODEL" \
    --task hypernym \
    --split_type $SPLIT_TYPE \
    --use_full_completion_logprobs \
    --debug_save_values
cd ../typicality

# Step 2: Compute GPT-2 typicality scores
echo ""
echo "Step 2/3: Computing GPT-2 typicality scores..."
TYPICALITY_OUTPUT="gpt2_typicality_scores_${SPLIT_TYPE}_seed${SEED}.csv"
source ~/venvs/venv_lexcons/bin/activate
python compute_gpt2_typicality.py \
    --split_type $SPLIT_TYPE \
    --output "$TYPICALITY_OUTPUT"

# Step 3: Merge data
echo ""
echo "Step 3/3: Merging data..."
MERGED_OUTPUT="merged_data_${MODEL_SAFE}_${SPLIT_TYPE}_seed${SEED}.csv"
source ~/venvs/venv_lexcons/bin/activate
python merge_data.py \
    --typicality "$TYPICALITY_OUTPUT" \
    --eval_output "$EVAL_OUTPUT" \
    --output "$MERGED_OUTPUT"

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Merged data saved to: $MERGED_OUTPUT"
echo ""
echo "Next steps:"
echo "  1. Examine the data: head $MERGED_OUTPUT"
echo "  2. Run analysis: python analyze_typicality.py --data $MERGED_OUTPUT"
echo "=========================================="

