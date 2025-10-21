#!/bin/bash
# Full pipeline to generate train and test data for typicality analysis

set -e  # Exit on error

# Default parameters
MODEL="${1:-google/gemma-2-2b}"
DEVICE="${2:-0}"
SEED="${3:-0}"
SPLIT_TYPE="${4:-random}"

echo "=========================================="
echo "Train/Test Typicality Analysis Pipeline"
echo "=========================================="
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Seed: $SEED"
echo "Split type: $SPLIT_TYPE"
echo "=========================================="

# Get safe model name for files (replace / with -)
MODEL_SAFE=$(echo "$MODEL" | tr '/' '-')

# Activate environment
source ~/venvs/venv_lexcons/bin/activate

# ==============================================
# Step 1: Generate TRAIN data from eval.py
# ==============================================
echo ""
echo "Step 1/6: Running eval.py for TRAIN set..."
cd ../scripts
CUDA_VISIBLE_DEVICES=$DEVICE python eval.py \
    --model "$MODEL" \
    --task hypernym \
    --split_type $SPLIT_TYPE \
    --train \
    --use_full_completion_logprobs \
    --debug_save_values

# Check if train file was created
if [ ! -f "../outputs/debug_values_hypernym_logprobs.csv" ]; then
    echo "ERROR: Train eval output not found!"
    exit 1
fi

# Rename to include _train suffix
mv ../outputs/debug_values_hypernym_logprobs.csv ../outputs/debug_values_hypernym_logprobs_train.csv
echo "✓ Train eval data saved"

cd ../typicality

# ==============================================
# Step 2: Generate TEST data from eval.py
# ==============================================
echo ""
echo "Step 2/6: Running eval.py for TEST set..."
cd ../scripts
CUDA_VISIBLE_DEVICES=$DEVICE python eval.py \
    --model "$MODEL" \
    --task hypernym \
    --split_type $SPLIT_TYPE \
    --use_full_completion_logprobs \
    --debug_save_values

# Check if test file was created
if [ ! -f "../outputs/debug_values_hypernym_logprobs.csv" ]; then
    echo "ERROR: Test eval output not found!"
    exit 1
fi

# Rename to include _test suffix
mv ../outputs/debug_values_hypernym_logprobs.csv ../outputs/debug_values_hypernym_logprobs_test.csv
echo "✓ Test eval data saved"

cd ../typicality

# ==============================================
# Step 3: Compute GPT-2 typicality for TRAIN
# ==============================================
echo ""
echo "Step 3/6: Computing GPT-2 typicality scores for TRAIN set..."
python compute_gpt2_typicality.py \
    --split_type $SPLIT_TYPE \
    --train \
    --output "gpt2_typicality_scores_${SPLIT_TYPE}_train.csv"

echo "✓ Train typicality scores saved"

# ==============================================
# Step 4: Compute GPT-2 typicality for TEST
# ==============================================
echo ""
echo "Step 4/6: Computing GPT-2 typicality scores for TEST set..."
python compute_gpt2_typicality.py \
    --split_type $SPLIT_TYPE \
    --output "gpt2_typicality_scores_${SPLIT_TYPE}_test.csv"

echo "✓ Test typicality scores saved"

# ==============================================
# Step 5: Merge TRAIN data
# ==============================================
echo ""
echo "Step 5/6: Merging TRAIN data..."
python merge_data.py \
    --typicality "gpt2_typicality_scores_${SPLIT_TYPE}_train.csv" \
    --eval_output "../outputs/debug_values_hypernym_logprobs_train.csv" \
    --output "merged_data_${MODEL_SAFE}_${SPLIT_TYPE}_train.csv"

echo "✓ Train data merged"

# ==============================================
# Step 6: Merge TEST data
# ==============================================
echo ""
echo "Step 6/6: Merging TEST data..."
python merge_data.py \
    --typicality "gpt2_typicality_scores_${SPLIT_TYPE}_test.csv" \
    --eval_output "../outputs/debug_values_hypernym_logprobs_test.csv" \
    --output "merged_data_${MODEL_SAFE}_${SPLIT_TYPE}_test.csv"

echo "✓ Test data merged"

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  Train: merged_data_${MODEL_SAFE}_${SPLIT_TYPE}_train.csv"
echo "  Test:  merged_data_${MODEL_SAFE}_${SPLIT_TYPE}_test.csv"
echo ""
echo "Next step: Run train/test analysis"
echo "  python train_test_analysis.py \\"
echo "    --train merged_data_${MODEL_SAFE}_${SPLIT_TYPE}_train.csv \\"
echo "    --test merged_data_${MODEL_SAFE}_${SPLIT_TYPE}_test.csv \\"
echo "    --output_dir train_test_outputs"
echo "=========================================="

