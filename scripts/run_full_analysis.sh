#!/bin/bash
# Run full analysis to verify log-odds vs log-probs implementation

set -e  # Exit on error

TASK=${1:-collie}
MODEL=${2:-google/gemma-2-2b-it}
DEVICE=${3:-7}

echo "=========================================="
echo "Running Full Analysis for Task: $TASK"
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "=========================================="

cd "$(dirname "$0")"

# Run with log-odds (default mode)
echo ""
echo "Step 1/3: Running evaluation with log-odds..."
CUDA_VISIBLE_DEVICES=$DEVICE python eval.py \
    --model "$MODEL" \
    --task "$TASK" \
    --disc-shots zero \
    --debug_save_values

# Run with log-probs (full completion mode)
echo ""
echo "Step 2/3: Running evaluation with log-probs..."
CUDA_VISIBLE_DEVICES=$DEVICE python eval.py \
    --model "$MODEL" \
    --task "$TASK" \
    --disc-shots zero \
    --debug_save_values \
    --use_full_completion_logprobs

# Run analysis
echo ""
echo "Step 3/3: Running comprehensive analysis..."
python analyze_debug_values.py "$TASK"

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "Check outputs directory for:"
echo "  - debug_values_${TASK}_logodds.csv"
echo "  - debug_values_${TASK}_logprobs.csv"
echo "  - roc_comparison.png"
echo "  - logodds_vs_logprobs_scatter.png"
echo "=========================================="

