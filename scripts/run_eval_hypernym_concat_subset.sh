#!/bin/bash

# Run eval.py on hypernym-concat-bananas-to-dogs
# Runs 4 variants in parallel, one on each GPU
#
# Usage: ./run_eval_hypernym_concat_subset.sh <GPU_LIST> [--full-completion] [--model-full-completion]
# Example: ./run_eval_hypernym_concat_subset.sh 0,1,2,3
# Example: ./run_eval_hypernym_concat_subset.sh 0,1,2,3 --full-completion --model-full-completion

GPU_LIST=$1
shift  # Remove first argument (GPU list)

# Parse optional flags
EVAL_FULL_COMPLETION=""
MODEL_FULL_COMPLETION=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --full-completion)
            EVAL_FULL_COMPLETION="--use_full_completion_logprobs"
            shift
            ;;
        --model-full-completion)
            MODEL_FULL_COMPLETION="--full-completion"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$GPU_LIST" ]; then
    echo "Usage: $0 <GPU_LIST> [--full-completion] [--model-full-completion]"
    echo "  GPU_LIST: comma-separated GPU IDs (e.g., 0,1,2,3) - need 4 GPUs"
    echo "  --full-completion       Pass --use-full-completion to eval.py"
    echo "  --model-full-completion Look for models with --full-completion in name"
    exit 1
fi

# Parse GPU list into array
IFS=',' read -ra GPUS <<< "$GPU_LIST"

if [ ${#GPUS[@]} -lt 4 ]; then
    echo "Error: Need at least 4 GPUs for the 4 variants"
    echo "  Provided: ${#GPUS[@]} GPUs (${GPU_LIST})"
    exit 1
fi

EPOCH=1
TASK="hypernym-concat-bananas-to-dogs"

echo "========================================"
echo "Running eval on $TASK (4 variants in parallel)"
echo "  GPUs: ${GPUS[0]}, ${GPUS[1]}, ${GPUS[2]}, ${GPUS[3]}"
echo "  EVAL_FULL_COMPLETION: $EVAL_FULL_COMPLETION"
echo "  MODEL_FULL_COMPLETION: $MODEL_FULL_COMPLETION"
echo "========================================"

# d2g (delta=0.15), no typcorr - GPU 0
MODEL="../models/v5-google--gemma-2-2b-delta0.15-epoch${EPOCH}--${TASK}-all--d2g--random--alpha1.0${MODEL_FULL_COMPLETION}--nllv1.0--nllg1.0"
echo "[GPU ${GPUS[0]}] Evaluating: $MODEL"
CUDA_VISIBLE_DEVICES=${GPUS[0]} python eval.py \
    --model "$MODEL" \
    --task "$TASK" \
    --split_type random \
    --viz \
    $EVAL_FULL_COMPLETION &
PID1=$!

# g2d (delta=2.5), no typcorr - GPU 1
MODEL="../models/v5-google--gemma-2-2b-delta2.5-epoch${EPOCH}--${TASK}-all--g2d--random--alpha1.0${MODEL_FULL_COMPLETION}--nllv1.0--nllg1.0"
echo "[GPU ${GPUS[1]}] Evaluating: $MODEL"
CUDA_VISIBLE_DEVICES=${GPUS[1]} python eval.py \
    --model "$MODEL" \
    --task "$TASK" \
    --split_type random \
    --viz \
    $EVAL_FULL_COMPLETION &
PID2=$!

# d2g (delta=0.15), with typcorr - GPU 2
MODEL="../models/v5-google--gemma-2-2b-delta0.15-epoch${EPOCH}--${TASK}-all--d2g--random--alpha1.0--typcorr${MODEL_FULL_COMPLETION}--nllv1.0--nllg1.0"
echo "[GPU ${GPUS[2]}] Evaluating: $MODEL"
CUDA_VISIBLE_DEVICES=${GPUS[2]} python eval.py \
    --model "$MODEL" \
    --task "$TASK" \
    --split_type random \
    --typicality-correction \
    --viz \
    $EVAL_FULL_COMPLETION &
PID3=$!

# g2d (delta=2.5), with typcorr - GPU 3
MODEL="../models/v5-google--gemma-2-2b-delta2.5-epoch${EPOCH}--${TASK}-all--g2d--random--alpha1.0--typcorr${MODEL_FULL_COMPLETION}--nllv1.0--nllg1.0"
echo "[GPU ${GPUS[3]}] Evaluating: $MODEL"
CUDA_VISIBLE_DEVICES=${GPUS[3]} python eval.py \
    --model "$MODEL" \
    --task "$TASK" \
    --split_type random \
    --typicality-correction \
    --viz \
    $EVAL_FULL_COMPLETION &
PID4=$!

# Wait for all background jobs to complete
echo ""
echo "Waiting for all 4 evaluations to complete..."
wait $PID1 $PID2 $PID3 $PID4

echo ""
echo "All evaluations on $TASK completed!"
