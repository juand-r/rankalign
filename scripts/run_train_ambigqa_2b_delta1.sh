#!/bin/bash
# AmbigQA train-as-test, gemma-2-2b, RankAlign baseline ONLY, delta=1.0.
#
# Why delta=1.0:
#   The base validator on ambigqa 2b is well-calibrated but pair-correctness
#   is only 58.82% with delta=0.15 because the bottom |Δv| buckets are at
#   chance. Raising delta to 1.0 keeps only the high-confidence buckets,
#   predicted pair-correctness ≈ 92%. See:
#     docs/ambigqa_validator_calibration_analysis.md
#     docs/diag_pair_correctness_report.md
#
# This is a single diagnostic run: if gen-ROC jumps materially above the
# delta=0.15 RankAlign baseline (71.82) toward SFT (≈92), the validator-
# quality hypothesis is confirmed and we'll follow up with TC variants
# under the same delta.

set -e

MODEL=google/gemma-2-2b
TASK=ambigqa-train-as-test
NUM_EPOCHS=3
DELTA=1.0
SAVE_STEPS=999
MODELS_DIR=../models-quickiter

mkdir -p "$(dirname "$0")/$MODELS_DIR"

PREF_BASE=" \
    --model $MODEL \
    --task $TASK \
    --train_g_or_d g \
    --split_type random \
    --num_epochs $NUM_EPOCHS \
    --delta $DELTA \
    --save_steps $SAVE_STEPS \
    --all \
    --force-same-x \
    --nll_validator_weight 0 \
    --nll_generator_weight 0 \
    --preference_loss_weight 1 \
    --models-dir $MODELS_DIR \
    --no-upload-hf"

cd "$(dirname "$0")"

echo "[1/1] RankAlign baseline (ambigqa, 2b, delta=1.0)"
run 1 4 "python ranking_loss_ref_online.py $PREF_BASE"

echo ""
echo "Submitted. Watch with: tail -f ~/logs/<JOBID>.out"
