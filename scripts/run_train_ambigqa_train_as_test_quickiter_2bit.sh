#!/bin/bash
# AmbigQA train-as-test (gemma-2-2b-IT). Mirror of
# run_train_ambigqa_train_as_test_quickiter.sh; same 9 variants, same task
# (ambigqa-train-as-test), only the model differs. See the 2b version's
# header comment for variant descriptions and timing notes.
#
# Memorization probe — DO NOT report cross-task generalization numbers.

set -e

MODEL=google/gemma-2-2b-it
TASK=ambigqa-train-as-test
NUM_EPOCHS=3
DELTA=0.15
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

SFT_BASE=" \
    --model $MODEL \
    --task $TASK \
    --train_g_or_d g \
    --split_type random \
    --num_epochs $NUM_EPOCHS \
    --delta $DELTA \
    --save_steps $SAVE_STEPS \
    --all \
    --force-same-x \
    --nll_validator_weight 1 \
    --nll_generator_weight 1 \
    --preference_loss_weight 0 \
    --models-dir $MODELS_DIR \
    --no-upload-hf"

cd "$(dirname "$0")"

echo "[1/9] RankAlign baseline (ambigqa, 2b-it)"
run 1 4 "python ranking_loss_ref_online.py $PREF_BASE"

echo "[2/9] RankAlign + offline self-TC (ambigqa, 2b-it)"
run 1 4 "python ranking_loss_ref_online.py $PREF_BASE --self-typicality"

echo "[3/9] RankAlign + ONLINE self-TC (ambigqa, 2b-it)"
run 1 8 "python ranking_loss_ref_online.py $PREF_BASE --self-typicality --online-typicality"

echo "[4/9] RankAlign + ONLINE pair selection (ambigqa, 2b-it)"
run 1 4 "python ranking_loss_ref_online.py $PREF_BASE --online-pair-selection"

echo "[5/9] RankAlign + ONLINE self-TC + ONLINE pair selection (ambigqa, 2b-it)"
run 1 8 "python ranking_loss_ref_online.py $PREF_BASE --self-typicality --online-typicality --online-pair-selection"

echo "[6/9] SFT-on-all (ambigqa, 2b-it)"
run 1 4 "python ranking_loss_ref_online.py $SFT_BASE"

echo "[7/9] RankAlign + offline neg-TC (ambigqa, 2b-it)"
run 1 4 "python ranking_loss_ref_online.py $PREF_BASE --neg-typicality"

echo "[8/9] RankAlign + ONLINE neg-TC (ambigqa, 2b-it)"
run 1 8 "python ranking_loss_ref_online.py $PREF_BASE --neg-typicality --online-typicality"

echo "[9/9] RankAlign + ONLINE neg-TC + ONLINE pair selection (ambigqa, 2b-it)"
run 1 8 "python ranking_loss_ref_online.py $PREF_BASE --neg-typicality --online-typicality --online-pair-selection"

echo ""
echo "All 9 jobs submitted (ambigqa-train-as-test, gemma-2-2b-it)."
