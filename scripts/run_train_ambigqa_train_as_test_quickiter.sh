#!/bin/bash
# AmbigQA train-as-test: 9 variants on the full ambigqa training set
# (data/ambigqa/with_negatives/train.csv, 7992 items) used as BOTH train and
# test. Mirror of run_train_rosch_online_quickiter.sh; only the model name
# stays the same (gemma-2-2b), the task and timing differ. Memorization
# probe — DO NOT report cross-task generalization numbers from this run.
#
# All runs use:
#   - gemma-2-2b (full fine-tuning, no LoRA)
#   - g-mode (train generator scores to match validator ranking)
#   - --force-same-x (pairs only within the same question)
#   - 3 epochs, delta=0.15, total_samples=5110 (script default)
#   - --no-upload-hf (these are disposable)
#   - --save_steps 999 (save only at last epoch + epoch 0)
#
# Variants (numbering matches the rosch quick-iter scripts and ambigqa eval
# scripts to come):
#   1. RankAlign baseline             (no TC, no online; pref-only)
#   2. RankAlign + offline self-TC
#   3. RankAlign + online self-TC
#   4. RankAlign + online pair sel.
#   5. RankAlign + BOTH online (self)
#   6. SFT (NLL only, on ALL 7992)
#   7. RankAlign + offline neg-TC
#   8. RankAlign + online neg-TC
#   9. RankAlign + BOTH online (neg)
#
# Approximate timing (gemma-2-2b full FT, batch_size=4 from task registry,
# 5110 pairs * 3 epochs / 4 ~= 3830 steps; longer prompts than rosch):
#   - offline runs:  ~2-3 h  -> reserve 4 h
#   - online-TC:     ~5-6 h  -> reserve 8 h
# Bump up if --time hits walltime; the runs ARE allowed to redo, so partial
# completes are recoverable by relaunching the missing variants.
#
# Usage:
#   bash scripts/run_train_ambigqa_train_as_test_quickiter.sh

set -e

MODEL=google/gemma-2-2b
TASK=ambigqa-train-as-test
NUM_EPOCHS=3
DELTA=0.15
SAVE_STEPS=999
MODELS_DIR=../models-quickiter

mkdir -p "$(dirname "$0")/$MODELS_DIR"

# Pref-only base flags (used for runs 1-5, 7-9).
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

# SFT (run 6): same flags, NLL=1, pref=0.
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

echo "[1/9] RankAlign baseline (ambigqa, 2b)"
run 1 4 "python ranking_loss_ref_online.py $PREF_BASE"

echo "[2/9] RankAlign + offline self-TC (ambigqa, 2b)"
run 1 4 "python ranking_loss_ref_online.py $PREF_BASE --self-typicality"

echo "[3/9] RankAlign + ONLINE self-TC (ambigqa, 2b)"
run 1 8 "python ranking_loss_ref_online.py $PREF_BASE --self-typicality --online-typicality"

echo "[4/9] RankAlign + ONLINE pair selection (ambigqa, 2b)"
run 1 4 "python ranking_loss_ref_online.py $PREF_BASE --online-pair-selection"

echo "[5/9] RankAlign + ONLINE self-TC + ONLINE pair selection (ambigqa, 2b)"
run 1 8 "python ranking_loss_ref_online.py $PREF_BASE --self-typicality --online-typicality --online-pair-selection"

echo "[6/9] SFT-on-all (ambigqa, 2b)"
run 1 4 "python ranking_loss_ref_online.py $SFT_BASE"

echo "[7/9] RankAlign + offline neg-TC (ambigqa, 2b)"
run 1 4 "python ranking_loss_ref_online.py $PREF_BASE --neg-typicality"

echo "[8/9] RankAlign + ONLINE neg-TC (ambigqa, 2b)"
run 1 8 "python ranking_loss_ref_online.py $PREF_BASE --neg-typicality --online-typicality"

echo "[9/9] RankAlign + ONLINE neg-TC + ONLINE pair selection (ambigqa, 2b)"
run 1 8 "python ranking_loss_ref_online.py $PREF_BASE --neg-typicality --online-typicality --online-pair-selection"

echo ""
echo "All 9 jobs submitted (ambigqa-train-as-test, gemma-2-2b)."
