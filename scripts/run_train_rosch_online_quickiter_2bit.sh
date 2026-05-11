#!/bin/bash
# 2b-it parallel of run_train_rosch_online_quickiter.sh.
# Same 9 variants on rosch-furniture-and-bird (train=test=186 items),
# but using google/gemma-2-2b-IT (instruction-tuned). The training
# script auto-detects '-it' in the model name and applies the chat
# template (with_chat=True). Gemma-it has no system role.
#
# Save dirs include the model name, so they won't collide with the
# bare 2b runs. No LoRA (2b is small enough for full FT).
#
# Variants (numbering matches the bare-2b script and eval scripts):
#   1. RankAlign baseline             (no TC, no online; pref-only)
#   2. RankAlign + offline self-TC
#   3. RankAlign + online self-TC
#   4. RankAlign + online pair sel.
#   5. RankAlign + BOTH online (self)
#   6. SFT (NLL only, on ALL 186)
#   7. RankAlign + offline neg-TC
#   8. RankAlign + online neg-TC
#   9. RankAlign + BOTH online (neg)
#
# Usage:
#   bash scripts/run_train_rosch_online_quickiter_2bit.sh

set -e

MODEL=google/gemma-2-2b-it
TASK=rosch-furniture-and-bird
NUM_EPOCHS=3
DELTA=0.15
SAVE_STEPS=999
TOTAL_SAMPLES=1500
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
    --total_samples $TOTAL_SAMPLES \
    --save_steps $SAVE_STEPS \
    --all \
    --force-same-x \
    --nll_validator_weight 0 \
    --nll_generator_weight 0 \
    --preference_loss_weight 1 \
    --models-dir $MODELS_DIR \
    --no-upload-hf"

# SFT (run 6) — NLL=1, pref=0.
SFT_BASE=" \
    --model $MODEL \
    --task $TASK \
    --train_g_or_d g \
    --split_type random \
    --num_epochs $NUM_EPOCHS \
    --delta $DELTA \
    --total_samples $TOTAL_SAMPLES \
    --save_steps $SAVE_STEPS \
    --all \
    --force-same-x \
    --nll_validator_weight 1 \
    --nll_generator_weight 1 \
    --preference_loss_weight 0 \
    --models-dir $MODELS_DIR \
    --no-upload-hf"

cd "$(dirname "$0")"

echo "[1/9] RankAlign baseline (2b-it)"
run 1 1 "python ranking_loss_ref_online.py $PREF_BASE"

echo "[2/9] RankAlign + offline self-TC (2b-it)"
run 1 1 "python ranking_loss_ref_online.py $PREF_BASE --self-typicality"

echo "[3/9] RankAlign + ONLINE self-TC (2b-it)"
run 1 2 "python ranking_loss_ref_online.py $PREF_BASE --self-typicality --online-typicality"

echo "[4/9] RankAlign + ONLINE pair selection (2b-it)"
run 1 1 "python ranking_loss_ref_online.py $PREF_BASE --online-pair-selection"

echo "[5/9] RankAlign + ONLINE self-TC + ONLINE pair selection (2b-it)"
run 1 2 "python ranking_loss_ref_online.py $PREF_BASE --self-typicality --online-typicality --online-pair-selection"

echo "[6/9] SFT-on-all (2b-it)"
run 1 1 "python ranking_loss_ref_online.py $SFT_BASE"

echo "[7/9] RankAlign + offline neg-TC (2b-it)"
run 1 1 "python ranking_loss_ref_online.py $PREF_BASE --neg-typicality"

echo "[8/9] RankAlign + ONLINE neg-TC (2b-it)"
run 1 2 "python ranking_loss_ref_online.py $PREF_BASE --neg-typicality --online-typicality"

echo "[9/9] RankAlign + ONLINE neg-TC + ONLINE pair selection (2b-it)"
run 1 2 "python ranking_loss_ref_online.py $PREF_BASE --neg-typicality --online-typicality --online-pair-selection"

echo ""
echo "All 9 jobs submitted (gemma-2-2b-it)."
