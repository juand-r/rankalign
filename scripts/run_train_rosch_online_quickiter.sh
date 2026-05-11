#!/bin/bash
# Quick-iteration training: 9 runs on rosch-furniture-and-bird (train=test, 186
# items). Tests whether online pair selection / online typicality produce a
# distinguishable training signal in the simplest possible memorization probe.
# See docs/IMPORTANT-RESEARCH-PLAN.md §3a for context.
#
# All runs use:
#   - gemma-2-2b (full fine-tuning, no LoRA)
#   - g-mode (train generator scores to match validator ranking)
#   - --force-same-x (pairs only within the same generator prompt; furniture
#     items only pair with furniture items, bird with bird)
#   - 3 epochs, delta=0.15, total_samples=1500
#   - --no-upload-hf (these are disposable)
#   - --save_steps 999 (save only at last epoch + epoch 0; reduces disk)
#
# Variants (numbering matches eval script):
#   1. RankAlign baseline             (no TC, no online; pref-only)
#   2. RankAlign + offline self-TC    (precomputed TC from initial model)
#   3. RankAlign + online self-TC     (live with grads)
#   4. RankAlign + online pair sel.   (pair winners refreshed each epoch)
#   5. RankAlign + BOTH online (self) (online self-TC + online pair sel)
#   6. SFT (NLL only, on ALL 186)     (no semi/labelonly — train=test means
#                                      "labeled-only" with 10% would be ~18
#                                      items, too few for the probe)
#   7. RankAlign + offline neg-TC     (parallel to #2 but neg-TC)
#   8. RankAlign + online neg-TC      (parallel to #3 but neg-TC)
#   9. RankAlign + BOTH online (neg)  (parallel to #5 but neg-TC)
#
# Usage:
#   bash scripts/run_train_rosch_online_quickiter.sh
#   # or to re-launch a subset, comment out the runs you don't want.

set -e

MODEL=google/gemma-2-2b
TASK=rosch-furniture-and-bird
NUM_EPOCHS=3
DELTA=0.15
SAVE_STEPS=999
TOTAL_SAMPLES=1500   # Subsample from ~4866 valid pairs for fast iteration
MODELS_DIR=../models-quickiter

mkdir -p "$(dirname "$0")/$MODELS_DIR"

# Approximate timing on 1 H100 (gemma-2-2b full FT, batch_size=1):
#   - offline runs:  ~2 it/s × 1500 × 3 = 2250 s = ~40 min  -> reserve 1.5 h
#   - online-TC:     ~1 it/s × 1500 × 3 = 4500 s = ~75 min  -> reserve 2.5 h
#   - online-pairs:  ~40 min + 2 × ~2 min recompute         -> reserve 1.5 h
#   - both online:   ~75 min + 2 × ~2 min recompute         -> reserve 2.5 h
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

# SFT (run 6) — same as PREF_BASE but NLL=1, pref=0.
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

# 1. RankAlign baseline (no TC, no online)
echo "[1/9] RankAlign baseline"
run 1 1 "python ranking_loss_ref_online.py $PREF_BASE"

# 2. RankAlign + offline self-TC (frozen base-TC; identical to ranking_loss_ref.py)
echo "[2/9] RankAlign + offline self-TC"
run 1 1 "python ranking_loss_ref_online.py $PREF_BASE --self-typicality"

# 3. RankAlign + ONLINE self-TC (live, with grads)
echo "[3/9] RankAlign + ONLINE self-TC"
run 1 2 "python ranking_loss_ref_online.py $PREF_BASE --self-typicality --online-typicality"

# 4. RankAlign + ONLINE pair selection (no TC)
echo "[4/9] RankAlign + ONLINE pair selection"
run 1 1 "python ranking_loss_ref_online.py $PREF_BASE --online-pair-selection"

# 5. RankAlign + BOTH online (self-TC + pair selection)
echo "[5/9] RankAlign + ONLINE self-TC + ONLINE pair selection"
run 1 2 "python ranking_loss_ref_online.py $PREF_BASE --self-typicality --online-typicality --online-pair-selection"

# 6. SFT on all 186 (NLL only, no labelonly — see header note)
echo "[6/9] SFT-on-all (NLL only)"
run 1 1 "python ranking_loss_ref_online.py $SFT_BASE"

# 7. RankAlign + offline neg-TC
echo "[7/9] RankAlign + offline neg-TC"
run 1 1 "python ranking_loss_ref_online.py $PREF_BASE --neg-typicality"

# 8. RankAlign + ONLINE neg-TC (live, with grads)
echo "[8/9] RankAlign + ONLINE neg-TC"
run 1 2 "python ranking_loss_ref_online.py $PREF_BASE --neg-typicality --online-typicality"

# 9. RankAlign + BOTH online (neg-TC + pair selection)
echo "[9/9] RankAlign + ONLINE neg-TC + ONLINE pair selection"
run 1 2 "python ranking_loss_ref_online.py $PREF_BASE --neg-typicality --online-typicality --online-pair-selection"

echo ""
echo "All 9 jobs submitted. Check ~/logs/<JOBID>.out for each."
