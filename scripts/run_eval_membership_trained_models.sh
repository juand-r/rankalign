#!/bin/bash
# Evaluate membership-sans-rosch-v0-trained models on rosch test tasks.
# All evals use --self-typcorr --log-odds.
#
# Models trained with: google/gemma-2-9b-it, task=membership-sans-rosch-v0, LoRA (_merged).
# See run_train_membership.sh for the 5 training variants.
#
# Usage: bash scripts/run_eval_membership_trained_models.sh

TASKS_ROSCH="rosch-bird rosch-carpenters-tool rosch-clothing rosch-fruit rosch-furniture rosch-sport rosch-toy rosch-vehicle rosch-vegetable rosch-weapon"

# Common eval flags
EVAL="--self-typcorr --log-odds"

# Model path base (epoch 2, LoRA merged)
B=../models/v6-google--gemma-2-9b-it-delta0.15-epoch2--membership-sans-rosch-v0-all--d2g--random--alpha1.0

# ============================================================
# 1. SFT baseline (labelonly, no force-same-x)
# ============================================================
run 1 2 scripts/run_eval_semi.sh \
    ${B}--full-completion--pref0.0--nllv1.0--nllg1.0--labelonly0.1_merged \
    $EVAL -- $TASKS_ROSCH

# ============================================================
# 2. RankAlign (pref-only, semi, no force-same-x)
# ============================================================
run 1 2 scripts/run_eval_semi.sh \
    ${B}--full-completion--semi0.1_merged \
    $EVAL -- $TASKS_ROSCH

# ============================================================
# 3. Comb + log-odds + force-same-x
# ============================================================
run 1 2 scripts/run_eval_semi.sh \
    ${B}--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1_merged \
    $EVAL -- $TASKS_ROSCH

# ============================================================
# 4. Comb + log-odds + self-TC + force-same-x
# ============================================================
run 1 2 scripts/run_eval_semi.sh \
    ${B}--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1_merged \
    $EVAL -- $TASKS_ROSCH

# ============================================================
# 5. Pref-only + log-odds + self-TC + force-same-x
# ============================================================
run 1 2 scripts/run_eval_semi.sh \
    ${B}--tc-self--full-completion--force-same-x--vallogodds--semi0.1_merged \
    $EVAL -- $TASKS_ROSCH
