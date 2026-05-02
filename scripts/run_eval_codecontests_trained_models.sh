#!/bin/bash
# Evaluate codecontests-trained models on the 70 "short" test tasks.
# All evals use --self-typcorr --log-odds --disc-shots-zero.
#
# Models trained with: google/gemma-2-9b-it, task=codecontests, LoRA (_merged).
# See run_train_codecontests.sh for the 5 training variants.
#
# Usage: bash scripts/run_eval_codecontests_trained_models.sh

TASKS_CC="codecontests-1575a codecontests-1575d codecontests-1575j codecontests-1575k \
codecontests-1579a codecontests-1579b codecontests-1579d \
codecontests-1581a codecontests-1581b \
codecontests-1582a codecontests-1582b codecontests-1582c codecontests-1582d \
codecontests-1582e codecontests-1582f1 codecontests-1582f2 \
codecontests-1586a codecontests-1586b codecontests-1586f \
codecontests-1591a codecontests-1591b \
codecontests-1594a codecontests-1594b codecontests-1594c codecontests-1594e1 codecontests-1594f \
codecontests-1598a codecontests-1598c \
codecontests-1599a codecontests-1599c codecontests-1599h \
codecontests-1600e \
codecontests-1601a \
codecontests-1604a codecontests-1604b codecontests-1604c codecontests-1604d codecontests-1604e \
codecontests-1606a codecontests-1606b codecontests-1606c \
codecontests-1607a codecontests-1607b codecontests-1607c codecontests-1607d \
codecontests-1608a \
codecontests-1613a codecontests-1613b codecontests-1613c \
codecontests-1615a codecontests-1615b \
codecontests-1617a codecontests-1617b codecontests-1617c \
codecontests-1618a codecontests-1618b codecontests-1618c codecontests-1618d codecontests-1618e \
codecontests-1619a codecontests-1619b codecontests-1619e \
codecontests-1620a codecontests-1620b codecontests-1620e \
codecontests-1622a codecontests-1622b codecontests-1622c \
codecontests-1623a codecontests-1623c"

# Common eval flags
EVAL="--self-typcorr --log-odds --disc-shots-zero"

# Model path base (epoch 2, LoRA merged)
B=../models/v6-google--gemma-2-9b-it-delta0.15-epoch2--codecontests-all--d2g--random--alpha1.0

# ============================================================
# 1. SFT baseline (labelonly, no force-same-x)
# ============================================================
run 1 8 scripts/run_eval_semi.sh \
    ${B}--full-completion--pref0.0--nllv1.0--nllg1.0--labelonly0.1_merged \
    $EVAL -- $TASKS_CC

# ============================================================
# 2. RankAlign (pref-only, semi, no force-same-x)
# ============================================================
run 1 8 scripts/run_eval_semi.sh \
    ${B}--full-completion--semi0.1_merged \
    $EVAL -- $TASKS_CC

# ============================================================
# 3. Comb + log-odds + force-same-x
# ============================================================
run 1 8 scripts/run_eval_semi.sh \
    ${B}--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1_merged \
    $EVAL -- $TASKS_CC

# ============================================================
# 4. Comb + log-odds + self-TC + force-same-x
# ============================================================
run 1 8 scripts/run_eval_semi.sh \
    ${B}--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1_merged \
    $EVAL -- $TASKS_CC

# ============================================================
# 5. Pref-only + log-odds + self-TC + force-same-x
# ============================================================
run 1 8 scripts/run_eval_semi.sh \
    ${B}--tc-self--full-completion--force-same-x--vallogodds--semi0.1_merged \
    $EVAL -- $TASKS_CC
