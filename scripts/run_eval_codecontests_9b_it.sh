#!/bin/bash
# Evaluate base model google/gemma-2-9b-it on CodeContests "short" test tasks.
# Uses run_eval_semi.sh with --self-typcorr --log-odds and zero-shot discriminator
# (codecontests has no few-shot discriminator preamble).
#
# 70 short tasks (median completion < 300 tokens), split into 3 jobs.
#
# Usage: bash scripts/run_eval_codecontests_9b_it.sh

MODEL=google/gemma-2-9b-it

# ============================================================
# BATCH 1 (24 tasks): 1575a - 1594f
# ============================================================
run 1 6 scripts/run_eval_semi.sh $MODEL --self-typcorr --log-odds --disc-shots-zero -- \
    codecontests-1575a codecontests-1575d codecontests-1575j codecontests-1575k \
    codecontests-1579a codecontests-1579b codecontests-1579d \
    codecontests-1581a codecontests-1581b \
    codecontests-1582a codecontests-1582b codecontests-1582c codecontests-1582d \
    codecontests-1582e codecontests-1582f1 codecontests-1582f2 \
    codecontests-1586a codecontests-1586b codecontests-1586f \
    codecontests-1591a codecontests-1591b \
    codecontests-1594a codecontests-1594b codecontests-1594c

# ============================================================
# BATCH 2 (23 tasks): 1594e1 - 1613c
# ============================================================
run 1 6 scripts/run_eval_semi.sh $MODEL --self-typcorr --log-odds --disc-shots-zero -- \
    codecontests-1594e1 codecontests-1594f \
    codecontests-1598a codecontests-1598c \
    codecontests-1599a codecontests-1599c codecontests-1599h \
    codecontests-1600e \
    codecontests-1601a \
    codecontests-1604a codecontests-1604b codecontests-1604c codecontests-1604d codecontests-1604e \
    codecontests-1606a codecontests-1606b codecontests-1606c \
    codecontests-1607a codecontests-1607b codecontests-1607c codecontests-1607d \
    codecontests-1608a \
    codecontests-1613a

# ============================================================
# BATCH 3 (23 tasks): 1613b - 1623c
# ============================================================
run 1 6 scripts/run_eval_semi.sh $MODEL --self-typcorr --log-odds --disc-shots-zero -- \
    codecontests-1613b codecontests-1613c \
    codecontests-1615a codecontests-1615b \
    codecontests-1617a codecontests-1617b codecontests-1617c \
    codecontests-1618a codecontests-1618b codecontests-1618c codecontests-1618d codecontests-1618e \
    codecontests-1619a codecontests-1619b codecontests-1619e \
    codecontests-1620a codecontests-1620b codecontests-1620e \
    codecontests-1622a codecontests-1622b codecontests-1622c \
    codecontests-1623a codecontests-1623c
