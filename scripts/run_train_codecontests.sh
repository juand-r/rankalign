#!/bin/bash
# Training runs for codecontests task on gemma-2-9b-it (LoRA, auto-added).
# Discriminator is zero-shot (no few-shot preamble for code).
#
# 5 variants:
#   1. SFT, labelonly, no force-same-x
#   2. RankAlign: pref-only, semi, no TC, no val log-odds, no force-same-x
#   3. Comb, semi, val log-odds, force-same-x
#   4. Comb, semi, val log-odds, self-TC, force-same-x
#   5. Pref-only, semi, val log-odds, self-TC, force-same-x
#
# Usage:
#   bash scripts/run_train_codecontests.sh                 # default: codecontests (~90 problems, ~3K items)
#   bash scripts/run_train_codecontests.sh codecontests-double  # double: ~180 problems, ~6K items

MODEL=google/gemma-2-9b-it
TASK=${1:-codecontests}
# Char filter in codecontests task is coarse; cap tokenizer length to avoid OOM on 9B.
COMMON="--disc-shots zero --max-seq-len 1024"

# 1. SFT, labelonly, no force-same-x
run 2 16 scripts/run_train_semi.sh $MODEL $TASK sft labelonly 0.1 $COMMON --no-force-same-x

# 2. RankAlign: pref-only, semi, no TC, no val log-odds, no force-same-x
run 2 16 scripts/run_train_semi.sh $MODEL $TASK pref-only semi 0.1 $COMMON --no-force-same-x

# 3. Comb, semi, val log-odds, force-same-x
run 2 16 scripts/run_train_semi.sh $MODEL $TASK comb semi 0.1 $COMMON --log-odds

# 4. Comb, semi, val log-odds, self-TC, force-same-x
run 2 16 scripts/run_train_semi.sh $MODEL $TASK comb semi 0.1 $COMMON --self-typcorr --log-odds

# 5. Pref-only, semi, val log-odds, self-TC, force-same-x
run 2 16 scripts/run_train_semi.sh $MODEL $TASK pref-only semi 0.1 $COMMON --self-typcorr --log-odds
