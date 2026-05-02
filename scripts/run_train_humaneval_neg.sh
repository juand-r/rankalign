#!/bin/bash
# Training runs for humaneval task on gemma-2-9b-it with NEG-typicality
# correction (LoRA, auto-added). Discriminator is zero-shot. Uses 2 GPUs.
#
# Sibling of run_train_humaneval.sh: launches only the TC-using variants
# with --neg-typcorr instead of --self-typcorr. The non-TC variants
# (sft labelonly, pref-only semi vanilla, comb semi log-odds) live in
# run_train_humaneval.sh and don't need to be re-run.
#
# Variants:
#   1. Comb, semi, val log-odds, neg-TC, force-same-x
#   2. Pref-only, semi, val log-odds, neg-TC, force-same-x
#   3. Pref-only, semi, val log-odds, neg-TC, no force-same-x
#
# Usage:
#   bash scripts/run_train_humaneval_neg.sh

MODEL=google/gemma-2-9b-it
TASK=humaneval
COMMON="--disc-shots zero --max-seq-len 1024"

# 1. Comb, semi, val log-odds, neg-TC, force-same-x
run 2 16 scripts/run_train_semi.sh $MODEL $TASK comb semi 0.1 $COMMON --neg-typcorr --log-odds

# 2. Pref-only, semi, val log-odds, neg-TC, force-same-x
run 2 16 scripts/run_train_semi.sh $MODEL $TASK pref-only semi 0.1 $COMMON --neg-typcorr --log-odds

# 3. Pref-only, semi, val log-odds, neg-TC, no force-same-x (RankAlign-like w/ neg-TC)
run 2 16 scripts/run_train_semi.sh $MODEL $TASK pref-only semi 0.1 $COMMON --neg-typcorr --no-force-same-x
