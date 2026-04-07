#!/bin/bash
# Eval runs for the 16 ifeval-concat semi-supervised models.
# All use eval_by_claude.py with --self-typicality --validator-log-odds.
# 99 ifeval test tasks per model.
#
# NOTE: label-only models use epoch2, semi models use epoch1
#       (semi training timed out before epoch 3).
#
# Usage: source this file, or copy TASKS= line first, then paste run commands.

TASKS="ifeval-prompt_1 ifeval-prompt_2 ifeval-prompt_3 ifeval-prompt_4 ifeval-prompt_5 ifeval-prompt_6 ifeval-prompt_7 ifeval-prompt_8 ifeval-prompt_9 ifeval-prompt_10 ifeval-prompt_11 ifeval-prompt_12 ifeval-prompt_13 ifeval-prompt_15 ifeval-prompt_16 ifeval-prompt_17 ifeval-prompt_18 ifeval-prompt_19 ifeval-prompt_20 ifeval-prompt_21 ifeval-prompt_22 ifeval-prompt_23 ifeval-prompt_24 ifeval-prompt_25 ifeval-prompt_26 ifeval-prompt_27 ifeval-prompt_28 ifeval-prompt_29 ifeval-prompt_30 ifeval-prompt_32 ifeval-prompt_33 ifeval-prompt_34 ifeval-prompt_35 ifeval-prompt_36 ifeval-prompt_37 ifeval-prompt_38 ifeval-prompt_39 ifeval-prompt_40 ifeval-prompt_41 ifeval-prompt_42 ifeval-prompt_43 ifeval-prompt_44 ifeval-prompt_45 ifeval-prompt_46 ifeval-prompt_47 ifeval-prompt_48 ifeval-prompt_49 ifeval-prompt_50 ifeval-prompt_51 ifeval-prompt_52 ifeval-prompt_53 ifeval-prompt_54 ifeval-prompt_56 ifeval-prompt_57 ifeval-prompt_58 ifeval-prompt_59 ifeval-prompt_61 ifeval-prompt_63 ifeval-prompt_64 ifeval-prompt_65 ifeval-prompt_66 ifeval-prompt_67 ifeval-prompt_68 ifeval-prompt_70 ifeval-prompt_72 ifeval-prompt_73 ifeval-prompt_74 ifeval-prompt_75 ifeval-prompt_76 ifeval-prompt_77 ifeval-prompt_78 ifeval-prompt_79 ifeval-prompt_80 ifeval-prompt_82 ifeval-prompt_83 ifeval-prompt_84 ifeval-prompt_85 ifeval-prompt_87 ifeval-prompt_88 ifeval-prompt_89 ifeval-prompt_90 ifeval-prompt_91 ifeval-prompt_92 ifeval-prompt_93 ifeval-prompt_94 ifeval-prompt_95 ifeval-prompt_96 ifeval-prompt_97 ifeval-prompt_98 ifeval-prompt_99 ifeval-prompt_100 ifeval-prompt_102 ifeval-prompt_103 ifeval-prompt_104 ifeval-prompt_105 ifeval-prompt_106 ifeval-prompt_107 ifeval-prompt_108 ifeval-prompt_109"

# Label-only models: epoch 2
MLO=../models/v6-google--gemma-2-2b-delta0.15-epoch2--ifeval-concat-all--d2g--random--alpha1.0

# Semi models: epoch 1 (epoch 2 not yet trained)
MSE=../models/v6-google--gemma-2-2b-delta0.15-epoch1--ifeval-concat-all--d2g--random--alpha1.0

# ============================================================
# PLAIN (6 models)
# ============================================================

# --- plain, pref-only labelonly (no log-odds) ---
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh $MLO--full-completion--force-same-x--labelonly0.1 --self-typcorr --log-odds --disc-shots-zero -- $TASKS

# --- plain, pref-only labelonly (log-odds) ---
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh $MLO--full-completion--force-same-x--vallogodds--labelonly0.1 --self-typcorr --log-odds --disc-shots-zero -- $TASKS

# --- plain, comb labelonly (log-odds) ---
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh $MLO--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--labelonly0.1 --self-typcorr --log-odds --disc-shots-zero -- $TASKS

# --- plain, comb semi (log-odds) ---
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh $MSE--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1 --self-typcorr --log-odds --disc-shots-zero -- $TASKS

# --- plain, sft labelonly ---
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh $MLO--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x--labelonly0.1 --self-typcorr --log-odds --disc-shots-zero -- $TASKS

# --- plain, sft semi ---
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh $MSE--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x--semi0.1 --self-typcorr --log-odds --disc-shots-zero -- $TASKS

# ============================================================
# TC-SELF (5 models)
# ============================================================

# --- tc-self, pref-only labelonly (no log-odds) ---
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh $MLO--tc-self--full-completion--force-same-x--labelonly0.1 --self-typcorr --log-odds --disc-shots-zero -- $TASKS

# --- tc-self, pref-only labelonly (log-odds) ---
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh $MLO--tc-self--full-completion--force-same-x--vallogodds--labelonly0.1 --self-typcorr --log-odds --disc-shots-zero -- $TASKS

# --- tc-self, comb labelonly (log-odds) ---
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh $MLO--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--labelonly0.1 --self-typcorr --log-odds --disc-shots-zero -- $TASKS

# --- tc-self, comb semi (log-odds) ---
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh $MSE--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1 --self-typcorr --log-odds --disc-shots-zero -- $TASKS

# --- tc-self, sft semi ---
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh $MSE--tc-self--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x--semi0.1 --self-typcorr --log-odds --disc-shots-zero -- $TASKS

# ============================================================
# TC-SELF + LENORM (5 models)
# ============================================================

# --- tc-self+len, pref-only labelonly (no log-odds) ---
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh $MLO--tc-self--lenorm--full-completion--force-same-x--labelonly0.1 --self-typcorr --log-odds --disc-shots-zero -- $TASKS

# --- tc-self+len, pref-only labelonly (log-odds) ---
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh $MLO--tc-self--lenorm--full-completion--force-same-x--vallogodds--labelonly0.1 --self-typcorr --log-odds --disc-shots-zero -- $TASKS

# --- tc-self+len, comb labelonly (log-odds) ---
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh $MLO--tc-self--lenorm--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--labelonly0.1 --self-typcorr --log-odds --disc-shots-zero -- $TASKS

# --- tc-self+len, comb semi (log-odds) ---
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh $MSE--tc-self--lenorm--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1 --self-typcorr --log-odds --disc-shots-zero -- $TASKS

# --- tc-self+len, sft semi ---
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh $MSE--tc-self--lenorm--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x--semi0.1 --self-typcorr --log-odds --disc-shots-zero -- $TASKS
