#!/bin/bash
# Eval runs for the 16 plausibleqa semi-supervised models (epoch 2).
# All use eval_by_claude.py with --self-typicality --validator-log-odds.
# 100 plausibleqa test tasks per model, ~2 hours each.
#
# Usage: source this file, or copy TASKS= line first, then paste run commands.

TASKS="plausibleqa-nq_1114 plausibleqa-nq_1324 plausibleqa-nq_1328 plausibleqa-nq_1369 plausibleqa-nq_1394 plausibleqa-nq_1438 plausibleqa-nq_1663 plausibleqa-nq_2031 plausibleqa-nq_207 plausibleqa-nq_2174 plausibleqa-nq_2281 plausibleqa-nq_2421 plausibleqa-nq_2436 plausibleqa-nq_2535 plausibleqa-nq_2622 plausibleqa-nq_2637 plausibleqa-nq_2759 plausibleqa-nq_2824 plausibleqa-nq_2856 plausibleqa-nq_2867 plausibleqa-nq_2876 plausibleqa-nq_3004 plausibleqa-nq_3015 plausibleqa-nq_3068 plausibleqa-nq_3099 plausibleqa-nq_3127 plausibleqa-nq_3137 plausibleqa-nq_316 plausibleqa-nq_3276 plausibleqa-nq_54 plausibleqa-nq_562 plausibleqa-nq_709 plausibleqa-nq_958 plausibleqa-trivia_1655 plausibleqa-trivia_2984 plausibleqa-trivia_3035 plausibleqa-trivia_3043 plausibleqa-trivia_3180 plausibleqa-trivia_3245 plausibleqa-trivia_3433 plausibleqa-trivia_3492 plausibleqa-trivia_3599 plausibleqa-trivia_4009 plausibleqa-trivia_4234 plausibleqa-trivia_4489 plausibleqa-trivia_4697 plausibleqa-trivia_5003 plausibleqa-trivia_560 plausibleqa-trivia_5675 plausibleqa-trivia_6317 plausibleqa-trivia_6777 plausibleqa-trivia_7272 plausibleqa-trivia_7579 plausibleqa-trivia_9589 plausibleqa-webq_1000 plausibleqa-webq_1046 plausibleqa-webq_1086 plausibleqa-webq_1097 plausibleqa-webq_1163 plausibleqa-webq_1187 plausibleqa-webq_1278 plausibleqa-webq_1307 plausibleqa-webq_1310 plausibleqa-webq_1338 plausibleqa-webq_134 plausibleqa-webq_1383 plausibleqa-webq_141 plausibleqa-webq_1421 plausibleqa-webq_1442 plausibleqa-webq_1476 plausibleqa-webq_1498 plausibleqa-webq_15 plausibleqa-webq_1584 plausibleqa-webq_1613 plausibleqa-webq_1668 plausibleqa-webq_1714 plausibleqa-webq_1723 plausibleqa-webq_1836 plausibleqa-webq_1972 plausibleqa-webq_212 plausibleqa-webq_299 plausibleqa-webq_342 plausibleqa-webq_373 plausibleqa-webq_428 plausibleqa-webq_435 plausibleqa-webq_520 plausibleqa-webq_611 plausibleqa-webq_650 plausibleqa-webq_669 plausibleqa-webq_672 plausibleqa-webq_713 plausibleqa-webq_744 plausibleqa-webq_749 plausibleqa-webq_760 plausibleqa-webq_77 plausibleqa-webq_803 plausibleqa-webq_84 plausibleqa-webq_88 plausibleqa-webq_882 plausibleqa-webq_898"

M=../models/v6-google--gemma-2-2b-delta0.15-epoch2--plausibleqa-all--d2g--random--alpha1.0

# ============================================================
# PLAIN (6 models)
# ============================================================

# --- plain, pref-only labelonly (no log-odds) ---
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh $M--full-completion--force-same-x--labelonly0.1 --self-typcorr --log-odds -- $TASKS

# --- plain, pref-only labelonly (log-odds) ---
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh $M--full-completion--force-same-x--vallogodds--labelonly0.1 --self-typcorr --log-odds -- $TASKS

# --- plain, comb labelonly (log-odds) ---
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh $M--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--labelonly0.1 --self-typcorr --log-odds -- $TASKS

# --- plain, comb semi (log-odds) ---
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh $M--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1 --self-typcorr --log-odds -- $TASKS

# --- plain, sft labelonly ---
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh $M--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x--labelonly0.1 --self-typcorr --log-odds -- $TASKS

# --- plain, sft semi ---
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh $M--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x--semi0.1 --self-typcorr --log-odds -- $TASKS

# ============================================================
# TC-SELF (5 models)
# ============================================================

# --- tc-self, pref-only labelonly (no log-odds) ---
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh $M--tc-self--full-completion--force-same-x--labelonly0.1 --self-typcorr --log-odds -- $TASKS

# --- tc-self, pref-only labelonly (log-odds) ---
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh $M--tc-self--full-completion--force-same-x--vallogodds--labelonly0.1 --self-typcorr --log-odds -- $TASKS

# --- tc-self, comb labelonly (log-odds) ---
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh $M--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--labelonly0.1 --self-typcorr --log-odds -- $TASKS

# --- tc-self, comb semi (log-odds) ---
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh $M--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1 --self-typcorr --log-odds -- $TASKS

# --- tc-self, sft semi ---
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh $M--tc-self--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x--semi0.1 --self-typcorr --log-odds -- $TASKS

# ============================================================
# TC-SELF + LENORM (5 models)
# ============================================================

# --- tc-self+lenorm, pref-only labelonly (no log-odds) ---
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh $M--tc-self--lenorm--full-completion--force-same-x--labelonly0.1 --self-typcorr --log-odds -- $TASKS

# --- tc-self+lenorm, pref-only labelonly (log-odds) ---
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh $M--tc-self--lenorm--full-completion--force-same-x--vallogodds--labelonly0.1 --self-typcorr --log-odds -- $TASKS

# --- tc-self+lenorm, comb labelonly (log-odds) ---
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh $M--tc-self--lenorm--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--labelonly0.1 --self-typcorr --log-odds -- $TASKS

# --- tc-self+lenorm, comb semi (log-odds) ---
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh $M--tc-self--lenorm--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1 --self-typcorr --log-odds -- $TASKS

# --- tc-self+lenorm, sft semi ---
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh $M--tc-self--lenorm--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x--semi0.1 --self-typcorr --log-odds -- $TASKS
