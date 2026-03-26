#!/bin/bash
# Eval runs for the 16 ambigqa semi-supervised models (epoch 2).
# All use eval_by_claude.py with --self-typicality --validator-log-odds.
# 17 ambigqa test tasks per model, ~2 hours each.
#
# Usage: source this file, or copy TASKS= line first, then paste run commands.

TASKS="ambigqa-american ambigqa-danube ambigqa-executed ambigqa-gives ambigqa-harry ambigqa-involved ambigqa-jack ambigqa-plays ambigqa-received ambigqa-sang ambigqa-soccer ambigqa-used ambigqa-voice ambigqa-winter ambigqa-won ambigqa-world ambigqa-year"

M=../models/v6-google--gemma-2-2b-delta0.15-epoch2--ambigqa-all--d2g--random--alpha1.0

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
