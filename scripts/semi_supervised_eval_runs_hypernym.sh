#!/bin/bash
# Eval runs for the 16 hypernym-concat-bananas-to-dogs-double semi-supervised models (epoch 2).
# All use eval_by_claude.py with --self-typicality --validator-log-odds.
# 18 individual hypernym test tasks per model (all v2 data, excluding "magnifying glasses" due to space).
#
# Usage: source this file, or copy TASKS= line first, then paste run commands.

TASKS="hypernym-bananas hypernym-bazookas hypernym-cabinets hypernym-cars hypernym-chairs hypernym-crows hypernym-diapers hypernym-dogs hypernym-dolls hypernym-ducklings hypernym-elephants hypernym-guns hypernym-hammers hypernym-helmets hypernym-jackets hypernym-kayaks hypernym-kites hypernym-mirrors"

M=../models/v6-google--gemma-2-2b-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-double-all--d2g--random--alpha1.0

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
