#!/bin/bash
# All semi-supervised / labeled-only training runs.
# Each line is a standalone `run` command.
# Default: mode=g (delta=0.15), ratio=0.1, split-seed=42
#
# Recipes:
#   (1) plain:              comb gets --log-odds, sft plain, pref both
#   (2) tc-self:            + --self-typcorr
#   (3) tc-self + lenorm:   + --self-typcorr --lenorm
#   (4) tc-online:          COMMENTED OUT
#   (5) tc-online + lenorm: COMMENTED OUT

# ============================================================
# PLAUSIBLEQA
# ============================================================

# --- plain ---
run 1 3 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa comb semi 0.1 --log-odds
run 1 3 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa sft semi 0.1
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa comb labelonly 0.1 --log-odds
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa sft labelonly 0.1
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa pref-only labelonly 0.1 --log-odds
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa pref-only labelonly 0.1

# --- tc-self ---
run 1 3 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa comb semi 0.1 --self-typcorr --log-odds
run 1 3 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa sft semi 0.1 --self-typcorr
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa comb labelonly 0.1 --self-typcorr --log-odds
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa sft labelonly 0.1 --self-typcorr
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa pref-only labelonly 0.1 --self-typcorr --log-odds
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa pref-only labelonly 0.1 --self-typcorr

# --- tc-self + lenorm ---
run 1 3 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa comb semi 0.1 --self-typcorr --lenorm --log-odds
run 1 3 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa sft semi 0.1 --self-typcorr --lenorm
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa comb labelonly 0.1 --self-typcorr --lenorm --log-odds
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa sft labelonly 0.1 --self-typcorr --lenorm
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa pref-only labelonly 0.1 --self-typcorr --lenorm --log-odds
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa pref-only labelonly 0.1 --self-typcorr --lenorm

# --- tc-online (commented out) ---
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa comb semi 0.1 --typcorr --log-odds
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa sft semi 0.1 --typcorr
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa comb labelonly 0.1 --typcorr --log-odds
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa sft labelonly 0.1 --typcorr
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa pref-only labelonly 0.1 --typcorr --log-odds
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa pref-only labelonly 0.1 --typcorr

# --- tc-online + lenorm (commented out) ---
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa comb semi 0.1 --typcorr --lenorm --log-odds
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa sft semi 0.1 --typcorr --lenorm
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa comb labelonly 0.1 --typcorr --lenorm --log-odds
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa sft labelonly 0.1 --typcorr --lenorm
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa pref-only labelonly 0.1 --typcorr --lenorm --log-odds
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa pref-only labelonly 0.1 --typcorr --lenorm

# ============================================================
# AMBIGQA
# ============================================================

# --- plain ---
run 1 3 scripts/run_train_semi.sh google/gemma-2-2b ambigqa comb semi 0.1 --log-odds
run 1 3 scripts/run_train_semi.sh google/gemma-2-2b ambigqa sft semi 0.1
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ambigqa comb labelonly 0.1 --log-odds
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ambigqa sft labelonly 0.1
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ambigqa pref-only labelonly 0.1 --log-odds
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ambigqa pref-only labelonly 0.1

# --- tc-self ---
run 1 3 scripts/run_train_semi.sh google/gemma-2-2b ambigqa comb semi 0.1 --self-typcorr --log-odds
run 1 3 scripts/run_train_semi.sh google/gemma-2-2b ambigqa sft semi 0.1 --self-typcorr
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ambigqa comb labelonly 0.1 --self-typcorr --log-odds
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ambigqa sft labelonly 0.1 --self-typcorr
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ambigqa pref-only labelonly 0.1 --self-typcorr --log-odds
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ambigqa pref-only labelonly 0.1 --self-typcorr

# --- tc-self + lenorm ---
run 1 3 scripts/run_train_semi.sh google/gemma-2-2b ambigqa comb semi 0.1 --self-typcorr --lenorm --log-odds
run 1 3 scripts/run_train_semi.sh google/gemma-2-2b ambigqa sft semi 0.1 --self-typcorr --lenorm
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ambigqa comb labelonly 0.1 --self-typcorr --lenorm --log-odds
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ambigqa sft labelonly 0.1 --self-typcorr --lenorm
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ambigqa pref-only labelonly 0.1 --self-typcorr --lenorm --log-odds
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ambigqa pref-only labelonly 0.1 --self-typcorr --lenorm

# --- tc-online (commented out) ---
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ambigqa comb semi 0.1 --typcorr --log-odds
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ambigqa sft semi 0.1 --typcorr
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ambigqa comb labelonly 0.1 --typcorr --log-odds
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ambigqa sft labelonly 0.1 --typcorr
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ambigqa pref-only labelonly 0.1 --typcorr --log-odds
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ambigqa pref-only labelonly 0.1 --typcorr

# --- tc-online + lenorm (commented out) ---
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ambigqa comb semi 0.1 --typcorr --lenorm --log-odds
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ambigqa sft semi 0.1 --typcorr --lenorm
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ambigqa comb labelonly 0.1 --typcorr --lenorm --log-odds
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ambigqa sft labelonly 0.1 --typcorr --lenorm
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ambigqa pref-only labelonly 0.1 --typcorr --lenorm --log-odds
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ambigqa pref-only labelonly 0.1 --typcorr --lenorm

# ============================================================
# HYPERNYM-CONCAT-BANANAS-TO-DOGS-V2
# ============================================================

# --- plain ---
run 1 3 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double comb semi 0.1 --log-odds
run 1 3 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double sft semi 0.1
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double comb labelonly 0.1 --log-odds
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double sft labelonly 0.1
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double pref-only labelonly 0.1 --log-odds
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double pref-only labelonly 0.1

#STOP HERE
# --- tc-self ---
run 1 3 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double comb semi 0.1 --self-typcorr --log-odds
run 1 3 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double sft semi 0.1 --self-typcorr
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double comb labelonly 0.1 --self-typcorr --log-odds
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double sft labelonly 0.1 --self-typcorr
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double pref-only labelonly 0.1 --self-typcorr --log-odds
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double pref-only labelonly 0.1 --self-typcorr

# --- tc-self + lenorm ---
run 1 3 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double comb semi 0.1 --self-typcorr --lenorm --log-odds
run 1 3 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double sft semi 0.1 --self-typcorr --lenorm
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double comb labelonly 0.1 --self-typcorr --lenorm --log-odds
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double sft labelonly 0.1 --self-typcorr --lenorm
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double pref-only labelonly 0.1 --self-typcorr --lenorm --log-odds
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double pref-only labelonly 0.1 --self-typcorr --lenorm

# --- tc-online (commented out) ---
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double comb semi 0.1 --typcorr --log-odds
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double sft semi 0.1 --typcorr
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double comb labelonly 0.1 --typcorr --log-odds
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double sft labelonly 0.1 --typcorr
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double pref-only labelonly 0.1 --typcorr --log-odds
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double pref-only labelonly 0.1 --typcorr

# --- tc-online + lenorm (commented out) ---
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double comb semi 0.1 --typcorr --lenorm --log-odds
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double sft semi 0.1 --typcorr --lenorm
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double comb labelonly 0.1 --typcorr --lenorm --log-odds
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double sft labelonly 0.1 --typcorr --lenorm
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double pref-only labelonly 0.1 --typcorr --lenorm --log-odds
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double pref-only labelonly 0.1 --typcorr --lenorm

# ============================================================
# IFEVAL-CONCAT
# ============================================================

# --- plain ---
run 1 3 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat comb semi 0.1 --log-odds --disc-shots zero
run 1 3 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat sft semi 0.1 --disc-shots zero
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat comb labelonly 0.1 --log-odds --disc-shots zero
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat sft labelonly 0.1 --disc-shots zero
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat pref-only labelonly 0.1 --log-odds --disc-shots zero
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat pref-only labelonly 0.1 --disc-shots zero

# --- tc-self ---
run 1 3 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat comb semi 0.1 --self-typcorr --log-odds --disc-shots zero
run 1 3 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat sft semi 0.1 --self-typcorr --disc-shots zero
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat comb labelonly 0.1 --self-typcorr --log-odds --disc-shots zero
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat sft labelonly 0.1 --self-typcorr --disc-shots zero
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat pref-only labelonly 0.1 --self-typcorr --log-odds --disc-shots zero
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat pref-only labelonly 0.1 --self-typcorr --disc-shots zero

# --- tc-self + lenorm ---
run 1 3 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat comb semi 0.1 --self-typcorr --lenorm --log-odds --disc-shots zero
run 1 3 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat sft semi 0.1 --self-typcorr --lenorm --disc-shots zero
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat comb labelonly 0.1 --self-typcorr --lenorm --log-odds --disc-shots zero
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat sft labelonly 0.1 --self-typcorr --lenorm --disc-shots zero
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat pref-only labelonly 0.1 --self-typcorr --lenorm --log-odds --disc-shots zero
run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat pref-only labelonly 0.1 --self-typcorr --lenorm --disc-shots zero

# --- tc-online (commented out) ---
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat comb semi 0.1 --typcorr --log-odds
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat sft semi 0.1 --typcorr
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat comb labelonly 0.1 --typcorr --log-odds
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat sft labelonly 0.1 --typcorr
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat pref-only labelonly 0.1 --typcorr --log-odds
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat pref-only labelonly 0.1 --typcorr

# --- tc-online + lenorm (commented out) ---
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat comb semi 0.1 --typcorr --lenorm --log-odds
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat sft semi 0.1 --typcorr --lenorm
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat comb labelonly 0.1 --typcorr --lenorm --log-odds
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat sft labelonly 0.1 --typcorr --lenorm
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat pref-only labelonly 0.1 --typcorr --lenorm --log-odds
# run 1 2 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat pref-only labelonly 0.1 --typcorr --lenorm
