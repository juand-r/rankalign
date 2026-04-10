#!/bin/bash
# All semi-supervised / labeled-only training runs.
# Each line is a standalone `run` command.
# Default: mode=g (delta=0.15), ratio=0.1, split-seed=42
#
# Recipes:
#   (1) plain:              comb gets --log-odds, sft plain, pref both
#   (2) tc:                 + $TC
#   (3) tc + lenorm:        + $TC --lenorm
#   (4) tc-online:          COMMENTED OUT
#   (5) tc-online + lenorm: COMMENTED OUT

# Toggle typicality correction variant: $TC or --neg-typcorr
#TC="--self-typcorr"
TC="--neg-typcorr"


# ============================================================
# PLAUSIBLEQA
# ============================================================

# --- plain ---
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it plausibleqa comb semi 0.1 --log-odds
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it plausibleqa sft semi 0.1
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it plausibleqa comb labelonly 0.1 --log-odds
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it plausibleqa sft labelonly 0.1
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it plausibleqa pref-only labelonly 0.1 --log-odds
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it plausibleqa pref-only labelonly 0.1
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it plausibleqa pref-only semi 0.1

# --- tc-self ---
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it plausibleqa comb semi 0.1 $TC --log-odds
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it plausibleqa sft semi 0.1 $TC
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it plausibleqa comb labelonly 0.1 $TC --log-odds
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it plausibleqa sft labelonly 0.1 $TC
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it plausibleqa pref-only labelonly 0.1 $TC --log-odds
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it plausibleqa pref-only labelonly 0.1 $TC
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it plausibleqa pref-only semi 0.1 $TC

# --- tc-self + lenorm ---
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it plausibleqa comb semi 0.1 $TC --lenorm --log-odds
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it plausibleqa sft semi 0.1 $TC --lenorm
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it plausibleqa comb labelonly 0.1 $TC --lenorm --log-odds
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it plausibleqa sft labelonly 0.1 $TC --lenorm
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it plausibleqa pref-only labelonly 0.1 $TC --lenorm --log-odds
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it plausibleqa pref-only labelonly 0.1 $TC --lenorm

# --- tc-online (commented out) ---
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it plausibleqa comb semi 0.1 --typcorr --log-odds
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it plausibleqa sft semi 0.1 --typcorr
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it plausibleqa comb labelonly 0.1 --typcorr --log-odds
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it plausibleqa sft labelonly 0.1 --typcorr
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it plausibleqa pref-only labelonly 0.1 --typcorr --log-odds
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it plausibleqa pref-only labelonly 0.1 --typcorr

# --- tc-online + lenorm (commented out) ---
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it plausibleqa comb semi 0.1 --typcorr --lenorm --log-odds
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it plausibleqa sft semi 0.1 --typcorr --lenorm
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it plausibleqa comb labelonly 0.1 --typcorr --lenorm --log-odds
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it plausibleqa sft labelonly 0.1 --typcorr --lenorm
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it plausibleqa pref-only labelonly 0.1 --typcorr --lenorm --log-odds
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it plausibleqa pref-only labelonly 0.1 --typcorr --lenorm

# ============================================================
# AMBIGQA
# ============================================================

# --- plain ---
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ambigqa comb semi 0.1 --log-odds
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ambigqa sft semi 0.1
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ambigqa comb labelonly 0.1 --log-odds
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ambigqa sft labelonly 0.1
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ambigqa pref-only labelonly 0.1 --log-odds
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ambigqa pref-only labelonly 0.1
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ambigqa pref-only semi 0.1

# --- tc-self ---
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ambigqa comb semi 0.1 $TC --log-odds
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ambigqa sft semi 0.1 $TC
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ambigqa comb labelonly 0.1 $TC --log-odds
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ambigqa sft labelonly 0.1 $TC
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ambigqa pref-only labelonly 0.1 $TC --log-odds
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ambigqa pref-only labelonly 0.1 $TC
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ambigqa pref-only semi 0.1 $TC

# --- tc-self + lenorm ---
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ambigqa comb semi 0.1 $TC --lenorm --log-odds
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ambigqa sft semi 0.1 $TC --lenorm
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ambigqa comb labelonly 0.1 $TC --lenorm --log-odds
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ambigqa sft labelonly 0.1 $TC --lenorm
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ambigqa pref-only labelonly 0.1 $TC --lenorm --log-odds
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ambigqa pref-only labelonly 0.1 $TC --lenorm

# --- tc-online (commented out) ---
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ambigqa comb semi 0.1 --typcorr --log-odds
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ambigqa sft semi 0.1 --typcorr
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ambigqa comb labelonly 0.1 --typcorr --log-odds
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ambigqa sft labelonly 0.1 --typcorr
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ambigqa pref-only labelonly 0.1 --typcorr --log-odds
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ambigqa pref-only labelonly 0.1 --typcorr

# --- tc-online + lenorm (commented out) ---
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ambigqa comb semi 0.1 --typcorr --lenorm --log-odds
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ambigqa sft semi 0.1 --typcorr --lenorm
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ambigqa comb labelonly 0.1 --typcorr --lenorm --log-odds
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ambigqa sft labelonly 0.1 --typcorr --lenorm
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ambigqa pref-only labelonly 0.1 --typcorr --lenorm --log-odds
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ambigqa pref-only labelonly 0.1 --typcorr --lenorm

# ============================================================
# HYPERNYM-CONCAT-BANANAS-TO-DOGS-V2
# ============================================================

# --- plain ---
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it hypernym-concat-bananas-to-dogs-double comb semi 0.1 --log-odds
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it hypernym-concat-bananas-to-dogs-double sft semi 0.1
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it hypernym-concat-bananas-to-dogs-double comb labelonly 0.1 --log-odds
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it hypernym-concat-bananas-to-dogs-double sft labelonly 0.1
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it hypernym-concat-bananas-to-dogs-double pref-only labelonly 0.1 --log-odds
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it hypernym-concat-bananas-to-dogs-double pref-only labelonly 0.1
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it hypernym-concat-bananas-to-dogs-double pref-only semi 0.1

#STOP HERE
# --- tc-self ---
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it hypernym-concat-bananas-to-dogs-double comb semi 0.1 $TC --log-odds
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it hypernym-concat-bananas-to-dogs-double sft semi 0.1 $TC
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it hypernym-concat-bananas-to-dogs-double comb labelonly 0.1 $TC --log-odds
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it hypernym-concat-bananas-to-dogs-double sft labelonly 0.1 $TC
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it hypernym-concat-bananas-to-dogs-double pref-only labelonly 0.1 $TC --log-odds
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it hypernym-concat-bananas-to-dogs-double pref-only labelonly 0.1 $TC
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it hypernym-concat-bananas-to-dogs-double pref-only semi 0.1 $TC

# --- tc-self + lenorm ---
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it hypernym-concat-bananas-to-dogs-double comb semi 0.1 $TC --lenorm --log-odds
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it hypernym-concat-bananas-to-dogs-double sft semi 0.1 $TC --lenorm
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it hypernym-concat-bananas-to-dogs-double comb labelonly 0.1 $TC --lenorm --log-odds
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it hypernym-concat-bananas-to-dogs-double sft labelonly 0.1 $TC --lenorm
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it hypernym-concat-bananas-to-dogs-double pref-only labelonly 0.1 $TC --lenorm --log-odds
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it hypernym-concat-bananas-to-dogs-double pref-only labelonly 0.1 $TC --lenorm

# --- tc-online (commented out) ---
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it hypernym-concat-bananas-to-dogs-double comb semi 0.1 --typcorr --log-odds
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it hypernym-concat-bananas-to-dogs-double sft semi 0.1 --typcorr
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it hypernym-concat-bananas-to-dogs-double comb labelonly 0.1 --typcorr --log-odds
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it hypernym-concat-bananas-to-dogs-double sft labelonly 0.1 --typcorr
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it hypernym-concat-bananas-to-dogs-double pref-only labelonly 0.1 --typcorr --log-odds
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it hypernym-concat-bananas-to-dogs-double pref-only labelonly 0.1 --typcorr

# --- tc-online + lenorm (commented out) ---
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it hypernym-concat-bananas-to-dogs-double comb semi 0.1 --typcorr --lenorm --log-odds
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it hypernym-concat-bananas-to-dogs-double sft semi 0.1 --typcorr --lenorm
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it hypernym-concat-bananas-to-dogs-double comb labelonly 0.1 --typcorr --lenorm --log-odds
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it hypernym-concat-bananas-to-dogs-double sft labelonly 0.1 --typcorr --lenorm
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it hypernym-concat-bananas-to-dogs-double pref-only labelonly 0.1 --typcorr --lenorm --log-odds
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it hypernym-concat-bananas-to-dogs-double pref-only labelonly 0.1 --typcorr --lenorm

# ============================================================
# IFEVAL-CONCAT
# ============================================================

# NOTE: I stopped here.

# --- plain ---
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ifeval-concat comb semi 0.1 --log-odds --disc-shots zero
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ifeval-concat sft semi 0.1 --disc-shots zero
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ifeval-concat comb labelonly 0.1 --log-odds --disc-shots zero
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ifeval-concat sft labelonly 0.1 --disc-shots zero
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ifeval-concat pref-only labelonly 0.1 --log-odds --disc-shots zero
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ifeval-concat pref-only labelonly 0.1 --disc-shots zero

# --- tc-self ---
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ifeval-concat comb semi 0.1 $TC --log-odds --disc-shots zero
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ifeval-concat sft semi 0.1 $TC --disc-shots zero
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ifeval-concat comb labelonly 0.1 $TC --log-odds --disc-shots zero
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ifeval-concat sft labelonly 0.1 $TC --disc-shots zero
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ifeval-concat pref-only labelonly 0.1 $TC --log-odds --disc-shots zero
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ifeval-concat pref-only labelonly 0.1 $TC --disc-shots zero

# --- tc-self + lenorm ---
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ifeval-concat comb semi 0.1 $TC --lenorm --log-odds --disc-shots zero
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ifeval-concat sft semi 0.1 $TC --lenorm --disc-shots zero
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ifeval-concat comb labelonly 0.1 $TC --lenorm --log-odds --disc-shots zero
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ifeval-concat sft labelonly 0.1 $TC --lenorm --disc-shots zero
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ifeval-concat pref-only labelonly 0.1 $TC --lenorm --log-odds --disc-shots zero
run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ifeval-concat pref-only labelonly 0.1 $TC --lenorm --disc-shots zero

# --- tc-online (commented out) ---
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ifeval-concat comb semi 0.1 --typcorr --log-odds
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ifeval-concat sft semi 0.1 --typcorr
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ifeval-concat comb labelonly 0.1 --typcorr --log-odds
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ifeval-concat sft labelonly 0.1 --typcorr
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ifeval-concat pref-only labelonly 0.1 --typcorr --log-odds
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ifeval-concat pref-only labelonly 0.1 --typcorr

# --- tc-online + lenorm (commented out) ---
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ifeval-concat comb semi 0.1 --typcorr --lenorm --log-odds
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ifeval-concat sft semi 0.1 --typcorr --lenorm
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ifeval-concat comb labelonly 0.1 --typcorr --lenorm --log-odds
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ifeval-concat sft labelonly 0.1 --typcorr --lenorm
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ifeval-concat pref-only labelonly 0.1 --typcorr --lenorm --log-odds
# run 1 16 scripts/run_train_semi.sh google/gemma-2-9b-it ifeval-concat pref-only labelonly 0.1 --typcorr --lenorm
