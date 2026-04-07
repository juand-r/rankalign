#!/bin/bash
# All semi-supervised / labeled-only training runs with NEG-TYPICALITY correction.
# Mirrors the tc-self blocks from semi_supervised_runs.sh with --neg-typcorr.
# Each line is a standalone `run` command.
# Default: mode=g (delta=0.15), ratio=0.1, split-seed=42

# ============================================================
# PLAUSIBLEQA
# ============================================================

run 1 4 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa comb semi 0.1 --neg-typcorr --log-odds
run 1 4 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa sft semi 0.1 --neg-typcorr
run 1 4 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa comb labelonly 0.1 --neg-typcorr --log-odds
run 1 4 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa pref-only labelonly 0.1 --neg-typcorr --log-odds
run 1 4 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa pref-only labelonly 0.1 --neg-typcorr
run 1 4 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa pref-only semi 0.1 --neg-typcorr

# ============================================================
# AMBIGQA
# ============================================================

run 1 4 scripts/run_train_semi.sh google/gemma-2-2b ambigqa comb semi 0.1 --neg-typcorr --log-odds
run 1 4 scripts/run_train_semi.sh google/gemma-2-2b ambigqa sft semi 0.1 --neg-typcorr
run 1 4 scripts/run_train_semi.sh google/gemma-2-2b ambigqa comb labelonly 0.1 --neg-typcorr --log-odds
run 1 4 scripts/run_train_semi.sh google/gemma-2-2b ambigqa pref-only labelonly 0.1 --neg-typcorr --log-odds
run 1 4 scripts/run_train_semi.sh google/gemma-2-2b ambigqa pref-only labelonly 0.1 --neg-typcorr
run 1 4 scripts/run_train_semi.sh google/gemma-2-2b ambigqa pref-only semi 0.1 --neg-typcorr

# ============================================================
# HYPERNYM-CONCAT-BANANAS-TO-DOGS-DOUBLE
# ============================================================

run 1 4 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double comb semi 0.1 --neg-typcorr --log-odds
run 1 4 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double sft semi 0.1 --neg-typcorr
run 1 4 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double comb labelonly 0.1 --neg-typcorr --log-odds
run 1 4 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double pref-only labelonly 0.1 --neg-typcorr --log-odds
run 1 4 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double pref-only labelonly 0.1 --neg-typcorr
run 1 4 scripts/run_train_semi.sh google/gemma-2-2b hypernym-concat-bananas-to-dogs-double pref-only semi 0.1 --neg-typcorr

# ============================================================
# IFEVAL-CONCAT
# ============================================================

run 1 6 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat comb semi 0.1 --neg-typcorr --log-odds --disc-shots zero
run 1 6 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat sft semi 0.1 --neg-typcorr --disc-shots zero
run 1 4 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat comb labelonly 0.1 --neg-typcorr --log-odds --disc-shots zero
run 1 4 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat pref-only labelonly 0.1 --neg-typcorr --log-odds --disc-shots zero
run 1 4 scripts/run_train_semi.sh google/gemma-2-2b ifeval-concat pref-only labelonly 0.1 --neg-typcorr --disc-shots zero
