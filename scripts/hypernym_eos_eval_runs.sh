#!/bin/bash
# EOS sensitivity eval runs for Hypernym.
#
# Uses eval_by_claude.py via run_eval_semi.sh with --include-eos and the same
# eval style as existing hypernym semi scripts (--validator-log-odds + self/neg typcorr).
#
# Requested settings included:
#   1a, 1b, 2a, 3a, 3b, 4a.1, 4a.2, 4b.1
# Skipped by request:
#   2b (SFT labelonly + neg eval)
#
# Notes on missing models:
#   - gemma-2-2b-it: no local finetuned hypernym checkpoints found for 2a/3a/3b/4a.1/4a.2/4b.1
#   - gemma-2-9b-it: no local checkpoint found for 4a.2
#
# Usage:
#   source scripts/hypernym_eos_eval_runs.sh
#   (or copy/paste selected run lines)

TASKS="hypernym-bananas hypernym-bazookas hypernym-cabinets hypernym-cars hypernym-chairs hypernym-crows hypernym-diapers hypernym-dogs hypernym-dolls hypernym-ducklings hypernym-elephants hypernym-guns hypernym-hammers hypernym-helmets hypernym-jackets hypernym-kayaks hypernym-kites hypernym-mirrors"

# Base model IDs
B2B="google/gemma-2-2b"
B2BIT="google/gemma-2-2b-it"
B9BIT="google/gemma-2-9b-it"

# Finetuned model prefixes/suffixes
M2B="../models/v6-google--gemma-2-2b-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-double-all--d2g--random--alpha1.0"
M9BIT="../models/v6-google--gemma-2-9b-it-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-double-all--d2g--random--alpha1.0"

# ============================================================
# 1a) Base model eval with self typicality + EOS
# ============================================================
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$B2B" --self-typcorr --log-odds --include-eos -- $TASKS
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$B2BIT" --self-typcorr --log-odds --include-eos -- $TASKS
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$B9BIT" --self-typcorr --log-odds --include-eos -- $TASKS

# ============================================================
# 1b) Base model eval with neg typicality + EOS
# ============================================================
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$B2B" --neg-typcorr --log-odds --include-eos -- $TASKS
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$B2BIT" --neg-typcorr --log-odds --include-eos -- $TASKS
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$B9BIT" --neg-typcorr --log-odds --include-eos -- $TASKS

# ============================================================
# 2a) SFT finetuned labelonly, eval self typicality + EOS
# ============================================================
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2B--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x--labelonly0.1" --self-typcorr --log-odds --include-eos -- $TASKS
# MISSING (2b-it): ../models/v6-google--gemma-2-2b-it-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-double-all--d2g--random--alpha1.0--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x--labelonly0.1
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M9BIT--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x--labelonly0.1_merged" --self-typcorr --log-odds --include-eos -- $TASKS

# ============================================================
# 3a) "RankAlign" (pref-only, no training TC, no fsx), eval self + EOS
# (semi flag intentionally not required)
# ============================================================
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2B--full-completion" --self-typcorr --log-odds --include-eos -- $TASKS
# MISSING (2b-it): ../models/v6-google--gemma-2-2b-it-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-double-all--d2g--random--alpha1.0--full-completion
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M9BIT--full-completion_merged" --self-typcorr --log-odds --include-eos -- $TASKS

# ============================================================
# 3b) "RankAlign" (pref-only, no training TC, no fsx), eval neg + EOS
# ============================================================
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2B--full-completion" --neg-typcorr --log-odds --include-eos -- $TASKS
# MISSING (2b-it): ../models/v6-google--gemma-2-2b-it-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-double-all--d2g--random--alpha1.0--full-completion
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M9BIT--full-completion_merged" --neg-typcorr --log-odds --include-eos -- $TASKS

# ============================================================
# 4a.1) Comb+v+vallogs, semi finetuned (plain train), eval self + EOS
# ============================================================
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2B--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1" --self-typcorr --log-odds --include-eos -- $TASKS
# MISSING (2b-it): ../models/v6-google--gemma-2-2b-it-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-double-all--d2g--random--alpha1.0--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M9BIT--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1_merged" --self-typcorr --log-odds --include-eos -- $TASKS

# ============================================================
# 4a.2) Comb+v+vallogs, semi finetuned trained with tc-self, eval self + EOS
# ============================================================
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2B--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1" --self-typcorr --log-odds --include-eos -- $TASKS
# MISSING (2b-it): ../models/v6-google--gemma-2-2b-it-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-double-all--d2g--random--alpha1.0--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1
# MISSING (9b-it): ../models/v6-google--gemma-2-9b-it-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-double-all--d2g--random--alpha1.0--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1[_merged]

# ============================================================
# 4b.1) Comb+v+vallogs, semi finetuned trained with tc-neg, eval neg + EOS
# ============================================================
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2B--tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1" --neg-typcorr --log-odds --include-eos -- $TASKS
# MISSING (2b-it): ../models/v6-google--gemma-2-2b-it-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-double-all--d2g--random--alpha1.0--tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M9BIT--tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1_merged" --neg-typcorr --log-odds --include-eos -- $TASKS

# NOTE:
# 2b (SFT labelonly + neg eval) is intentionally skipped per request.
