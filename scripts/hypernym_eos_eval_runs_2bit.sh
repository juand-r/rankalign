#!/bin/bash
# EOS eval runs for gemma-2-2b-it Hypernym settings.
#
# Covers:
#   1a, 1b, 2a, 3a, 3b, 4a.1, 4a.2, 4b.1
# Skips:
#   2b (SFT labelonly + neg eval) per prior decision.
#
# Note:
#   4b.1 checkpoint for 2b-it is still missing locally (tc-neg + comb+v+vallog + semi).

TASKS="hypernym-bananas hypernym-bazookas hypernym-cabinets hypernym-cars hypernym-chairs hypernym-crows hypernym-diapers hypernym-dogs hypernym-dolls hypernym-ducklings hypernym-elephants hypernym-guns hypernym-hammers hypernym-helmets hypernym-jackets hypernym-kayaks hypernym-kites hypernym-mirrors"

B2BIT="google/gemma-2-2b-it"
M2BIT="../models/v6-google--gemma-2-2b-it-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-double-all--d2g--random--alpha1.0"

# ============================================================
# 1a) Base model eval with self typicality + EOS
# ============================================================
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$B2BIT" --self-typcorr --log-odds --include-eos -- $TASKS

# ============================================================
# 1b) Base model eval with neg typicality + EOS
# ============================================================
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$B2BIT" --neg-typcorr --log-odds --include-eos -- $TASKS

# ============================================================
# 2a) SFT finetuned labelonly, eval self typicality + EOS
# ============================================================
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2BIT--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x--labelonly0.1" --self-typcorr --log-odds --include-eos -- $TASKS

# ============================================================
# 3a) "RankAlign" (pref-only, no training TC, no fsx), eval self + EOS
# (semi flag intentionally not required)
# ============================================================
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2BIT--full-completion" --self-typcorr --log-odds --include-eos -- $TASKS

# ============================================================
# 3b) "RankAlign" (pref-only, no training TC, no fsx), eval neg + EOS
# ============================================================
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2BIT--full-completion" --neg-typcorr --log-odds --include-eos -- $TASKS

# ============================================================
# 4a.1) Comb+v+vallogs, semi finetuned (plain train), eval self + EOS
# ============================================================
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2BIT--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1" --self-typcorr --log-odds --include-eos -- $TASKS

# ============================================================
# 4a.2) Comb+v+vallogs, semi finetuned trained with tc-self, eval self + EOS
# ============================================================
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2BIT--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1" --self-typcorr --log-odds --include-eos -- $TASKS

# ============================================================
# 4b.1) Comb+v+vallogs, semi finetuned trained with tc-neg, eval neg + EOS
# ============================================================
# MISSING (2b-it): ../models/v6-google--gemma-2-2b-it-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-double-all--d2g--random--alpha1.0--tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1

# NOTE:
# 2b (SFT labelonly + neg eval) is intentionally skipped per request.
