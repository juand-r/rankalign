#!/bin/bash
# Hypernym evals using --base-typicality (pre-finetuning base model for TC).
# No EOS. Scores go to outputs/ (standard location).
#
# For finetuned models only (base-TC on base model = self-TC, already done).
#
# Settings: 2a, 3a, 3b, 4a.1, 4a.2, 4b.1
# Each finetuned model is evaluated using its own base model for TC:
#   2b finetuned    → --base-model google/gemma-2-2b
#   2b-it finetuned → --base-model google/gemma-2-2b-it
#   9b-it finetuned → --base-model google/gemma-2-9b-it

TASKS="hypernym-bananas hypernym-bazookas hypernym-cabinets hypernym-cars hypernym-chairs hypernym-crows hypernym-diapers hypernym-dogs hypernym-dolls hypernym-ducklings hypernym-elephants hypernym-guns hypernym-hammers hypernym-helmets hypernym-jackets hypernym-kayaks hypernym-kites hypernym-mirrors"

# ============================================================
# gemma-2-2b
# ============================================================
M2B="../models/v6-google--gemma-2-2b-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-double-all--d2g--random--alpha1.0"
BASE2B="google/gemma-2-2b"

# 2a) SFT-LO, base-TC self-eval
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2B--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x--labelonly0.1" --self-typcorr --base-typcorr --base-model $BASE2B --log-odds -- $TASKS

# 3a) RankAlign, base-TC self-eval
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2B--full-completion" --self-typcorr --base-typcorr --base-model $BASE2B --log-odds -- $TASKS

# 3b) RankAlign, base-TC neg-eval
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2B--full-completion" --neg-typcorr --base-typcorr --base-model $BASE2B --log-odds -- $TASKS

# 4a.1) Comb+v+vlo, base-TC self-eval
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2B--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1" --self-typcorr --base-typcorr --base-model $BASE2B --log-odds -- $TASKS

# 4a.2) Comb+v+vlo tc-s, base-TC self-eval
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2B--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1" --self-typcorr --base-typcorr --base-model $BASE2B --log-odds -- $TASKS

# 4b.1) Comb+v+vlo tc-n, base-TC neg-eval
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2B--tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1" --neg-typcorr --base-typcorr --base-model $BASE2B --log-odds -- $TASKS

# STOPPED HERE.
# ============================================================
# gemma-2-2b-it
# ============================================================
M2BIT="../models/v6-google--gemma-2-2b-it-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-double-all--d2g--random--alpha1.0"
BASE2BIT="google/gemma-2-2b-it"

# 2a) SFT-LO, base-TC self-eval
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2BIT--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x--labelonly0.1" --self-typcorr --base-typcorr --base-model $BASE2BIT --log-odds -- $TASKS

# 3a) RankAlign, base-TC self-eval
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2BIT--full-completion" --self-typcorr --base-typcorr --base-model $BASE2BIT --log-odds -- $TASKS

# 3b) RankAlign, base-TC neg-eval
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2BIT--full-completion" --neg-typcorr --base-typcorr --base-model $BASE2BIT --log-odds -- $TASKS

# 4a.1) Comb+v+vlo, base-TC self-eval
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2BIT--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1" --self-typcorr --base-typcorr --base-model $BASE2BIT --log-odds -- $TASKS

# 4a.2) Comb+v+vlo tc-s, base-TC self-eval
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2BIT--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1" --self-typcorr --base-typcorr --base-model $BASE2BIT --log-odds -- $TASKS

# 4b.1) Comb+v+vlo tc-n, base-TC neg-eval
# MISSING (2b-it): no tc-neg checkpoint available
# run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2BIT--tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1" --neg-typcorr --base-typcorr --base-model $BASE2BIT --log-odds -- $TASKS

# ============================================================
# gemma-2-9b-it (LoRA merged)
# ============================================================
M9BIT="../models/v6-google--gemma-2-9b-it-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-double-all--d2g--random--alpha1.0"
BASE9BIT="google/gemma-2-9b-it"

# 2a) SFT-LO, base-TC self-eval
run 1 3 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M9BIT--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x--labelonly0.1_merged" --self-typcorr --base-typcorr --base-model $BASE9BIT --log-odds -- $TASKS

# 3a) RankAlign, base-TC self-eval
run 1 3 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M9BIT--full-completion_merged" --self-typcorr --base-typcorr --base-model $BASE9BIT --log-odds -- $TASKS

# 3b) RankAlign, base-TC neg-eval
run 1 3 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M9BIT--full-completion_merged" --neg-typcorr --base-typcorr --base-model $BASE9BIT --log-odds -- $TASKS

# 4a.1) Comb+v+vlo, base-TC self-eval
run 1 3 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M9BIT--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1_merged" --self-typcorr --base-typcorr --base-model $BASE9BIT --log-odds -- $TASKS

# 4a.2) Comb+v+vlo tc-s, base-TC self-eval
# MISSING (9b-it): no tc-self comb+v+vlo checkpoint available locally
# run 1 3 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M9BIT--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1_merged" --self-typcorr --base-typcorr --base-model $BASE9BIT --log-odds -- $TASKS

# 4b.1) Comb+v+vlo tc-n, base-TC neg-eval
run 1 3 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M9BIT--tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1_merged" --neg-typcorr --base-typcorr --base-model $BASE9BIT --log-odds -- $TASKS
