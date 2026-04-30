#!/bin/bash
# Evaluate the 18 EOS-trained hypernym models with self-TC + EOS.
# Score files go to outputs-eos-models/ (not outputs/).
#
# All evals use --self-typcorr --log-odds --include-eos.
# Models come from models-eos/.
#   9b-it: epoch 2, LoRA merged (_merged suffix)
#   2b-it: epoch 2, no LoRA (no _merged suffix)
#   2b:    epoch 2, no LoRA (no _merged suffix)

TASKS="hypernym-bananas hypernym-bazookas hypernym-cabinets hypernym-cars hypernym-chairs hypernym-crows hypernym-diapers hypernym-dogs hypernym-dolls hypernym-ducklings hypernym-elephants hypernym-guns hypernym-hammers hypernym-helmets hypernym-jackets hypernym-kayaks hypernym-kites hypernym-mirrors"

MDIR="../models-eos"
ODIR="../outputs-eos-models"

# ============================================================
# gemma-2-9b-it (LoRA merged)
# ============================================================
M9="$MDIR/v6-google--gemma-2-9b-it-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-double-all--d2g--random--alpha1.0"

# 1) RankAlign
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M9--full-completion--eos_merged" --self-typcorr --log-odds --include-eos --outputs-dir $ODIR -- $TASKS

# 2) SFT-LO
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M9--full-completion--eos--pref0.0--nllv1.0--nllg1.0--force-same-x--labelonly0.1_merged" --self-typcorr --log-odds --include-eos --outputs-dir $ODIR -- $TASKS

# 3) Comb+v+vlo (semi, plain)
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M9--full-completion--eos--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1_merged" --self-typcorr --log-odds --include-eos --outputs-dir $ODIR -- $TASKS

# 4) Comb+v+vlo + tc-self (semi)
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M9--tc-self--full-completion--eos--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1_merged" --self-typcorr --log-odds --include-eos --outputs-dir $ODIR -- $TASKS

# 5) Pref-only semi
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M9--full-completion--eos--semi0.1_merged" --self-typcorr --log-odds --include-eos --outputs-dir $ODIR -- $TASKS

# 6) Pref-only semi + tc-self
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M9--tc-self--full-completion--eos--semi0.1_merged" --self-typcorr --log-odds --include-eos --outputs-dir $ODIR -- $TASKS

# ============================================================
# gemma-2-2b-it (no LoRA, no _merged)
# ============================================================
M2I="$MDIR/v6-google--gemma-2-2b-it-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-double-all--d2g--random--alpha1.0"

# 1) RankAlign
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2I--full-completion--eos" --self-typcorr --log-odds --include-eos --outputs-dir $ODIR -- $TASKS

# 2) SFT-LO
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2I--full-completion--eos--pref0.0--nllv1.0--nllg1.0--force-same-x--labelonly0.1" --self-typcorr --log-odds --include-eos --outputs-dir $ODIR -- $TASKS

# 3) Comb+v+vlo (semi, plain)
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2I--full-completion--eos--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1" --self-typcorr --log-odds --include-eos --outputs-dir $ODIR -- $TASKS

# 4) Comb+v+vlo + tc-self (semi)
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2I--tc-self--full-completion--eos--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1" --self-typcorr --log-odds --include-eos --outputs-dir $ODIR -- $TASKS

# 5) Pref-only semi
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2I--full-completion--eos--semi0.1" --self-typcorr --log-odds --include-eos --outputs-dir $ODIR -- $TASKS

# 6) Pref-only semi + tc-self
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2I--tc-self--full-completion--eos--semi0.1" --self-typcorr --log-odds --include-eos --outputs-dir $ODIR -- $TASKS

# ============================================================
# gemma-2-2b (no LoRA, no _merged)
# ============================================================
M2="$MDIR/v6-google--gemma-2-2b-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-double-all--d2g--random--alpha1.0"

# 1) RankAlign
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2--full-completion--eos" --self-typcorr --log-odds --include-eos --outputs-dir $ODIR -- $TASKS

# 2) SFT-LO
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2--full-completion--eos--pref0.0--nllv1.0--nllg1.0--force-same-x--labelonly0.1" --self-typcorr --log-odds --include-eos --outputs-dir $ODIR -- $TASKS

# 3) Comb+v+vlo (semi, plain)
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2--full-completion--eos--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1" --self-typcorr --log-odds --include-eos --outputs-dir $ODIR -- $TASKS

# 4) Comb+v+vlo + tc-self (semi)
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2--tc-self--full-completion--eos--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1" --self-typcorr --log-odds --include-eos --outputs-dir $ODIR -- $TASKS

# 5) Pref-only semi
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2--full-completion--eos--semi0.1" --self-typcorr --log-odds --include-eos --outputs-dir $ODIR -- $TASKS

# 6) Pref-only semi + tc-self
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "$M2--tc-self--full-completion--eos--semi0.1" --self-typcorr --log-odds --include-eos --outputs-dir $ODIR -- $TASKS
