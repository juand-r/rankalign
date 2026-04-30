#!/bin/bash
# Train 6 hypernym models WITH EOS in completion scoring (gemma-2-9b-it).
# Models are saved to models-eos/ (not models/).
#
# Settings:
#   1) RankAlign: pref-only, no TC, no fsx
#   2) SFT-LO: pref0.0 + nllv1.0 + nllg1.0 + fsx + labelonly0.1
#   3) Comb+v+vlo: nllv1.0 + nllg1.0 + fsx + vallogodds + semi0.1
#   4) Comb+v+vlo + tc-self: same as 3 + self-typcorr
#   5) Pref-only semi: pref-only + semi0.1
#   6) Pref-only semi + tc-self: pref-only + semi0.1 + self-typcorr

MODEL="google/gemma-2-9b-it"
TASK="hypernym-concat-bananas-to-dogs-double"
MDIR="../models-eos"

# 1) RankAlign (pref-only, no TC, no fsx)
run 1 6 scripts/run_train_semi.sh $MODEL $TASK pref-only labelonly 0.1 --include-eos --models-dir $MDIR

# 2) SFT-LO (labelonly)
run 1 6 scripts/run_train_semi.sh $MODEL $TASK sft labelonly 0.1 --include-eos --models-dir $MDIR

# 3) Comb+v+vlo (semi, plain)
run 1 6 scripts/run_train_semi.sh $MODEL $TASK comb semi 0.1 --log-odds --include-eos --models-dir $MDIR

# 4) Comb+v+vlo + tc-self (semi)
run 1 6 scripts/run_train_semi.sh $MODEL $TASK comb semi 0.1 --self-typcorr --log-odds --include-eos --models-dir $MDIR

# 5) Pref-only semi (no TC)
run 1 6 scripts/run_train_semi.sh $MODEL $TASK pref-only semi 0.1 --include-eos --models-dir $MDIR

# 6) Pref-only semi + tc-self
run 1 6 scripts/run_train_semi.sh $MODEL $TASK pref-only semi 0.1 --self-typcorr --include-eos --models-dir $MDIR

# ============================================================
# gemma-2-2b-it
# ============================================================
MODEL2="google/gemma-2-2b-it"

# 1) RankAlign
run 1 4 scripts/run_train_semi.sh $MODEL2 $TASK pref-only labelonly 0.1 --include-eos --models-dir $MDIR

# 2) SFT-LO
run 1 4 scripts/run_train_semi.sh $MODEL2 $TASK sft labelonly 0.1 --include-eos --models-dir $MDIR

# 3) Comb+v+vlo (semi, plain)
run 1 4 scripts/run_train_semi.sh $MODEL2 $TASK comb semi 0.1 --log-odds --include-eos --models-dir $MDIR

# 4) Comb+v+vlo + tc-self (semi)
run 1 4 scripts/run_train_semi.sh $MODEL2 $TASK comb semi 0.1 --self-typcorr --log-odds --include-eos --models-dir $MDIR

# 5) Pref-only semi
run 1 4 scripts/run_train_semi.sh $MODEL2 $TASK pref-only semi 0.1 --include-eos --models-dir $MDIR

# 6) Pref-only semi + tc-self
run 1 4 scripts/run_train_semi.sh $MODEL2 $TASK pref-only semi 0.1 --self-typcorr --include-eos --models-dir $MDIR

# ============================================================
# gemma-2-2b
# ============================================================
MODEL3="google/gemma-2-2b"

# 1) RankAlign
run 1 4 scripts/run_train_semi.sh $MODEL3 $TASK pref-only labelonly 0.1 --include-eos --models-dir $MDIR

# 2) SFT-LO
run 1 4 scripts/run_train_semi.sh $MODEL3 $TASK sft labelonly 0.1 --include-eos --models-dir $MDIR

# 3) Comb+v+vlo (semi, plain)
run 1 4 scripts/run_train_semi.sh $MODEL3 $TASK comb semi 0.1 --log-odds --include-eos --models-dir $MDIR

# 4) Comb+v+vlo + tc-self (semi)
run 1 4 scripts/run_train_semi.sh $MODEL3 $TASK comb semi 0.1 --self-typcorr --log-odds --include-eos --models-dir $MDIR

# 5) Pref-only semi
run 1 4 scripts/run_train_semi.sh $MODEL3 $TASK pref-only semi 0.1 --include-eos --models-dir $MDIR

# 6) Pref-only semi + tc-self
run 1 4 scripts/run_train_semi.sh $MODEL3 $TASK pref-only semi 0.1 --self-typcorr --include-eos --models-dir $MDIR
