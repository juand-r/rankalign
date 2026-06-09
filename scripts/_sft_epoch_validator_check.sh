#!/bin/bash
# Controlled test: does the IFEval VALIDATOR score depend on training EPOCH for qwen SFT?
#
# Motivation: in the rerun-vs-original comparison, the qwen SFT IFEval validator jumped
# +25 (val_roc 57.5 -> 82.8). RankAlign's big shifts are a known epoch1-vs-epoch2 artifact,
# so we must rule out the same confound for SFT. This holds venv (qwen35), eval code, and
# data CONSTANT and varies ONLY the epoch of the rerun qwen SFT checkpoint
# (s1, delta0.84, labelonly0.1, merged), evaluating IFEval OOD (20 prompts), NO_BASE.
#
# Read-out: val_roc per epoch (aggregate the scores_*_val* later with summarize_scores_file).
#   - If val_roc is ~flat across epoch0/1/2  -> epoch is NOT the cause; the +25 is venv/eval.
#   - If epoch1 << epoch2 (~57 vs ~83)       -> epoch IS implicated; original was likely ep1.
# Reference points already in hand: original pod eval val_roc = 57.5 ; rerun epoch2 = 82.8.
#
# Scores land in outputs-epoch-check/ (isolated; does NOT touch outputs-rerun-wandb).
set -euo pipefail
REPO=/datastor2/jdr/rankalign
TMPL="$REPO/models2-rerun-wandb/v7-Qwen--Qwen3.5-9B-delta0.84-epochE--ifeval-concat-all--d2g--random--alpha1.0--full-completion--pref0.0--nllv1.0--nllg1.0--vallogodds--labelonly0.1--fix1_merged"
cd "$REPO"
for E in 0 1 2; do
    DIR="${TMPL/epochE/epoch$E}"
    if [ ! -d "$DIR" ]; then echo "MISSING checkpoint: $DIR"; exit 1; fi
    jid=$(sbatch --parsable \
        --job-name="sftep$E" \
        --export=ALL,EVAL_ONLY=1,NO_BASE=1,IFEVAL_SPLIT=ood,OUTPUTS_DIR=$REPO/outputs-epoch-check,MODEL_DIR_OVERRIDE="$DIR" \
        scripts/run_qwen35_cell_mll.sbatch ifeval s1)
    echo "epoch$E -> job $jid  ($DIR)"
done
