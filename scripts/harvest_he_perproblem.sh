#!/bin/bash
# harvest_he_perproblem.sh — runs ON mll. Summarizes the PER-PROBLEM TRAIN-set eval score CSVs
# (outputs-he-trainset-perproblem/, one file per train problem) into a per-file metrics CSV
# (gen_roc/val_roc/val_acc/pearson/spearman per file x gen-variant). Downstream the table
# builder aggregates these per (model,dataset,setting,epoch,eval-mode) as mean +/- SE over the
# 80 TRAIN problems -- the correct, per-problem train metric (replaces the global-pool eval).
# Safe to run repeatedly / on partial data.
set -uo pipefail
REPO=/datastor2/jdr/rankalign
VENV=/datastor2/jdr/venvs/qwen35
OUT="$REPO/.he_monitor/harvest"; mkdir -p "$OUT"
source "$VENV/bin/activate"
cd "$REPO/scripts"
echo "[harvest-perproblem] outputs-he-trainset-perproblem ->"
python summarize_scores_file.py \
  --glob "$REPO/outputs-he-trainset-perproblem/scores_*train*humaneval*.csv" \
  --csv "$OUT/perproblem_train_perfile_metrics.csv" --compact | tail -2
echo "[harvest-perproblem] done:"; ls -la "$OUT/perproblem_train_perfile_metrics.csv" 2>/dev/null
