#!/bin/bash
# harvest_he_trainset.sh — runs ON mll. Summarizes the humaneval TRAIN-SET eval score
# CSVs (gemma-4 s2/s4 × base/ep0/ep1/ep2 × {upper,multi}, scored on the --train split with
# --max-train 50 --stratified) into one per-file metric CSV for downstream per-epoch-dynamics
# table building. Safe to run repeatedly / incrementally.
#
# Train scores live in a single dir: outputs-he-trainset/scores_*_train_*.csv
# Output CSV (on mll): $REPO/.he_monitor/harvest/trainset_he_perfile_metrics.csv
# The laptop side downloads this and builds one LaTeX table per metric (raw + tc variants),
# adding an epoch axis (base/ep0/ep1/ep2) on top of the test-table parsing.
set -uo pipefail
REPO=/datastor2/jdr/rankalign
VENV=/datastor2/jdr/venvs/qwen35      # has pandas 2.3 / sklearn 1.5 / scipy
OUT="$REPO/.he_monitor/harvest"; mkdir -p "$OUT"
. "$VENV/bin/activate"
cd "$REPO/scripts"

echo "[harvest-trainset] outputs-he-trainset (both datasets, s2/s4, base+ep0/1/2) ->"
python summarize_scores_file.py \
  --glob "$REPO/outputs-he-trainset/scores_*_train_*humaneval*.csv" \
  --csv "$OUT/trainset_he_perfile_metrics.csv" --compact | tail -2

echo "[harvest-trainset] done. CSV:"; ls -la "$OUT/trainset_he_perfile_metrics.csv" 2>/dev/null
