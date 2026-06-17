#!/bin/bash
# harvest_he_runs.sh — runs ON mll. Summarizes the humaneval eval score CSVs into
# per-file metric CSVs (gen_roc / val_roc / val_acc / pearson / spearman per file x
# gen-variant) for downstream table building. Covers every setting present in each dir
# (base, s1, s2, s3, s4, s7, s13). Safe to run repeatedly / incrementally.
#
# Output CSVs (on mll): $REPO/.he_monitor/harvest/{qwen,gemma_cu,gemma_cm}_he_perfile_metrics.csv
# The laptop side downloads these and builds one LaTeX table per metric (raw + tc variants).
set -uo pipefail
REPO=/datastor2/jdr/rankalign
VENV=/datastor2/jdr/venvs/qwen35      # has pandas 2.3 / sklearn 1.5 / scipy
OUT="$REPO/.he_monitor/harvest"; mkdir -p "$OUT"
source "$VENV/bin/activate"
cd "$REPO/scripts"
S=summarize_scores_file.py

echo "[harvest] qwen (outputs-rerun-wandb, both datasets) ->"
python "$S" --glob "$REPO/outputs-rerun-wandb/scores_*.csv"        --csv "$OUT/qwen_he_perfile_metrics.csv"  --compact | tail -2
echo "[harvest] gemma upper (outputs_gemma4_mll_tmp) ->"
python "$S" --glob "$REPO/outputs_gemma4_mll_tmp/scores_*.csv"     --csv "$OUT/gemma_cu_perfile_metrics.csv" --compact | tail -2
echo "[harvest] gemma multi (outputs_gemma4_mll_tmp-multi) ->"
python "$S" --glob "$REPO/outputs_gemma4_mll_tmp-multi/scores_*.csv" --csv "$OUT/gemma_cm_perfile_metrics.csv" --compact | tail -2

echo "[harvest] done. CSVs:"; ls -la "$OUT"/*.csv 2>/dev/null
