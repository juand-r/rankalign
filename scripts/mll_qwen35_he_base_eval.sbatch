#!/bin/bash
#SBATCH --job-name=q35base
#SBATCH --partition=allnodes
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/datastor2/jdr/logs/%j.out
#SBATCH --error=/datastor2/jdr/logs/%j.err
#
# BASE-model eval of Qwen/Qwen3.5-9B (UNTRAINED) on humaneval upper|multi.
# NOT affected by the train-time prompt/completion seam bug: eval scores via
# get_completion_token_logprobs, which tokenizes prompt & completion SEPARATELY
# and concatenates token IDs (no cross-boundary BPE merge). Same function gemma
# eval used. No training, no merged adapter -- just the base model.
#
# Usage:
#   HE_TASK=humaneval-v2.1correct-upper MODE=--self-typicality sbatch scripts/mll_qwen35_he_base_eval.sbatch
#   TASK_LIMIT=2 ... (canary: only first 2 tasks)
set -euo pipefail
HE_TASK="${HE_TASK:-humaneval-v2.1correct-upper}"
MODE="${MODE:---self-typicality}"     # --self-typicality | --neg-typicality
WTAG=$(case "$HE_TASK" in *correct-upper) echo cu;; *correct-multi) echo cm;; *) echo x;; esac)
MODEL="Qwen/Qwen3.5-9B"
REPO=/datastor2/jdr/rankalign
VENV=/datastor2/jdr/venvs/qwen35
OUTDIR="${OUTDIR:-/datastor2/jdr/rankalign/outputs-rerun-wandb}"
DONE_DIR="$OUTDIR/.done_base"
LOGDIR=/datastor2/jdr/logs
TASK_LIMIT="${TASK_LIMIT:-0}"   # 0 = all 82

export HF_HOME=/datastor2/jdr/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
export HF_TOKEN=$(cat /datastor2/jdr/.hftoken 2>/dev/null || true)
export HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "$VENV/bin/activate"
cd "$REPO/scripts"
mkdir -p "$OUTDIR" "$DONE_DIR"

# moe.py patch (idempotent) — qwen35 venv / transformers 5.8 + torch 2.4.1
MOE_FILE=$(python -c "import transformers, os; print(os.path.join(os.path.dirname(transformers.__file__), 'integrations', 'moe.py'))" 2>/dev/null || true)
if [ -n "$MOE_FILE" ] && [ -f "$MOE_FILE" ] && grep -q "^from __future__ import annotations" "$MOE_FILE"; then
    sed -i "s/^from __future__ import annotations$/# from __future__ import annotations (patched)/" "$MOE_FILE"
fi

echo "[$(date -u +%FT%TZ)] === q35 BASE eval | model=$MODEL | $HE_TASK | MODE=$MODE | limit=$TASK_LIMIT | OUTDIR=$OUTDIR ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head

TASKS=$(ls "$REPO/data/humaneval/${HE_TASK#humaneval-}/humaneval_"*.csv | xargs -n1 basename | sed 's/\.csv$//' | sed "s/^/${HE_TASK}-/" | tr '\n' ' ')
if [ "$TASK_LIMIT" -gt 0 ]; then TASKS=$(echo $TASKS | tr ' ' '\n' | head -n "$TASK_LIMIT" | tr '\n' ' '); fi
NT=$(echo $TASKS | wc -w)
mn="${MODE#--}"
LOG="$LOGDIR/eval_q35base_${WTAG}_${mn}.log"
echo "[$(date -u +%FT%TZ)] tasks=$NT  log=$LOG"

n=0; ok=0
for T in $TASKS; do
  marker="$DONE_DIR/q35base_${WTAG}_${mn}_${T}.done"
  if [ -f "$marker" ]; then n=$((n+1)); ok=$((ok+1)); continue; fi
  # shellcheck disable=SC2086
  if python eval_by_claude.py --model "$MODEL" --task "$T" --split_type random \
        --gen-shots zero --disc-shots zero "$MODE" \
        --validator-log-odds --save-scores-csv --outputs-dir "$OUTDIR" >> "$LOG" 2>&1; then
    touch "$marker"; ok=$((ok+1))
  else
    echo "[$(date -u +%FT%TZ)]   FAIL task $T (see $LOG)"
  fi
  n=$((n+1))
  [ $((n % 10)) -eq 0 ] && echo "[$(date -u +%FT%TZ)]   $WTAG $mn: $n/$NT ($ok ok)"
done
echo "[$(date -u +%FT%TZ)] ======== BASE EVAL $WTAG $mn COMPLETE ($ok/$NT ok) ========"
