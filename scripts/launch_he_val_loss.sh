#!/bin/bash
# Submit the HumanEval held-out (test-set) loss jobs — the §2.3 mirror.
#
# Grid: 5 settings (s1 SFT, s2 RankAlign, s3 New+fsx, s4 selfTC, s7 negTC)
#       x 2 models (gemma4, qwen) x 2 datasets (upper, multi) = 20 jobs.
# Each job loops base+ep0+ep1+ep2 internally and writes its OWN per-job CSV.
# gemma4 -> gpu:4 ; qwen -> gpu:1.
#
# Usage (run ON mll, from the repo root):
#   CANARY=1 bash scripts/launch_he_val_loss.sh     # one capped fast cell (gemma4 upper s4)
#   bash scripts/launch_he_val_loss.sh              # all 20 full jobs
set -uo pipefail
SB=scripts/mll_he_val_loss.sbatch
SETTINGS="${SETTINGS:-1 2 3 4 7}"

submit() {  # submit <model> <he_task> <setting> [extra sbatch args...]
  local model="$1" task="$2" s="$3"; shift 3
  local ds; ds=$(case "$task" in *upper) echo cu;; *multi) echo cm;; esac)
  local gres; [ "$model" = qwen ] && gres="--gres=gpu:1" || gres="--gres=gpu:4"
  MODEL="$model" HE_TASK="$task" sbatch $gres --job-name="vl-${ds}-${model}-s${s}" "$@" "$SB" "$s"
}

if [ -n "${CANARY:-}" ]; then
  echo "=== CANARY: gemma4 upper s4, --max-items-per-class=15 (validates LoRA load + aggregation + CSV) ==="
  MAXITEMS=15 OUTSUFFIX=_canary submit gemma4 humaneval-v2.1correct-upper 4
  exit 0
fi

echo "=== Submitting 20 held-out val-loss jobs ==="
for model in gemma4 qwen; do
  for task in humaneval-v2.1correct-upper humaneval-v2.1correct-multi; do
    for s in $SETTINGS; do
      submit "$model" "$task" "$s"
    done
  done
done
echo "=== all submitted; squeue -u $USER ==="
