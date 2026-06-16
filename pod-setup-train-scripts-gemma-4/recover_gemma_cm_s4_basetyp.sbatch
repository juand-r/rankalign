#!/bin/bash
#SBATCH --job-name=g4cm-s4-recov
#SBATCH --partition=allnodes
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=9
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=/datastor2/jdr/logs/%j.out
#SBATCH --error=/datastor2/jdr/logs/%j.err
#
# Recover the 12 gemma-4-31B-it MULTI s4 base-typ (bt1) eval scores that failed when
# /datastor1 hit 0 bytes free: eval_by_claude.py writes a cwd-relative summary log
# (../outputs/eval_results.csv), and the gemma run's cwd was /datastor1 -> ENOSPC ->
# the task aborted before saving those 12 score CSVs (the other 70 + multi-s4-self
# succeeded). This re-run uses the /datastor2 checkout as cwd, so the summary write
# lands on /datastor2 (117T free) and cannot recur. Same eval flags as the 70 that
# succeeded, same merged model + outputs dir -> identical filenames -> cell becomes 82/82.
set -euo pipefail
REPO=/datastor2/jdr/rankalign
VENV=/datastor2/jdr/venvs/gemma4
MODEL=/datastor2/jdr/rankalign/gemma-4-models-mll-tmp-multi/v7-google--gemma-4-31B-it-delta3.71-epoch2--humaneval-v2.1correct-multi-all--d2g--random--alpha1.0--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1
BASE=google/gemma-4-31B-it
OUTDIR=/datastor2/jdr/rankalign/outputs_gemma4_mll_tmp-multi
IDS="8 9 80 81 82 83 85 87 88 91 94 99"

export HF_HOME=/datastor2/jdr/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
export HF_TOKEN=$(cat /datastor2/jdr/.hftoken 2>/dev/null || true)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled   # eval; no wandb writes anywhere
export TOKENIZERS_PARALLELISM=false

source "$VENV/bin/activate"
cd "$REPO/scripts"   # cwd on /datastor2 -> summary log ../outputs/eval_results.csv lands on /datastor2
echo "[$(date -u +%FT%TZ)] recover gemma cm s4 basetyp | cwd=$(pwd) | model=$(basename "$MODEL")"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head

n=0; ok=0
for id in $IDS; do
  T="humaneval-v2.1correct-multi-humaneval_${id}"
  n=$((n+1))
  echo "[$(date -u +%FT%TZ)] ($n/12) eval $T"
  if python eval_by_claude.py --model "$MODEL" --task "$T" --split_type random \
        --gen-shots zero --disc-shots zero --self-typicality \
        --base-typicality --base-model "$BASE" \
        --validator-log-odds --save-scores-csv --outputs-dir "$OUTDIR"; then
    ok=$((ok+1))
  else
    echo "[$(date -u +%FT%TZ)]   FAIL $T"
  fi
done
echo "[$(date -u +%FT%TZ)] ======== DONE recover cm s4 basetyp ($ok/12 ok) ========"
