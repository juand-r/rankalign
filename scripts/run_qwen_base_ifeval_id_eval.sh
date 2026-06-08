#!/bin/bash
#SBATCH --job-name=qwbase-ifid
#SBATCH --partition=allnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/datastor2/jdr/logs/%j.out
#SBATCH --error=/datastor2/jdr/logs/%j.err
# Qwen/Qwen3.5-9B BASE eval on the ifeval IN-DOMAIN (id) prompts (>=22, ~79), self+neg,
# disc-shots ZERO (correct for ifeval, all models). Fills the base row of the ifeval-ID
# table. -> outputs-rerun-wandb (model_short v6-Qwen_Qwen3.5-9B). Resumable (skip-if-exists).
set -uo pipefail
REPO=/datastor2/jdr/rankalign; VENV=/datastor2/jdr/venvs/qwen35
export HF_HOME=/datastor2/jdr/hf_cache HF_HUB_CACHE=/datastor2/jdr/hf_cache/hub HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
source "$VENV/bin/activate"
MOE=$(python -c "import transformers,os;print(os.path.join(os.path.dirname(transformers.__file__),'integrations','moe.py'))" 2>/dev/null||true)
[ -n "$MOE" ] && grep -q "^from __future__ import annotations" "$MOE" 2>/dev/null && sed -i "s/^from __future__ import annotations$/# patched/" "$MOE"
cd "$REPO/scripts"
IDS=$(for f in "$REPO"/data/fixed-prompts-ifeval/gpt_ifeval_results_prompt_*.jsonl; do basename "$f"|grep -oE '[0-9]+'|head -1; done | awk '$1+0>=22' | sort -n)
for n in $IDS; do for TC in self neg; do
  [ "$TC" = self ] && F="--self-typicality" || F="--neg-typicality"
  echo "[$(date -u +%FT%TZ)] [qwbase-ifid][$TC] ifeval-prompt_$n"
  python eval_by_claude.py --model Qwen/Qwen3.5-9B --task "ifeval-prompt_$n" --split_type random \
    --disc-shots zero --gen-shots zero --outputs-dir "$REPO/outputs-rerun-wandb" \
    --validator-log-odds $F --save-scores-csv
done; done
echo "[$(date -u +%FT%TZ)] === qwen base ifeval ID eval DONE ==="
