#!/bin/bash
#SBATCH --job-name=qwbase-rosch
#SBATCH --partition=allnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=/datastor2/jdr/logs/%j.out
#SBATCH --error=/datastor2/jdr/logs/%j.err
#
# Few-shot-disc base eval of Qwen/Qwen3.5-9B on the 10 rosch tasks.
# WHY: the existing base row (v6-Qwen_Qwen3.5-9B, May-25) was evaluated with
# disc_shots=ZERO, inconsistent with the few-trained models on validator-dependent
# metrics (rho, ROC_V, Acc_V). This re-evals the base with disc_shots=FEW.
# Mirrors run_eval_base_on_one_rosch.sh (already disc-shots few) but for qwen + the
# qwen35 venv + outputs-rerun-wandb. eval_by_claude.py names HF "org/name" models
# v6-<org>_<name> -> "v6-Qwen_Qwen3.5-9B", which the table builder's Base row matches.
# The new files (today's date) supersede the old zero ones via the builder's newest-wins
# dedup (SEARCH_DIRS includes both outputs/ and outputs-rerun-wandb/).
set -euo pipefail

REPO=/datastor2/jdr/rankalign
VENV=/datastor2/jdr/venvs/qwen35
BASE_MODEL=Qwen/Qwen3.5-9B
OUTPUTS_DIR=$REPO/outputs-rerun-wandb

export HF_HOME=/datastor2/jdr/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

source "$VENV/bin/activate"

# moe.py patch (idempotent) — transformers 5.x custom_op infer_schema vs string annotations
MOE=$(python -c "import transformers, os; print(os.path.join(os.path.dirname(transformers.__file__), 'integrations', 'moe.py'))" 2>/dev/null || true)
if [ -n "$MOE" ] && grep -q "^from __future__ import annotations" "$MOE" 2>/dev/null; then
    sed -i "s/^from __future__ import annotations$/# from __future__ import annotations (patched)/" "$MOE"
fi

cd "$REPO/scripts"
mkdir -p "$OUTPUTS_DIR"

ROSCH_TASKS=(rosch-bird rosch-carpenters-tool rosch-clothing rosch-fruit
             rosch-furniture rosch-sport rosch-toy rosch-vehicle
             rosch-vegetable rosch-weapon)

for EVAL_TASK in "${ROSCH_TASKS[@]}"; do
    for TC_TAG in self neg; do
        if [ "$TC_TAG" = "self" ]; then TC_FLAGS="--self-typicality"; else TC_FLAGS="--neg-typicality"; fi
        echo "[$(date -u +%FT%TZ)] [qwbase][$TC_TAG] $EVAL_TASK"
        python eval_by_claude.py \
            --model "$BASE_MODEL" \
            --task "$EVAL_TASK" \
            --split_type random \
            --disc-shots few \
            --gen-shots zero \
            --outputs-dir "$OUTPUTS_DIR" \
            --validator-log-odds \
            $TC_FLAGS \
            --save-scores-csv
    done
done
echo "[$(date -u +%FT%TZ)] === qwen base rosch few-shot eval DONE ==="
