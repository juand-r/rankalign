#!/bin/bash
# mll_eval_base.sbatch — eval base google/gemma-4-31B-it on humaneval-v2.1correct-upper
# (or any other humaneval-v2.x dataset on disk). Mirrors mll_train_arm.sbatch's
# infra conventions; produces scores_self-* and scores_neg-* CSVs.
#
# Submit (does NOT auto-run — you launch it):
#   sbatch pod-setup-train-scripts-gemma-4/mll_eval_base.sbatch self_tc
#   sbatch pod-setup-train-scripts-gemma-4/mll_eval_base.sbatch neg_tc
#   sbatch pod-setup-train-scripts-gemma-4/mll_eval_base.sbatch both
#
# Override the dataset via env:
#   HE_TASK=humaneval-v2.1correct-multi sbatch ... mll_eval_base.sbatch self_tc
#
# Override the output dir via env (default writes to the same dir the
# _build_humaneval_table.py already scans, so no builder edit is needed):
#   OUTDIR=/datastor2/jdr/outputs_gemma4_mll sbatch ... mll_eval_base.sbatch self_tc
#
# Prereqs (already set up per docs/mll_gemma4_training_setup.md):
#   - venv:  /datastor2/jdr/venvs/gemma4
#   - repo:  /datastor1/jdr/gv-gap/rankalign on longform
#   - token: HF_TOKEN in env, or in /datastor2/jdr/.hftoken (gated gemma-4 dl)
#
# Why 3 GPUs (no more, no less):
#   - base model only (1 × gemma-4-31B-it ≈ 62 GB bf16); no --base-typicality
#     means no second model is ever resident, so 3 × A40 = 144 GB is plenty
#     of headroom. 2 × A40 = 96 GB would also fit but is the floor.
#
#SBATCH --job-name=g4-eval-base
#SBATCH --partition=p-gdurret
#SBATCH --account=gdurrett
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/datastor2/jdr/logs/slurm-%x-%j.log
set -uo pipefail

MODE="${1:?usage: sbatch mll_eval_base.sbatch <self_tc|neg_tc|both>}"
case "$MODE" in
  self_tc) MODES="--self-typicality" ;;
  neg_tc)  MODES="--neg-typicality" ;;
  both)    MODES="--self-typicality --neg-typicality" ;;
  *) echo "FATAL: unknown mode '$MODE' (self_tc|neg_tc|both)"; exit 2 ;;
esac

HE_TASK="${HE_TASK:-humaneval-v2.1correct-upper}"
DATASET_DIR="${HE_TASK#humaneval-}"

# --- mll paths (overridable; defaults land where the table builder already scans) ---
export RANKALIGN_DIR=/datastor1/jdr/gv-gap/rankalign
export VENV_DIR=/datastor2/jdr/venvs/gemma4
export OUTDIR="${OUTDIR:-/datastor2/jdr/rankalign/outputs_gemma4_from_pod}"
export LOGDIR=/datastor2/jdr/logs
export HF_HOME=/datastor2/jdr/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- HF token: env, else /datastor2/jdr/.hftoken (never committed) ---
if [ -z "${HF_TOKEN:-}" ] && [ -f /datastor2/jdr/.hftoken ]; then
    HF_TOKEN=$(cat /datastor2/jdr/.hftoken)
fi
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN not set and /datastor2/jdr/.hftoken missing}"
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

mkdir -p "$LOGDIR" "$OUTDIR" "$HF_HUB_CACHE" "$OUTDIR/.done_base"

echo "=== mll gemma-4 base-eval $MODE $HE_TASK : node $(hostname) job $SLURM_JOB_ID ==="
echo "GPUs visible: ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

source "$VENV_DIR/bin/activate"
"$VENV_DIR/bin/python" -c "import torch; assert torch.cuda.is_available(), 'no CUDA in job'; print('cuda OK,', torch.cuda.device_count(), 'GPUs')"
cd "$RANKALIGN_DIR/scripts"

MODEL=google/gemma-4-31B-it
MODEL_SLUG=$(echo "$MODEL" | sed 's|/|--|')

# All per-task task names for this dataset (e.g. humaneval-v2.1correct-upper-humaneval_42)
TASKS=$(ls "$RANKALIGN_DIR/data/humaneval/${DATASET_DIR}/humaneval_"*.csv \
    | xargs -n1 basename | sed 's/\.csv$//' \
    | sed "s/^/${HE_TASK}-/" | tr '\n' ' ')
NT=$(echo "$TASKS" | wc -w)
LOG="$LOGDIR/eval_base_${MODE}_${HE_TASK}.log"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
log "MODE=$MODE  HE_TASK=$HE_TASK  tasks=$NT  OUTDIR=$OUTDIR"

# Idempotent: per (model, mode, task) done markers, so re-submission resumes.
for M in $MODES; do
    mname="${M#--}"
    log "==== eval base $mname ($MODE arm) — $HE_TASK ===="
    n=0; skipped=0
    for T in $TASKS; do
        marker="$OUTDIR/.done_base/${MODEL_SLUG}_${mname}_${T}.done"
        if [ -f "$marker" ]; then skipped=$((skipped+1)); n=$((n+1)); continue; fi
        python eval_by_claude.py --model "$MODEL" --task "$T" --split_type random \
            --gen-shots zero --disc-shots zero "$M" \
            --validator-log-odds --save-scores-csv \
            --outputs-dir "$OUTDIR" \
            >> "$LOGDIR/eval_base_${mname}_${HE_TASK}.log" 2>&1 \
            && touch "$marker"
        n=$((n+1))
        [ $((n % 10)) -eq 0 ] && log "  $mname: $n/$NT (skipped=$skipped)"
    done
    log "  base $mname DONE: $n/$NT (skipped=$skipped)"
done

log "============ ALL DONE — base, $MODE, $HE_TASK ============"
