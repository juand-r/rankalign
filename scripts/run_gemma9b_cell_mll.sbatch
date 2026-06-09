#!/bin/bash
#SBATCH --job-name=qwen35-mll
#SBATCH --partition=allnodes
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/datastor2/jdr/logs/%j.out
#SBATCH --error=/datastor2/jdr/logs/%j.err
#
# run_qwen35_cell_mll.sbatch DATASET SETTING
# ---------------------------------------------------------------------------
# mll port of scripts/run_qwen35_cell.sh: trains Qwen/Qwen3.5-9B on one
# (dataset x setting) cell then evals, on 2x A40 via the gemma4 venv.
# Based on the gemma-2-9b-it ifeval 2-GPU path (_overnight_launch.sh) + the
# proven qwen recipe (run_qwen35_cell.sh).
#
# Submit:   sbatch scripts/run_qwen35_cell_mll.sbatch ifeval s1
#   DATASET: persona | membership | ifeval
#   SETTING: s1 | s2 | s3 | s4 | s7 | s13   (the wandb-rerun matrix; see docs/RERUN_WANDB_TODO.md)
#
# Env knobs:
#   NO_BASE=1     eval drops --base-typicality (faster; self-/neg- CSVs only)
#   TRAIN_ONLY=1  stop after training (skip eval)
#   EVAL_ONLY=1   skip training; eval an existing checkpoint
#   EPOCHS=N      override num_epochs (DEFAULT 3 — used for ALL settings incl. s13)
#
# KEY DIFFERENCES vs the pod cell (run_qwen35_cell.sh):
#   - VENV = /datastor2/jdr/venvs/gemma4  (transformers 5.x; NOT venv_lexcons)
#   - mll paths; MODELS_DIR = /datastor2/jdr/rankalign/models2-rerun-wandb (separate
#     from models2/ so the wandb reruns never collide with the originals)
#   - 2 GPUs (gres=gpu:2) so the 9B + LoRA fits (1x A40 OOMs in qwen3.5 attn)
#   - wandb ONLINE (mll is logged in via ~/.netrc) -> project "rankalign", with
#     --wandb_run_name "rerun-wandb-qwen3.5-9b-<dataset>-<setting>" so the new runs
#     are filterable vs the old ones on the cloud. Still --no-upload-hf.
#   - canonical trainer ranking_loss_ref_fix.py (NEVER ranking_loss_ref_qwen.py, obsolete)
#   - settings s1/s2/s4/s7 mirror run_qwen35_cell.sh; s3 (New+fsx) and s13 (SFT+cft)
#     added for the rerun. ALL settings (incl. s13) train 3 epochs (delta-bins 10).
#     (The original qwen s13 was fixed-d0.15/1-epoch; this 3-epoch rerun is intentional —
#     fuller curve + reaches epoch2.)
set -euo pipefail

DATASET="${1:?DATASET required (persona|membership|ifeval)}"
SETTING="${2:?SETTING required (s1|s2|s3|s4|s7|s13)}"
MODEL="google/gemma-2-9b-it"

# qwen35 venv = clone of gemma4 + flash-linear-attention (fla) for the Qwen3.5 gated-delta-rule
# fast kernel (gemma4 alone -> torch fallback -> ~11.5 s/it on A40). Overridable via VENV=.
VENV="${VENV:-/datastor2/jdr/venvs/gemma4}"
REPO=/datastor2/jdr/rankalign
# Rerun batch -> separate models dir so reruns never collide with the originals in models2/.
MODELS_DIR=/datastor2/jdr/rankalign/models2-rerun-wandb
# Rerun eval scores go to a SEPARATE dir so they never overwrite the canonical outputs/
# (these models already exist; we re-ran for wandb + the disc-shots fix). Overridable via env.
OUTPUTS_DIR="${OUTPUTS_DIR:-/datastor2/jdr/rankalign/outputs-rerun-wandb}"
LOG_DIR=/datastor2/jdr/logs
mkdir -p "$MODELS_DIR" "$OUTPUTS_DIR" "$LOG_DIR"

echo "[$(date -u +%FT%TZ)] === run_qwen35_cell_mll DATASET=$DATASET SETTING=$SETTING job=${SLURM_JOB_ID:-local} host=$(hostname) ==="
nvidia-smi -L 2>/dev/null || true

source "$VENV/bin/activate"
cd "$REPO/scripts"

export HF_HOME=/datastor2/jdr/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
# wandb: leave default (ONLINE) -> project "rankalign" via ~/.netrc creds.

# --- moe.py patch (idempotent) -------------------------------------------------
# transformers 5.8 integrations/moe.py uses `from __future__ import annotations`,
# making annotations strings; torch 2.4.1 custom_op() infer_schema() can't parse
# them -> `import peft` (LoRA) fails. Comment out the import. Safe no-op if absent.
MOE_FILE=$(python -c "import transformers, os; print(os.path.join(os.path.dirname(transformers.__file__), 'integrations', 'moe.py'))" 2>/dev/null || true)
if [ -n "$MOE_FILE" ] && [ -f "$MOE_FILE" ] && grep -q "^from __future__ import annotations" "$MOE_FILE"; then
    sed -i "s/^from __future__ import annotations$/# from __future__ import annotations (patched: torch 2.4.1 infer_schema compat)/" "$MOE_FILE"
    echo "[$(date -u +%FT%TZ)] Applied moe.py patch -> $MOE_FILE"
else
    echo "[$(date -u +%FT%TZ)] moe.py already patched / not present (ok)"
fi

# --- Dataset config (mirrors run_qwen35_cell.sh) -------------------------------
case "$DATASET" in
    persona)
        TASK="persona-v1"
        EVAL_TASKS=(persona-v1-psychopathy persona-v1-machiavellianism
                    persona-v1-narcissism persona-v1-desire-to-create-allies
                    persona-v1-interest-in-music persona-v1-interest-in-science)
        MAX_SEQ="" ; GRAD_CKP="" ; DISC_SHOTS="--disc-shots few" ;;
    membership)
        TASK="membership-sans-rosch-v0"
        EVAL_TASKS=(rosch-bird rosch-carpenters-tool rosch-clothing rosch-fruit
                    rosch-furniture rosch-sport rosch-toy rosch-vehicle
                    rosch-vegetable rosch-weapon)
        # membership/rosch uses FEW-shot disc — matches gemma-2-9b-it (run_gemma2_cell.sh).
        # NB: Qwen3+ auto-detects to zero, so this override is REQUIRED. The original qwen
        # membership runs (run_qwen35_cell.sh) wrongly used zero; this fixes that.
        MAX_SEQ="" ; GRAD_CKP="" ; DISC_SHOTS="--disc-shots few" ;;
    ifeval)
        TASK="ifeval-concat"
        EVAL_TASKS=()
        IFEVAL_DATA_DIR="$REPO/data/fixed-prompts-ifeval"
        # IFEVAL_SPLIT=ood -> prompts 1..21 (the held-out OOD set, ~20 after gaps; default).
        # IFEVAL_SPLIT=id  -> prompts >=22 (the in-domain set, ~79). Prompt IDs are NON-sequential
        # (gaps, e.g. no 14) — matches _build_ifeval_table_v7.py's ood/id definition.
        IFEVAL_SPLIT="${IFEVAL_SPLIT:-ood}"
        while IFS= read -r n; do EVAL_TASKS+=("ifeval-prompt_$n"); done < <(
            for f in "$IFEVAL_DATA_DIR"/gpt_ifeval_results_prompt_*.jsonl; do
                basename "$f" | grep -oE '[0-9]+' | head -1
            done | awk -v s="$IFEVAL_SPLIT" '(s=="ood" && $1+0<=21) || (s=="id" && $1+0>=22)' | sort -n )
        MAX_SEQ="--max-seq-len 1024"
        GRAD_CKP="--gradient_checkpointing"
        # ifeval: make_prompt_ifeval doesn't implement few-shot — MUST stay zero (all models).
        DISC_SHOTS="--disc-shots zero" ;;
    *) echo "Unknown DATASET: $DATASET"; exit 1 ;;
esac

# --- Per-setting flags (mirrors run_qwen35_cell.sh; +s3/s13 for the wandb rerun) ---
CFT_FLAG=""   # only s13 sets this (--consistency-ft)
case "$SETTING" in
    s1)
        LOSS_FLAGS="--nll_validator_weight 1 --nll_generator_weight 1 --preference_loss_weight 0"
        SEMI_FLAG="--labeled-only 0.1" ; FSX_FLAG="" ; TC_FLAG="" ; VLO_FLAG="--validator-log-odds"
        PPD_FLAGS="" ; EVAL_MODES="--self-typicality --neg-typicality"
        DIR_SUFFIX="--full-completion--pref0.0--nllv1.0--nllg1.0--vallogodds--labelonly0.1--fix1" ;;
    s2)
        LOSS_FLAGS="--nll_validator_weight 0 --nll_generator_weight 0 --preference_loss_weight 1"
        SEMI_FLAG="--semi-supervised 0.1" ; FSX_FLAG="" ; TC_FLAG="" ; VLO_FLAG="--validator-log-odds"
        PPD_FLAGS="" ; EVAL_MODES="--self-typicality --neg-typicality"
        DIR_SUFFIX="--full-completion--vallogodds--semi0.1--fix1" ;;
    s3)   # New+fsx (comb + fsx + vlo, NO TC); eval self+neg (no-TC trained)
        LOSS_FLAGS="--nll_validator_weight 1 --nll_generator_weight 1 --preference_loss_weight 1"
        SEMI_FLAG="--semi-supervised 0.1" ; FSX_FLAG="--force-same-x" ; TC_FLAG=""
        VLO_FLAG="--validator-log-odds" ; PPD_FLAGS="--per-prompt-delta --shape-budget-mode global"
        EVAL_MODES="--self-typicality --neg-typicality"
        DIR_SUFFIX="--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1" ;;
    s4)
        LOSS_FLAGS="--nll_validator_weight 1 --nll_generator_weight 1 --preference_loss_weight 1"
        SEMI_FLAG="--semi-supervised 0.1" ; FSX_FLAG="--force-same-x" ; TC_FLAG="--self-typicality"
        VLO_FLAG="--validator-log-odds" ; PPD_FLAGS="--per-prompt-delta --shape-budget-mode global"
        EVAL_MODES="--self-typicality"
        DIR_SUFFIX="--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1" ;;
    s7)
        LOSS_FLAGS="--nll_validator_weight 1 --nll_generator_weight 1 --preference_loss_weight 1"
        SEMI_FLAG="--semi-supervised 0.1" ; FSX_FLAG="--force-same-x" ; TC_FLAG="--neg-typicality"
        VLO_FLAG="--validator-log-odds" ; PPD_FLAGS="--per-prompt-delta --shape-budget-mode global"
        EVAL_MODES="--neg-typicality"
        DIR_SUFFIX="--tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1" ;;
    s13)  # SFT + consistency-ft (pref0/nll1, --consistency-ft, labeled-only; no fsx/TC/vlo)
        # NB: original qwen s13 paper model was FIXED delta0.15, 1 epoch. This rerun uses the
        # launcher default (delta-bins 10, 3 epochs) for consistency + a fuller curve + epoch2.
        # To reproduce the original exactly: EPOCHS=1 (and you'd drop --delta-bins, see below).
        LOSS_FLAGS="--nll_validator_weight 1 --nll_generator_weight 1 --preference_loss_weight 0"
        SEMI_FLAG="--labeled-only 0.1" ; FSX_FLAG="" ; TC_FLAG="" ; VLO_FLAG=""
        PPD_FLAGS="" ; CFT_FLAG="--consistency-ft" ; EVAL_MODES="--self-typicality --neg-typicality"
        DIR_SUFFIX="--full-completion--pref0.0--nllv1.0--nllg1.0--cft--labelonly0.1--fix1" ;;
    *) echo "Unknown SETTING: $SETTING (s1|s2|s3|s4|s7|s13)"; exit 1 ;;
esac

LORA_FLAG="--lora"   # qwen uses plain --lora (NOT --gemma4-lora)
MODEL_REPL=$(echo "$MODEL" | sed 's|/|--|g')
GLOB_PATTERN="${MODELS_DIR}/v7-${MODEL_REPL}-delta*-epoch[012]--${TASK}-all--d2g--random--alpha1.0${DIR_SUFFIX}_merged"
echo "[$(date -u +%FT%TZ)] model dir glob: $GLOB_PATTERN"

# ============================ TRAIN ============================
echo "[$(date -u +%FT%TZ)] === TRAIN: $SETTING x $TASK ==="
EXISTING=$(ls -dt $GLOB_PATTERN 2>/dev/null | head -1 || true)
# MODEL_DIR_OVERRIDE bypasses the setting->suffix glob (gemma s1/s2 reruns use the
# _eval_only.sh naming, which differs from the qwen sbatch's per-setting flags).
[ -n "${MODEL_DIR_OVERRIDE:-}" ] && EXISTING="$MODEL_DIR_OVERRIDE"
if [ -n "${EVAL_ONLY:-}" ]; then
    [ -z "$EXISTING" ] && { echo "ERROR: EVAL_ONLY but no model dir: $GLOB_PATTERN"; exit 1; }
    MODEL_DIR="$EXISTING"; echo "[$(date -u +%FT%TZ)] EVAL_ONLY — using $MODEL_DIR"
elif [ -n "$EXISTING" ]; then
    MODEL_DIR="$EXISTING"; echo "[$(date -u +%FT%TZ)] model dir exists, skipping train: $MODEL_DIR"
else
    # wandb ONLINE; name each run so the rerun is filterable vs old runs on the cloud.
    # WANDB_SUFFIX lets a batch tag its runs (e.g. "-discfew") to keep them distinct on wandb.
    WANDB_RUN_NAME="rerun-wandb-qwen3.5-9b-${DATASET}-${SETTING}${WANDB_SUFFIX:-}"
    # shellcheck disable=SC2086
    python ranking_loss_ref_fix.py \
        --model "$MODEL" \
        --num_epochs "${EPOCHS:-3}" \
        --task "$TASK" \
        --train_g_or_d g \
        --split_type random \
        --all \
        --delta 0.15 \
        --delta-bins 10 \
        --models-dir "$MODELS_DIR" \
        --wandb_run_name "$WANDB_RUN_NAME" \
        $LOSS_FLAGS $SEMI_FLAG $FSX_FLAG $TC_FLAG $VLO_FLAG $PPD_FLAGS $CFT_FLAG \
        $LORA_FLAG $MAX_SEQ $GRAD_CKP $DISC_SHOTS \
        --no-upload-hf
    MODEL_DIR=$(ls -dt $GLOB_PATTERN 2>/dev/null | head -1 || true)
    [ -z "$MODEL_DIR" ] && { echo "ERROR: no model dir after train: $GLOB_PATTERN"; exit 1; }
    echo "[$(date -u +%FT%TZ)] Train complete: $MODEL_DIR"
fi

if [ -n "${TRAIN_ONLY:-}" ]; then
    echo "[$(date -u +%FT%TZ)] TRAIN_ONLY=1 — done (skipping eval)."; exit 0
fi

# ============================ EVAL ============================
echo "[$(date -u +%FT%TZ)] === EVAL: ${#EVAL_TASKS[@]} tasks, modes='$EVAL_MODES' ==="
# Name the eval symlink with the canonical abbreviated v7 descriptor (to_hf_repo_name) so
# scores_ filenames carry full provenance (model/delta/epoch/method) and parse with
# checkpoint_name_parser — NOT a generic eval_model_qw_<setting> (which loses all of that and
# breaks _build_*_table_v7.py). The abbreviated form is short enough for the 255-char cap.
EVAL_NAME=$(PYTHONPATH="$REPO/src" python -c "from checkpoint_name_parser import parse_checkpoint_name, to_hf_repo_name; import os; print(to_hf_repo_name(parse_checkpoint_name(os.path.basename('$MODEL_DIR')), prefix=''))")
[ -z "$EVAL_NAME" ] && { echo "ERROR: could not derive abbreviated eval name from $MODEL_DIR"; exit 1; }
EVAL_MODEL_DIR="$REPO/$EVAL_NAME"
ln -sfn "$MODEL_DIR" "$EVAL_MODEL_DIR"
if [ -n "${NO_BASE:-}" ]; then
    BASE_FLAGS=""; echo "[$(date -u +%FT%TZ)] NO_BASE=1 — self-/neg- only"
else
    BASE_FLAGS="--base-typicality --base-model-name $MODEL"
fi
for MODE in $EVAL_MODES; do
    for EVAL_TASK in "${EVAL_TASKS[@]}"; do
        echo "[$(date -u +%FT%TZ)] eval: $EVAL_TASK ($MODE)"
        python eval_by_claude.py \
            --model "$EVAL_MODEL_DIR" --task "$EVAL_TASK" --split_type random \
            $DISC_SHOTS --gen-shots zero --outputs-dir "$OUTPUTS_DIR" \
            --validator-log-odds $MODE $BASE_FLAGS --save-scores-csv
    done
done
echo "[$(date -u +%FT%TZ)] === ALL DONE: DATASET=$DATASET SETTING=$SETTING ==="
