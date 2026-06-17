#!/bin/bash
#SBATCH --job-name=g4tr
# NOTE: static fallback. Override at submit time with --job-name=g4${WTAG}-s${S}
# (cu=correct-upper, cm=correct-multi) so upper/multi are distinct in squeue.
#SBATCH --partition=allnodes
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=9
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/datastor2/jdr/logs/%j.out
#SBATCH --error=/datastor2/jdr/logs/%j.err
#
# Train + eval ONE gemma-4-31B-it setting on humaneval-v2.1correct-upper|multi, on mll.
# This is the GENCTX launcher for the NEW settings ONLY:
#   s1 (SFT) / s3 (New+fsx) / s7 (New+fsx+neg-tc, FLORA-Neg) / s13 (SFT+consistency-ft).
# s2/s4 are NOT here — those were trained with the ORIGINAL ranking_loss_ref_fix.py and their
# faithful record is the sibling launcher mll_g4_train_eval.sbatch (do not run s2/s4 here).
# --gemma4-lora --lora --delta-bins 10.
#
# Uses the genctx trainer copy ranking_loss_ref_fix_genctx.py (NOT the original
# ranking_loss_ref_fix.py, which is left untouched): it drops the ~6 humaneval BPE-seam items
# whose prompt->completion tokenization is inconsistent, instead of hard-aborting. s3/s7 use
# --force-same-x, which pairs items differently and could surface such an item mid-run.
# When no mismatch occurs the drop filter is a no-op -> identical to fix.py. WANDB online.
#
# Eval: the epoch-2 adapter, for EACH eval mode x {bt1 = base-typicality, bt0 = no base}:
#   s1, s3, s13:  {self,neg} x {bt1,bt0} = 4 score-sets each
#   s7:           {neg}      x {bt1,bt0} = 2 score-sets
# scores_ files for ALL 82 humaneval test tasks (upper or multi) per score-set.
#
# allnodes / 9 CPU / 96G / 4xA40 (raca defaults; gpu:4 gives eval-bt1 headroom — it loads the
# trained model AND a frozen base ~= 124G). NOTHING in $HOME: HF cache + models + outputs on
# /datastor2. Idempotent: per-(epoch,mode,bt,task) done-markers; resubmitting resumes.
#
# Usage:  sbatch mll_g4_train_eval_genctx.sbatch <1|3|7|13>
# Smoke:  SMOKE=1 sbatch mll_g4_train_eval_genctx.sbatch <1|3|7|13>  (1 epoch, 64 samples, 2 tasks, -smoke dirs)
set -uo pipefail
S="${1:?usage: mll_g4_train_eval_genctx.sbatch <1|3|7|13>}"

REPO=/datastor2/jdr/rankalign   # MUST be /datastor2: cd $REPO/scripts sets cwd, and
# eval_by_claude.py writes a cwd-relative summary log (../outputs/eval_results.csv).
# When this was /datastor1 (nearly full) it hit ENOSPC and killed 12 evals (2026-06-16).
VENV=/datastor2/jdr/venvs/gemma4
HE_TASK="${HE_TASK:-humaneval-v2.1correct-upper}"   # override e.g. humaneval-v2.1correct-multi
WTAG=$(case "$HE_TASK" in *correct-upper) echo cu;; *correct-multi) echo cm;; *) echo "${HE_TASK##*-}";; esac)  # wandb run tag (cu/cm)
MODEL="google/gemma-4-31B-it"; BASE="google/gemma-4-31B-it"
LOGDIR=/datastor2/jdr/logs
NUM_EPOCHS=3 ; TOTAL_SAMPLES=5110 ; EVAL_EPOCHS="${EVAL_EPOCHS:-2}"
MODELS_DIR="${MODELS_DIR:-/datastor2/jdr/rankalign/gemma-4-models-mll-tmp}"
OUTDIR="${OUTDIR:-/datastor2/jdr/rankalign/outputs_gemma4_mll_tmp}"
TASK_LIMIT=0   # 0 = all tasks

if [ -n "${SMOKE:-}" ]; then
    NUM_EPOCHS=1 ; TOTAL_SAMPLES=64 ; TASK_LIMIT=2 ; EVAL_EPOCHS=0
    MODELS_DIR="/datastor2/jdr/rankalign/gemma-4-models-mll-smoke"
    OUTDIR="/datastor2/jdr/rankalign/outputs_gemma4_mll_smoke"
fi
DONE_DIR="$OUTDIR/.done"

# ---- env (everything off $HOME) ----
export HF_HOME=/datastor2/jdr/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
export HF_TOKEN=$(cat /datastor2/jdr/.hftoken)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# WANDB online — get the curves. Key from ~/.bashrc (non-interactive shells don't source it).
export WANDB_API_KEY=$(grep -oP 'WANDB_API_KEY=\K\S+' /u/jdr/.bashrc 2>/dev/null | head -1)
if [ -n "$WANDB_API_KEY" ]; then export WANDB_MODE=online; else echo "[WARN] no WANDB_API_KEY -> OFFLINE"; export WANDB_MODE=offline; fi
export WANDB_PROJECT=rankalign

source "$VENV/bin/activate"
cd "$REPO/scripts"
mkdir -p "$MODELS_DIR" "$OUTDIR" "$DONE_DIR" "$LOGDIR"

echo "[$(date -u +%FT%TZ)] === g4 s$S on $HE_TASK | epochs=$NUM_EPOCHS samples=$TOTAL_SAMPLES smoke=${SMOKE:-0} | MODELS_DIR=$MODELS_DIR OUTDIR=$OUTDIR ==="
echo "[$(date -u +%FT%TZ)] node=$(hostname) gpus=$CUDA_VISIBLE_DEVICES wandb=$WANDB_MODE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head

# ---- test-task list (all humaneval_*.csv) ----
TASKS=$(ls "$REPO/data/humaneval/${HE_TASK#humaneval-}/humaneval_"*.csv | xargs -n1 basename | sed 's/\.csv$//' | sed "s/^/${HE_TASK}-/" | tr '\n' ' ')
if [ "$TASK_LIMIT" -gt 0 ]; then TASKS=$(echo $TASKS | tr ' ' '\n' | head -n "$TASK_LIMIT" | tr '\n' ' '); fi
NT=$(echo $TASKS | wc -w)

# ---- per-setting flags ----
# s2 matches the verified pod recipe (run_settings_v21correct_upper.sh); s1/s3/s4/s7/s13 are
# the canonical settings from run_qwen35_cell_mll.sbatch (model-independent flags). gemma keeps
# a LoRA adapter (NO "_merged"), so ADAPTER_SUFFIX = "-all--d2g--random--alpha1.0" + canonical
# DIR_SUFFIX. s3==s4 except s4 adds --self-typicality; both use --validator-log-odds + ppd.
pref_w=1 nll_v_w=0 nll_g_w=0; semi_flag="--semi-supervised 0.1"; fsx=""; vlo=""; tc_train=""; ppd=""; sbm=""; cft=""
case "$S" in
  1) SETTING_NAME="SFT"
     nll_v_w=1; nll_g_w=1; pref_w=0; semi_flag="--labeled-only 0.1"; vlo="--validator-log-odds"
     ADAPTER_SUFFIX="-all--d2g--random--alpha1.0--full-completion--pref0.0--nllv1.0--nllg1.0--vallogodds--labelonly0.1--fix1"
     EVAL_MODES="--self-typicality --neg-typicality" ;;
  3) SETTING_NAME="New+fsx"
     nll_v_w=1; nll_g_w=1; fsx="--force-same-x"; vlo="--validator-log-odds"
     ADAPTER_SUFFIX="-all--d2g--random--alpha1.0--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1"
     EVAL_MODES="--self-typicality --neg-typicality" ;;
  7) SETTING_NAME="New+fsx+neg-tc"
     nll_v_w=1; nll_g_w=1; fsx="--force-same-x"; vlo="--validator-log-odds"; tc_train="--neg-typicality"
     ADAPTER_SUFFIX="-all--d2g--random--alpha1.0--tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1"
     EVAL_MODES="--neg-typicality" ;;
  13) SETTING_NAME="SFT+cft"
     nll_v_w=1; nll_g_w=1; pref_w=0; semi_flag="--labeled-only 0.1"; cft="--consistency-ft"
     ADAPTER_SUFFIX="-all--d2g--random--alpha1.0--full-completion--pref0.0--nllv1.0--nllg1.0--cft--labelonly0.1--fix1"
     EVAL_MODES="--self-typicality --neg-typicality" ;;
  *) echo "FATAL: genctx launcher handles 1|3|7|13 only (s2/s4 record -> mll_g4_train_eval.sbatch)"; exit 2 ;;
esac
TRAIN_FLAGS="--model $MODEL --num_epochs $NUM_EPOCHS --task $HE_TASK --train_g_or_d g --split_type random --nll_validator_weight $nll_v_w --nll_generator_weight $nll_g_w --preference_loss_weight $pref_w --all --delta 0.15 $semi_flag --disc-shots zero --lora --gradient_checkpointing --models-dir $MODELS_DIR --total_samples $TOTAL_SAMPLES --no-upload-hf"
if [ -n "$fsx" ]; then ppd="--per-prompt-delta"; sbm="--shape-budget-mode global"; fi
for f in "$fsx" "$ppd" "$sbm" "$vlo" "$tc_train" "$cft"; do [ -n "$f" ] && TRAIN_FLAGS="$TRAIN_FLAGS $f"; done

adapter_dir_for() {
  local found
  found=$(ls -d "${MODELS_DIR}/v7-google--gemma-4-31B-it-delta"*"-epoch${1}--${HE_TASK}${ADAPTER_SUFFIX}" 2>/dev/null | head -1)
  [ -z "$found" ] && found=$(ls -d "${MODELS_DIR}/v6-google--gemma-4-31B-it-delta"*"-epoch${1}--${HE_TASK}${ADAPTER_SUFFIX}" 2>/dev/null | head -1)
  echo "$found"
}

echo "[$(date -u +%FT%TZ)] s$S ($SETTING_NAME) | tasks=$NT | EVAL_MODES='$EVAL_MODES'"
echo "[$(date -u +%FT%TZ)] TRAIN_FLAGS=$TRAIN_FLAGS --delta-bins 10 --gemma4-lora"

# Subset overrides for parallel per-combo eval jobs:
#   EVAL_ONLY=1            skip training (adapter must already exist)
#   ONLY_MODE=--self-typicality   eval just this mode (default: all EVAL_MODES)
#   ONLY_BT="1" or "0"     eval just base-typ (1) or no-base (0) (default: "1 0")
[ -n "${ONLY_MODE:-}" ] && EVAL_MODES="$ONLY_MODE"
BTS="${ONLY_BT:-1 0}"

# ---- STEP 1: TRAIN (skip if final-epoch adapter present; or EVAL_ONLY) ----
adfinal=$(adapter_dir_for $((NUM_EPOCHS-1)))
if [ -n "${EVAL_ONLY:-}" ]; then
  [ -d "$adfinal" ] || { echo "[$(date -u +%FT%TZ)] FATAL: EVAL_ONLY but no final-epoch adapter: glob ${ADAPTER_SUFFIX}"; exit 1; }
  echo "[$(date -u +%FT%TZ)] EVAL_ONLY — adapter: $adfinal"
elif [ -d "$adfinal" ]; then
  echo "[$(date -u +%FT%TZ)] final-epoch adapter present, skip train: $adfinal"
else
  echo "[$(date -u +%FT%TZ)] TRAINING (log: $LOGDIR/train_g4_s${S}.log)"
  # shellcheck disable=SC2086
  python ranking_loss_ref_fix_genctx.py $TRAIN_FLAGS --delta-bins 10 --gemma4-lora \
      --wandb_run_name "g4-${WTAG}-s${S}${SMOKE:+-smoke}" > "$LOGDIR/train_g4_${WTAG}_s${S}${SMOKE:+_smoke}.log" 2>&1
  echo "[$(date -u +%FT%TZ)] train exit=$?"
  adfinal=$(adapter_dir_for $((NUM_EPOCHS-1)))
  [ -d "$adfinal" ] || { echo "[$(date -u +%FT%TZ)] FATAL: final-epoch adapter missing after train. See $LOGDIR/train_g4_s${S}.log"; tail -30 "$LOGDIR/train_g4_s${S}${SMOKE:+_smoke}.log"; exit 1; }
  echo "[$(date -u +%FT%TZ)] TRAIN DONE: $adfinal"
fi

# ---- STEP 2: EVAL — each mode x {bt1 base, bt0 no-base} x all tasks ----
# NO_EVAL=1 -> train-only (eval is launched separately as dependent parallel per-combo jobs).
if [ -n "${NO_EVAL:-}" ]; then echo "[$(date -u +%FT%TZ)] NO_EVAL=1 — training done; eval handled by dependent parallel jobs"; exit 0; fi
for EP in $EVAL_EPOCHS; do
  ad=$(adapter_dir_for "$EP"); [ -d "$ad" ] || { echo "[$(date -u +%FT%TZ)] WARN: epoch$EP adapter missing, skip"; continue; }
  for MODE in $EVAL_MODES; do
    mname="${MODE#--}"
    for BT in $BTS; do
      if [ "$BT" = 1 ]; then BT_ARGS="--base-typicality --base-model $BASE"; else BT_ARGS=""; fi
      EVAL_LOG="$LOGDIR/eval_g4_s${S}_ep${EP}_${mname}_bt${BT}.log"
      echo "[$(date -u +%FT%TZ)] ==== EVAL s$S ep$EP $MODE bt$BT (base=${BT}) ===="
      n=0; ok=0
      for T in $TASKS; do
        marker="${DONE_DIR}/s${S}_ep${EP}_${mname}_bt${BT}_${T}.done"
        if [ -f "$marker" ]; then n=$((n+1)); ok=$((ok+1)); continue; fi
        # shellcheck disable=SC2086
        if python eval_by_claude.py --model "$ad" --task "$T" --split_type random \
              --gen-shots zero --disc-shots zero "$MODE" $BT_ARGS \
              --validator-log-odds --save-scores-csv --outputs-dir "$OUTDIR" >> "$EVAL_LOG" 2>&1; then
          touch "$marker"; ok=$((ok+1))
        else
          echo "[$(date -u +%FT%TZ)]   FAIL task $T (see $EVAL_LOG)"
        fi
        n=$((n+1))
        [ $((n % 10)) -eq 0 ] && echo "[$(date -u +%FT%TZ)]   s$S $MODE bt$BT ep$EP: $n/$NT ($ok ok)"
      done
      echo "[$(date -u +%FT%TZ)] EVAL s$S ep$EP $MODE bt$BT DONE ($ok/$NT ok)"
    done
  done
done
echo "[$(date -u +%FT%TZ)] ======== SETTING ${S} COMPLETE ========"
