#!/bin/bash
#SBATCH --job-name=g4cu
#SBATCH --partition=allnodes
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=9
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/datastor2/jdr/logs/%j.out
#SBATCH --error=/datastor2/jdr/logs/%j.err
#
# Train + eval ONE gemma-4-31B-it setting (s2 RankAlign or s4 FLORA-PMI/New+fsx+tc) on
# humaneval-v2.1correct-upper, on mll. Uses the canonical bug-fixed trainer
# ranking_loss_ref_fix.py with --gemma4-lora --lora --delta-bins 10 (flags verbatim from the
# verified pod recipe run_settings_v21correct_upper.sh). WANDB online (the point of this run).
#
# Eval: the epoch-2 adapter, for EACH eval mode x {bt1 = base-typicality, bt0 = no base}:
#   s2 (RankAlign):    {self,neg} x {bt1,bt0} = 4 score-sets
#   s4 (New+fsx+tc):   {self}     x {bt1,bt0} = 2 score-sets
# scores_ files for ALL 82 humaneval-v2.1correct-upper test tasks per score-set.
#
# allnodes / 9 CPU / 96G / 4xA40 (raca defaults; gpu:4 gives eval-bt1 headroom — it loads the
# trained model AND a frozen base ~= 124G). NOTHING in $HOME: HF cache + models + outputs on
# /datastor2. Idempotent: per-(epoch,mode,bt,task) done-markers; resubmitting resumes.
#
# Usage:  sbatch mll_g4_train_eval.sbatch <2|4>
# Smoke:  SMOKE=1 sbatch mll_g4_train_eval.sbatch <2|4>   (1 epoch, 64 samples, 2 tasks, -smoke dirs)
set -uo pipefail
S="${1:?usage: mll_g4_train_eval.sbatch <2|4>}"

REPO=/datastor1/jdr/gv-gap/rankalign
VENV=/datastor2/jdr/venvs/gemma4
HE_TASK="humaneval-v2.1correct-upper"
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
TASKS=$(ls "$REPO/data/humaneval/v2.1correct-upper/humaneval_"*.csv | xargs -n1 basename | sed 's/\.csv$//' | sed "s/^/${HE_TASK}-/" | tr '\n' ' ')
if [ "$TASK_LIMIT" -gt 0 ]; then TASKS=$(echo $TASKS | tr ' ' '\n' | head -n "$TASK_LIMIT" | tr '\n' ' '); fi
NT=$(echo $TASKS | wc -w)

# ---- per-setting flags (verbatim from run_settings_v21correct_upper.sh configure_setting) ----
pref_w=1 nll_v_w=0 nll_g_w=0; semi_flag="--semi-supervised 0.1"; fsx=""; vlo=""; tc_train=""; ppd=""; sbm=""
case "$S" in
  2) SETTING_NAME="RankAlign"
     # NB: ranking_loss_ref_fix.py appends --fix1 (the pod recipe's s2 suffix was the OLD v6
     # name because s2 was never retrained with fix.py there — confirmed via smoke 2026-06-13).
     ADAPTER_SUFFIX="-all--d2g--random--alpha1.0--full-completion--semi0.1--fix1"
     EVAL_MODES="--self-typicality --neg-typicality" ;;
  4) SETTING_NAME="New+fsx+tc"
     nll_v_w=1; nll_g_w=1; fsx="--force-same-x"; vlo="--validator-log-odds"; tc_train="--self-typicality"
     ADAPTER_SUFFIX="-all--d2g--random--alpha1.0--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1"
     EVAL_MODES="--self-typicality" ;;
  *) echo "FATAL: this launcher handles 2 or 4 only"; exit 2 ;;
esac
TRAIN_FLAGS="--model $MODEL --num_epochs $NUM_EPOCHS --task $HE_TASK --train_g_or_d g --split_type random --nll_validator_weight $nll_v_w --nll_generator_weight $nll_g_w --preference_loss_weight $pref_w --all --delta 0.15 $semi_flag --disc-shots zero --lora --gradient_checkpointing --models-dir $MODELS_DIR --total_samples $TOTAL_SAMPLES --no-upload-hf"
if [ -n "$fsx" ]; then ppd="--per-prompt-delta"; sbm="--shape-budget-mode global"; fi
for f in "$fsx" "$ppd" "$sbm" "$vlo" "$tc_train"; do [ -n "$f" ] && TRAIN_FLAGS="$TRAIN_FLAGS $f"; done

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
  python ranking_loss_ref_fix.py $TRAIN_FLAGS --delta-bins 10 --gemma4-lora \
      --wandb_run_name "g4-cu-s${S}${SMOKE:+-smoke}" > "$LOGDIR/train_g4_s${S}${SMOKE:+_smoke}.log" 2>&1
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
