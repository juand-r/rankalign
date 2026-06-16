#!/bin/bash
#SBATCH --job-name=q35he
#SBATCH --partition=allnodes
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/datastor2/jdr/logs/%j.out
#SBATCH --error=/datastor2/jdr/logs/%j.err
#
# Qwen/Qwen3.5-9B LoRA train + inline eval on humaneval (v2.1correct-upper|multi),
# settings s2 (RankAlign) / s4 (New+fsx+self-tc / FLORA-PMI).
#
# This is a port of the proven gemma-4 humaneval launcher
#   pod-setup-train-scripts-gemma-4/mll_g4_train_eval.sbatch
# with the qwen-3.5 specifics from scripts/run_qwen35_cell_mll.sbatch:
#   - MODEL Qwen/Qwen3.5-9B, base-typicality base = same
#   - venv /datastor2/jdr/venvs/qwen35 (fla fast kernel) + moe.py patch
#   - plain --lora (NOT --gemma4-lora); training MERGES -> dir ends in "_merged"
#   - 2x A40 (9B+LoRA OOMs on 1), HF cache /datastor2/jdr/hf_cache (Qwen cached there)
#   - models -> models2-rerun-wandb ; scores -> outputs-rerun-wandb
#   - wandb ONLINE, project rankalign, run name q35-<cu|cm>-s<S>
#
# Per-setting flags match the gemma-4 humaneval run EXACTLY (user choice "option a"):
#   s2 trains WITHOUT --validator-log-odds  (suffix ...--full-completion--semi0.1--fix1)
#   s4 identical to gemma  (...--tc-self...--ppd--vallogodds--semi0.1--fix1)
# (NB: this makes qwen-humaneval-s2 differ from qwen-ifeval-s2, which used vlo. Intentional.)
#
# Usage:
#   sbatch --job-name=q35cu-s2 scripts/mll_qwen35_he_train_eval.sbatch 2
#   HE_TASK=humaneval-v2.1correct-multi sbatch --job-name=q35cm-s4 scripts/mll_qwen35_he_train_eval.sbatch 4
# Env knobs:
#   HE_TASK   humaneval-v2.1correct-upper (default) | humaneval-v2.1correct-multi
#   SMOKE=1   1 epoch / 64 samples / 2 tasks / *-smoke dirs (canary)
#   NO_EVAL=1 train only (skip eval)
#   EVAL_ONLY=1 / ONLY_MODE=--self-typicality / ONLY_BT="1 0"  (eval subsetting; eval an existing merged dir)
set -euo pipefail
S="${1:?usage: mll_qwen35_he_train_eval.sbatch <2|4>}"

REPO=/datastor2/jdr/rankalign
VENV=/datastor2/jdr/venvs/qwen35
HE_TASK="${HE_TASK:-humaneval-v2.1correct-upper}"   # override -> humaneval-v2.1correct-multi
WTAG=$(case "$HE_TASK" in *correct-upper) echo cu;; *correct-multi) echo cm;; *) echo "${HE_TASK##*-}";; esac)
MODEL="Qwen/Qwen3.5-9B"; BASE="Qwen/Qwen3.5-9B"
LOGDIR=/datastor2/jdr/logs
NUM_EPOCHS=3 ; TOTAL_SAMPLES=5110 ; EVAL_EPOCHS="${EVAL_EPOCHS:-2}"
MODELS_DIR="${MODELS_DIR:-/datastor2/jdr/rankalign/models2-rerun-wandb}"
OUTDIR="${OUTDIR:-/datastor2/jdr/rankalign/outputs-rerun-wandb}"
TASK_LIMIT=0   # 0 = all tasks

if [ -n "${SMOKE:-}" ]; then
    NUM_EPOCHS=1 ; TOTAL_SAMPLES=64 ; TASK_LIMIT=2 ; EVAL_EPOCHS=0
    MODELS_DIR="/datastor2/jdr/rankalign/models2-rerun-wandb-smoke"
    OUTDIR="/datastor2/jdr/rankalign/outputs-rerun-wandb-smoke"
fi
DONE_DIR="$OUTDIR/.done"

# ---- env (everything off $HOME; Qwen3.5-9B is cached in /datastor2/jdr/hf_cache) ----
export HF_HOME=/datastor2/jdr/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
export HF_TOKEN=$(cat /datastor2/jdr/.hftoken 2>/dev/null || true)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY=$(grep -oP 'WANDB_API_KEY=\K\S+' /u/jdr/.bashrc 2>/dev/null | head -1)
if [ -n "$WANDB_API_KEY" ]; then export WANDB_MODE=online; else echo "[WARN] no WANDB_API_KEY -> OFFLINE"; export WANDB_MODE=offline; fi
export WANDB_PROJECT=rankalign

source "$VENV/bin/activate"
cd "$REPO/scripts"
mkdir -p "$MODELS_DIR" "$OUTDIR" "$DONE_DIR" "$LOGDIR"

# ---- moe.py patch (idempotent): transformers 5.8 `from __future__ import annotations`
#      breaks torch 2.4.1 custom_op infer_schema -> `import peft` fails. (qwen35 venv) ----
MOE_FILE=$(python -c "import transformers, os; print(os.path.join(os.path.dirname(transformers.__file__), 'integrations', 'moe.py'))" 2>/dev/null || true)
if [ -n "$MOE_FILE" ] && [ -f "$MOE_FILE" ] && grep -q "^from __future__ import annotations" "$MOE_FILE"; then
    sed -i "s/^from __future__ import annotations$/# from __future__ import annotations (patched: torch 2.4.1 infer_schema compat)/" "$MOE_FILE"
    echo "[$(date -u +%FT%TZ)] moe.py patched -> $MOE_FILE"
fi

echo "[$(date -u +%FT%TZ)] === q35 s$S on $HE_TASK | epochs=$NUM_EPOCHS samples=$TOTAL_SAMPLES smoke=${SMOKE:-0} | MODELS_DIR=$MODELS_DIR OUTDIR=$OUTDIR ==="
echo "[$(date -u +%FT%TZ)] node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-?} wandb=$WANDB_MODE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head

# ---- test-task list (all humaneval_*.csv for this dataset) ----
TASKS=$(ls "$REPO/data/humaneval/${HE_TASK#humaneval-}/humaneval_"*.csv | xargs -n1 basename | sed 's/\.csv$//' | sed "s/^/${HE_TASK}-/" | tr '\n' ' ')
if [ "$TASK_LIMIT" -gt 0 ]; then TASKS=$(echo $TASKS | tr ' ' '\n' | head -n "$TASK_LIMIT" | tr '\n' ' '); fi
NT=$(echo $TASKS | wc -w)

# ---- per-setting flags (s2 matches gemma EXACTLY -> NO vlo; s4 identical to gemma) ----
pref_w=1 nll_v_w=0 nll_g_w=0; semi_flag="--semi-supervised 0.1"; fsx=""; vlo=""; tc_train=""; ppd=""; sbm=""
case "$S" in
  2) SETTING_NAME="RankAlign"
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

# qwen merges LoRA -> the saved dir ends in "_merged" (cf. run_qwen35_cell_mll.sbatch glob).
# NB: must end in `echo` (exit 0) and guard the glob `ls` with `|| true`, else a no-match
# `ls | head` returns exit 2 and `var=$(adapter_dir_for ...)` trips `set -euo pipefail`.
adapter_dir_for() {
  local found
  found=$(ls -d "${MODELS_DIR}/v7-Qwen--Qwen3.5-9B-delta"*"-epoch${1}--${HE_TASK}${ADAPTER_SUFFIX}_merged" 2>/dev/null | head -1) || true
  echo "$found"
}

echo "[$(date -u +%FT%TZ)] s$S ($SETTING_NAME) | tasks=$NT | EVAL_MODES='$EVAL_MODES'"
echo "[$(date -u +%FT%TZ)] TRAIN_FLAGS=$TRAIN_FLAGS --delta-bins 10"

# Eval subsetting (for re-running a single combo against an existing merged dir):
[ -n "${ONLY_MODE:-}" ] && EVAL_MODES="$ONLY_MODE"
BTS="${ONLY_BT:-1 0}"

# ---- STEP 1: TRAIN (skip if final-epoch merged dir present; or EVAL_ONLY) ----
adfinal=$(adapter_dir_for $((NUM_EPOCHS-1)))
if [ -n "${EVAL_ONLY:-}" ]; then
  [ -n "$adfinal" ] && [ -d "$adfinal" ] || { echo "[$(date -u +%FT%TZ)] FATAL: EVAL_ONLY but no final merged dir: glob ${ADAPTER_SUFFIX}_merged"; exit 1; }
  echo "[$(date -u +%FT%TZ)] EVAL_ONLY — merged dir: $adfinal"
elif [ -n "$adfinal" ] && [ -d "$adfinal" ]; then
  echo "[$(date -u +%FT%TZ)] final merged dir present, skip train: $adfinal"
else
  echo "[$(date -u +%FT%TZ)] TRAINING (log: $LOGDIR/train_q35_${WTAG}_s${S}${SMOKE:+_smoke}.log)"
  # shellcheck disable=SC2086
  python ranking_loss_ref_fix.py $TRAIN_FLAGS --delta-bins 10 \
      --wandb_run_name "q35-${WTAG}-s${S}${SMOKE:+-smoke}" > "$LOGDIR/train_q35_${WTAG}_s${S}${SMOKE:+_smoke}.log" 2>&1
  echo "[$(date -u +%FT%TZ)] train exit=$?"
  adfinal=$(adapter_dir_for $((NUM_EPOCHS-1)))
  [ -n "$adfinal" ] && [ -d "$adfinal" ] || { echo "[$(date -u +%FT%TZ)] FATAL: final merged dir missing after train. See $LOGDIR/train_q35_${WTAG}_s${S}${SMOKE:+_smoke}.log"; tail -30 "$LOGDIR/train_q35_${WTAG}_s${S}${SMOKE:+_smoke}.log"; exit 1; }
  echo "[$(date -u +%FT%TZ)] TRAIN DONE: $adfinal"
fi

# ---- STEP 2: EVAL — each mode x {bt1 base, bt0 no-base} x all tasks (inline; done-markers -> resumable) ----
if [ -n "${NO_EVAL:-}" ]; then echo "[$(date -u +%FT%TZ)] NO_EVAL=1 — training done; skipping eval"; exit 0; fi
for EP in $EVAL_EPOCHS; do
  ad=$(adapter_dir_for "$EP"); [ -n "$ad" ] && [ -d "$ad" ] || { echo "[$(date -u +%FT%TZ)] WARN: epoch$EP merged dir missing, skip"; continue; }
  for MODE in $EVAL_MODES; do
    mname="${MODE#--}"
    for BT in $BTS; do
      if [ "$BT" = 1 ]; then BT_ARGS="--base-typicality --base-model $BASE"; else BT_ARGS=""; fi
      EVAL_LOG="$LOGDIR/eval_q35_${WTAG}_s${S}_ep${EP}_${mname}_bt${BT}.log"
      echo "[$(date -u +%FT%TZ)] ==== EVAL s$S ep$EP $MODE bt$BT ===="
      n=0; ok=0
      for T in $TASKS; do
        marker="${DONE_DIR}/q35_${WTAG}_s${S}_ep${EP}_${mname}_bt${BT}_${T}.done"
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
echo "[$(date -u +%FT%TZ)] ======== q35 ${WTAG} SETTING ${S} COMPLETE ========"
