#!/bin/bash
#SBATCH --job-name=g4trset
#SBATCH --partition=allnodes
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=9
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=/datastor2/jdr/logs/%j.out
#SBATCH --error=/datastor2/jdr/logs/%j.err
#
# TRAIN-SET eval (a la run_trainset_dynamics_*.sh) for gemma-4-31B-it humaneval, s2/s4.
# For ONE (setting, dataset): eval base + epoch{0,1,2} checkpoints on the --train split,
# with the SAME eval modes as the test eval (s2: self+neg ; s4: self) x {base-typ, no-base},
# --validator-log-odds --disc-shots zero. Writes train scores to a dedicated dir.
# Resumable: per-(ckpt,mode,bt,task) done-markers; resubmitting skips finished work.
#
# >>>>> MODEL ORIGIN — DO NOT GUESS (see docs/he_trainset_eval_model_origins.md) <<<<<
#   gemma s2/s4 humaneval checkpoints live in  models2-rerun-wandb/  (NOT gemma-4-models-mll-tmp).
#   They were MOVED there (user-approved 2026-06-17) when consolidating finetuned gemma adapters.
#   The CURRENT-run settings s1/s3/s7/s13 instead live in gemma-4-models-mll-tmp[-multi]/ — so a
#   future s1/s3/s7/s13 train-eval must point MODELS_DIR there, NOT here.
#
# Usage:
#   HE_TASK=humaneval-v2.1correct-upper sbatch --job-name=g4trset-cu-s2 scripts/mll_he_trainset_eval.sbatch 2
#   HE_TASK=humaneval-v2.1correct-multi sbatch --job-name=g4trset-cm-s4 scripts/mll_he_trainset_eval.sbatch 4
#   SETTING in {2,4}; HE_TASK upper|multi; CKPTS env default "base 0 1 2".
# Together the 4 submits (cu-s2, cu-s4, cm-s2, cm-s4) cover the 16 checkpoint-evals
#   {upper,multi} x {s2,s4} x {base,0,1,2}.
set -uo pipefail
S="${1:?usage: mll_he_trainset_eval.sbatch <2|4>}"

REPO=/datastor2/jdr/rankalign
VENV=/datastor2/jdr/venvs/gemma4                       # gemma-4 needs transformers 5.x
HE_TASK="${HE_TASK:-humaneval-v2.1correct-upper}"
WTAG=$(case "$HE_TASK" in *correct-upper) echo cu;; *correct-multi) echo cm;; *) echo "${HE_TASK##*-}";; esac)
BASE="google/gemma-4-31B-it"
MODELS_DIR=/datastor2/jdr/rankalign/models2-rerun-wandb         # <-- s2/s4 live HERE (moved)
TRAIN_OUTDIR="${TRAIN_OUTDIR:-/datastor2/jdr/rankalign/outputs-he-trainset}"
LOGDIR=/datastor2/jdr/logs
CKPTS="${CKPTS:-base 0 1 2}"
DONE_DIR="$TRAIN_OUTDIR/.done"

export HF_HOME=/datastor2/jdr/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
export HF_TOKEN=$(cat /datastor2/jdr/.hftoken 2>/dev/null || true)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled          # eval only; no wandb
export TOKENIZERS_PARALLELISM=false

source "$VENV/bin/activate"
cd "$REPO/scripts"   # cwd on /datastor2 — eval_by_claude.py writes a cwd-relative summary log (avoid ENOSPC)
mkdir -p "$TRAIN_OUTDIR" "$DONE_DIR" "$LOGDIR"

# per-setting eval modes + the dir-suffix that identifies the checkpoint in models2-rerun-wandb
case "$S" in
  2) SETTING_NAME="RankAlign";  EVAL_MODES="--self-typicality --neg-typicality"
     SUFFIX="-all--d2g--random--alpha1.0--full-completion--semi0.1--fix1" ;;
  4) SETTING_NAME="New+fsx+tc"; EVAL_MODES="--self-typicality"
     SUFFIX="-all--d2g--random--alpha1.0--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1" ;;
  *) echo "FATAL: this launcher handles gemma s2 or s4 ONLY. s1/s3/s7/s13 live in gemma-4-models-mll-tmp -> separate launcher."; exit 2 ;;
esac

# resolve a checkpoint label -> model path ('base' = the base model; N = epoch N adapter dir)
ckpt_path() {
  local c="$1"
  if [ "$c" = base ]; then echo "$BASE"; return; fi
  ls -d "${MODELS_DIR}/v7-google--gemma-4-31B-it-delta"*"-epoch${c}--${HE_TASK}${SUFFIX}" 2>/dev/null | head -1
}

TASKS=$(ls "$REPO/data/humaneval/${HE_TASK#humaneval-}/humaneval_"*.csv | xargs -n1 basename | sed 's/\.csv$//' | sed "s/^/${HE_TASK}-/" | tr '\n' ' ')
NT=$(echo $TASKS | wc -w)
echo "[$(date -u +%FT%TZ)] === TRAIN-SET eval g4 s$S ($SETTING_NAME) $HE_TASK | ckpts='$CKPTS' modes='$EVAL_MODES' tasks=$NT ==="
echo "[$(date -u +%FT%TZ)] models <- $MODELS_DIR (suffix $SUFFIX) | scores -> $TRAIN_OUTDIR | node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head

for C in $CKPTS; do
  MP=$(ckpt_path "$C")
  if [ -z "$MP" ] || { [ "$C" != base ] && [ ! -d "$MP" ]; }; then
    echo "[$(date -u +%FT%TZ)] WARN: checkpoint '$C' not found (glob ${HE_TASK}${SUFFIX}) -> skip"; continue
  fi
  cname=$([ "$C" = base ] && echo base || echo "ep${C}")
  echo "[$(date -u +%FT%TZ)] ==== checkpoint $cname -> $MP ===="
  for MODE in $EVAL_MODES; do
    mname="${MODE#--}"
    for BT in 1 0; do
      if [ "$BT" = 1 ]; then BT_ARGS="--base-typicality --base-model $BASE"; else BT_ARGS=""; fi
      EVAL_LOG="$LOGDIR/trset_g4_${WTAG}_s${S}_${cname}_${mname}_bt${BT}.log"
      n=0; ok=0
      for T in $TASKS; do
        marker="${DONE_DIR}/trset_g4_${WTAG}_s${S}_${cname}_${mname}_bt${BT}_${T}.done"
        [ -f "$marker" ] && { n=$((n+1)); ok=$((ok+1)); continue; }
        # shellcheck disable=SC2086
        if python eval_by_claude.py --model "$MP" --task "$T" --split_type random \
              --gen-shots zero --disc-shots zero "$MODE" $BT_ARGS \
              --validator-log-odds --train --save-scores-csv --outputs-dir "$TRAIN_OUTDIR" >> "$EVAL_LOG" 2>&1; then
          touch "$marker"; ok=$((ok+1))
        else
          echo "[$(date -u +%FT%TZ)]   FAIL $cname $mname bt$BT $T (see $EVAL_LOG)"
        fi
        n=$((n+1))
        [ $((n % 20)) -eq 0 ] && echo "[$(date -u +%FT%TZ)]   $cname $mname bt$BT: $n/$NT ($ok ok)"
      done
      echo "[$(date -u +%FT%TZ)] DONE $cname $mname bt$BT ($ok/$NT ok)"
    done
  done
done
echo "[$(date -u +%FT%TZ)] ======== TRAIN-SET g4 ${WTAG} s${S} COMPLETE ========"
