#!/bin/bash
#SBATCH --job-name=hetrset2
#SBATCH --partition=allnodes
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=9
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/datastor2/jdr/logs/%j.out
#SBATCH --error=/datastor2/jdr/logs/%j.err
#
# GENERALIZED train-set eval (extends mll_he_trainset_eval.sbatch beyond gemma s2/s4).
# Handles: MODEL in {gemma4, qwen}; SETTING S in {2,3,4,7}. One (model,setting,dataset),
# eval base+epoch{0,1,2} on the --train split, N=50 stratified, --validator-log-odds
# --disc-shots zero, with the per-setting eval modes used at test. Per-(ckpt,mode,bt,task)
# done-markers => resumable. Scores -> outputs-he-trainset/ (same dir as GROUP B).
#
# MODEL ORIGINS (see docs/he_trainset_eval_model_origins.md, exact names verified 2026-06-19):
#   gemma4 s3/s7 -> gemma-4-models-mll-tmp/ (upper), gemma-4-models-mll-tmp-multi/ (multi)  [LoRA adapters]
#   qwen   s2/s3/s4/s7 -> models2-rerun-wandb/  [merged full models, _merged suffix]
#   For qwen base, run a base job with S=2 (gets self+neg modes covering all settings).
#
# Usage (submit one checkpoint per job for backfill; override --gres for qwen):
#   MODEL=gemma4 CKPTS=0 HE_TASK=humaneval-v2.1correct-upper sbatch --job-name=trs2-cu-g4-s3-ep0 mll_he_trainset_eval_v2.sbatch 3
#   MODEL=qwen   CKPTS=0 HE_TASK=humaneval-v2.1correct-multi sbatch --gres=gpu:2 --job-name=trs2-cm-qw-s4-ep0 mll_he_trainset_eval_v2.sbatch 4
set -uo pipefail
S="${1:?usage: mll_he_trainset_eval_v2.sbatch <2|3|4|7>}"
MODEL="${MODEL:?set MODEL=gemma4|qwen}"

REPO=/datastor2/jdr/rankalign
HE_TASK="${HE_TASK:-humaneval-v2.1correct-upper}"
WTAG=$(case "$HE_TASK" in *correct-upper) echo cu;; *correct-multi) echo cm;; *) echo "${HE_TASK##*-}";; esac)
TRAIN_OUTDIR="${TRAIN_OUTDIR:-/datastor2/jdr/rankalign/outputs-he-trainset}"
LOGDIR=/datastor2/jdr/logs
CKPTS="${CKPTS:-0 1 2}"               # base handled by a dedicated S=2 base job (see launcher)
MAXTRAIN="${MAXTRAIN:-50}"
DONE_DIR="$TRAIN_OUTDIR/.done"

# ---- model family ----------------------------------------------------------
case "$MODEL" in
  gemma4)
    VENV=/datastor2/jdr/venvs/gemma4
    BASE="google/gemma-4-31B-it"
    MERGED=""                          # adapters, no _merged suffix
    DIRPREFIX="v7-google--gemma-4-31B-it-delta"
    case "$HE_TASK" in
      *correct-upper) MODELS_DIR=/datastor2/jdr/rankalign/gemma-4-models-mll-tmp ;;
      *correct-multi) MODELS_DIR=/datastor2/jdr/rankalign/gemma-4-models-mll-tmp-multi ;;
    esac ;;
  qwen)
    VENV=/datastor2/jdr/venvs/qwen35
    BASE="Qwen/Qwen3.5-9B"
    MERGED="_merged"
    DIRPREFIX="v7-Qwen--Qwen3.5-9B-delta"
    MODELS_DIR=/datastor2/jdr/rankalign/models2-rerun-wandb ;;
  *) echo "FATAL: MODEL must be gemma4 or qwen"; exit 2 ;;
esac

# ---- setting -> eval modes + checkpoint dir suffix (tail after epoch{C}--{HE_TASK}) ----
case "$S" in
  2) SETTING_NAME="RankAlign";    EVAL_MODES="--self-typicality --neg-typicality"
     SUFFIX="-all--d2g--random--alpha1.0--full-completion--semi0.1--fix1" ;;
  3) SETTING_NAME="New+fsx[-TC]"; EVAL_MODES="--self-typicality --neg-typicality"
     SUFFIX="-all--d2g--random--alpha1.0--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1" ;;
  4) SETTING_NAME="New+fsx+selfTC"; EVAL_MODES="--self-typicality"
     SUFFIX="-all--d2g--random--alpha1.0--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1" ;;
  7) SETTING_NAME="New+fsx+negTC"; EVAL_MODES="--neg-typicality"
     SUFFIX="-all--d2g--random--alpha1.0--tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1" ;;
  *) echo "FATAL: S must be 2,3,4,7"; exit 2 ;;
esac

export HF_HOME=/datastor2/jdr/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
export HF_TOKEN=$(cat /datastor2/jdr/.hf_token 2>/dev/null || cat /datastor2/jdr/.hftoken 2>/dev/null || true)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

source "$VENV/bin/activate"
cd "$REPO/scripts"
mkdir -p "$TRAIN_OUTDIR" "$DONE_DIR" "$LOGDIR"

# tag for done-markers/logs that won't collide with GROUP B (which used trset_g4_*)
TAG="trs2_${MODEL}"

ckpt_path() {
  local c="$1"
  if [ "$c" = base ]; then echo "$BASE"; return; fi
  ls -d "${MODELS_DIR}/${DIRPREFIX}"*"-epoch${c}--${HE_TASK}${SUFFIX}${MERGED}" 2>/dev/null | head -1
}

TASKS=$(ls "$REPO/data/humaneval/${HE_TASK#humaneval-}/humaneval_"*.csv | xargs -n1 basename | sed 's/\.csv$//' | sed "s/^/${HE_TASK}-/" | tr '\n' ' ')
NT=$(echo $TASKS | wc -w)
echo "[$(date -u +%FT%TZ)] === TRAIN-SET v2 $MODEL s$S ($SETTING_NAME) $HE_TASK | ckpts='$CKPTS' modes='$EVAL_MODES' tasks=$NT ==="
echo "[$(date -u +%FT%TZ)] models <- $MODELS_DIR (suffix ${SUFFIX}${MERGED}) | base=$BASE | scores -> $TRAIN_OUTDIR | node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-?}"

for C in $CKPTS; do
  MP=$(ckpt_path "$C")
  if [ -z "$MP" ] || { [ "$C" != base ] && [ ! -d "$MP" ]; }; then
    echo "[$(date -u +%FT%TZ)] WARN: checkpoint '$C' not found (glob ${DIRPREFIX}*-epoch${C}--${HE_TASK}${SUFFIX}${MERGED}) -> skip"; continue
  fi
  cname=$([ "$C" = base ] && echo base || echo "ep${C}")
  echo "[$(date -u +%FT%TZ)] ==== checkpoint $cname -> $MP ===="
  for MODE in $EVAL_MODES; do
    mname="${MODE#--}"
    for BT in 1 0; do
      if [ "$BT" = 1 ]; then BT_ARGS="--base-typicality --base-model $BASE"; else BT_ARGS=""; fi
      EVAL_LOG="$LOGDIR/${TAG}_${WTAG}_s${S}_${cname}_${mname}_bt${BT}.log"
      n=0; ok=0
      for T in $TASKS; do
        marker="${DONE_DIR}/${TAG}_${WTAG}_s${S}_${cname}_${mname}_bt${BT}_${T}.done"
        [ -f "$marker" ] && { n=$((n+1)); ok=$((ok+1)); continue; }
        # shellcheck disable=SC2086
        if python eval_by_claude.py --model "$MP" --task "$T" --split_type random \
              --gen-shots zero --disc-shots zero "$MODE" $BT_ARGS \
              --validator-log-odds --train --max-train "$MAXTRAIN" --stratified \
              --save-scores-csv --outputs-dir "$TRAIN_OUTDIR" >> "$EVAL_LOG" 2>&1; then
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
echo "[$(date -u +%FT%TZ)] ======== TRAIN-SET v2 $MODEL ${WTAG} s${S} COMPLETE ========"
