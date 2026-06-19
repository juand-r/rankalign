#!/bin/bash
#SBATCH --job-name=hetrpp
#SBATCH --partition=allnodes
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=9
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/datastor2/jdr/logs/%j.out
#SBATCH --error=/datastor2/jdr/logs/%j.err
#
# PER-PROBLEM TRAIN-set eval (the CORRECT version; replaces the global-pool GROUP B/trs2).
# For ONE (model, setting, dataset): eval base+epoch{0,1,2} on each of the 80 TRAIN problems'
# OWN candidates (~29 each, all of them, ~15 pos/14 neg) via the per-problem TRAIN tasks
# humaneval-v2.1correct-{ds}-train-<slug>. NO --train flag (those tasks already put the
# per-problem train candidates in L_test). Same eval flags/modes as the test eval. Then ROC is
# averaged over the 80 train problems (downstream), like-for-like with the 82 test problems.
#
# MODEL ORIGINS (docs/he_trainset_eval_model_origins.md): gemma4 s2/s4 -> models2-rerun-wandb;
# gemma4 s3/s7 -> gemma-4-models-mll-tmp[-multi]; qwen all -> models2-rerun-wandb (_merged).
#
# Usage (one checkpoint per job; override --gres=gpu:2 for qwen):
#   MODEL=gemma4 CKPTS=2 HE_TASK=humaneval-v2.1correct-upper sbatch --job-name=trpp-cu-g4-s4-ep2 mll_he_trainset_perproblem.sbatch 4
set -uo pipefail
S="${1:?usage: mll_he_trainset_perproblem.sbatch <2|3|4|7>}"
MODEL="${MODEL:?set MODEL=gemma4|qwen}"

REPO=/datastor2/jdr/rankalign
HE_TASK="${HE_TASK:-humaneval-v2.1correct-upper}"
DS=$(case "$HE_TASK" in *correct-upper) echo upper;; *correct-multi) echo multi;; esac)
WTAG=$(case "$DS" in upper) echo cu;; multi) echo cm;; esac)
OUTDIR="${OUTDIR:-/datastor2/jdr/rankalign/outputs-he-trainset-perproblem}"
LOGDIR=/datastor2/jdr/logs
CKPTS="${CKPTS:-0 1 2}"
DONE_DIR="$OUTDIR/.done"
TRAIN_PP_DIR="$REPO/data/humaneval/v2.1correct-${DS}-train-perproblem"

case "$MODEL" in
  gemma4)
    VENV=/datastor2/jdr/venvs/gemma4; BASE="google/gemma-4-31B-it"; MERGED=""; DIRPREFIX="v7-google--gemma-4-31B-it-delta"
    case "$DS" in upper) MODELS_DIR=$REPO/gemma-4-models-mll-tmp;; multi) MODELS_DIR=$REPO/gemma-4-models-mll-tmp-multi;; esac
    # s2/s4 live in models2-rerun-wandb (moved); s3/s7 in the tmp dirs
    case "$S" in 2|4) MODELS_DIR=$REPO/models2-rerun-wandb;; esac ;;
  qwen)
    VENV=/datastor2/jdr/venvs/qwen35; BASE="Qwen/Qwen3.5-9B"; MERGED="_merged"; DIRPREFIX="v7-Qwen--Qwen3.5-9B-delta"
    MODELS_DIR=$REPO/models2-rerun-wandb ;;
  *) echo "FATAL MODEL"; exit 2 ;;
esac

case "$S" in
  2) NAME=RankAlign;     MODES="--self-typicality --neg-typicality"; SUFFIX="-all--d2g--random--alpha1.0--full-completion--semi0.1--fix1" ;;
  3) NAME="New+fsx";     MODES="--self-typicality --neg-typicality"; SUFFIX="-all--d2g--random--alpha1.0--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1" ;;
  4) NAME="New+fsx+selfTC"; MODES="--self-typicality"; SUFFIX="-all--d2g--random--alpha1.0--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1" ;;
  7) NAME="New+fsx+negTC";  MODES="--neg-typicality";  SUFFIX="-all--d2g--random--alpha1.0--tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1" ;;
  *) echo "FATAL S"; exit 2 ;;
esac

export HF_HOME=/datastor2/jdr/.cache/huggingface HF_HUB_CACHE=/datastor2/jdr/.cache/huggingface/hub
export TRANSFORMERS_CACHE=$HF_HUB_CACHE HF_HUB_DISABLE_XET=1
export HF_TOKEN=$(cat /datastor2/jdr/.hf_token 2>/dev/null || cat /datastor2/jdr/.hftoken 2>/dev/null || true)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
source "$VENV/bin/activate"
cd "$REPO/scripts"
mkdir -p "$OUTDIR" "$DONE_DIR" "$LOGDIR"
TAG="trpp_${MODEL}"

ckpt_path() { local c="$1"; if [ "$c" = base ]; then echo "$BASE"; return; fi
  ls -d "${MODELS_DIR}/${DIRPREFIX}"*"-epoch${c}--${HE_TASK}${SUFFIX}${MERGED}" 2>/dev/null | head -1; }

# the 80 per-problem TRAIN tasks for this dataset
SLUGS=$(ls "$TRAIN_PP_DIR"/humaneval_*.csv | xargs -n1 basename | sed 's/\.csv$//')
[ -n "${MAXPROB:-}" ] && SLUGS=$(echo $SLUGS | tr ' ' '\n' | head -"$MAXPROB" | tr '\n' ' ')   # canary: limit #problems
NT=$(echo $SLUGS | wc -w)
echo "[$(date -u +%FT%TZ)] PER-PROBLEM TRAIN eval: $MODEL s$S ($NAME) $HE_TASK | ckpts='$CKPTS' modes='$MODES' train-problems=$NT"
echo "[$(date -u +%FT%TZ)] models<-$MODELS_DIR | scores->$OUTDIR | node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-?}"

for C in $CKPTS; do
  MP=$(ckpt_path "$C")
  if [ -z "$MP" ] || { [ "$C" != base ] && [ ! -d "$MP" ]; }; then echo "[$(date -u +%FT%TZ)] WARN ckpt $C not found -> skip"; continue; fi
  cname=$([ "$C" = base ] && echo base || echo "ep${C}")
  for MODE in $MODES; do
    mname="${MODE#--}"
    for BT in 1 0; do
      [ "$BT" = 1 ] && BT_ARGS="--base-typicality --base-model $BASE" || BT_ARGS=""
      LOG="$LOGDIR/${TAG}_${WTAG}_s${S}_${cname}_${mname}_bt${BT}.log"
      n=0; ok=0
      for slug in $SLUGS; do
        T="${HE_TASK}-train-${slug}"
        marker="${DONE_DIR}/${TAG}_${WTAG}_s${S}_${cname}_${mname}_bt${BT}_${slug}.done"
        [ -f "$marker" ] && { n=$((n+1)); ok=$((ok+1)); continue; }
        # shellcheck disable=SC2086
        if python eval_by_claude.py --model "$MP" --task "$T" --split_type random \
              --gen-shots zero --disc-shots zero "$MODE" $BT_ARGS \
              --validator-log-odds --save-scores-csv --outputs-dir "$OUTDIR" >> "$LOG" 2>&1; then
          touch "$marker"; ok=$((ok+1))
        else echo "[$(date -u +%FT%TZ)]   FAIL $cname $mname bt$BT $slug (see $LOG)"; fi
        n=$((n+1)); [ $((n % 20)) -eq 0 ] && echo "[$(date -u +%FT%TZ)]   $cname $mname bt$BT: $n/$NT ($ok ok)"
      done
      echo "[$(date -u +%FT%TZ)] DONE $cname $mname bt$BT ($ok/$NT ok)"
    done
  done
done
echo "[$(date -u +%FT%TZ)] ======== PER-PROBLEM TRAIN $MODEL ${WTAG} s${S} COMPLETE ========"
