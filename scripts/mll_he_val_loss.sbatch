#!/bin/bash
#SBATCH --job-name=hevalloss
#SBATCH --partition=allnodes
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=9
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/datastor2/jdr/logs/%j.out
#SBATCH --error=/datastor2/jdr/logs/%j.err
#
# HELD-OUT (test-set) LOSS for HumanEval — the §2.3 "Held-out Validation Loss" mirror.
# For ONE (model, setting, dataset): recompute the 3 training loss components
# (preference, NLL-G, NLL-V) on the held-out 82-problem TEST set for base + epoch{0,1,2}.
# This is the HumanEval analogue of the IFEval/membership compute_val_loss.py path.
#
# One job loops base+ep0+ep1+ep2 internally (compute_val_loss_he.py), writing one row per
# checkpoint INCREMENTALLY to a PER-JOB CSV (resumable; never shares a CSV across jobs).
#
# Checkpoint dirs are located by the proven ckpt_path() glob (same as the trpp eval), so we
# do NOT depend on the stale scripts_settings suffix map. gemma-4 ckpts are LoRA adapters
# (compute_val_loss_he.py loads base+PeftModel); qwen ckpts are merged (loaded directly).
#
# Usage (the launcher sets everything):
#   MODEL=gemma4 HE_TASK=humaneval-v2.1correct-upper sbatch --job-name=vl-cu-g4-s4 scripts/mll_he_val_loss.sbatch 4
#   MODEL=qwen   HE_TASK=humaneval-v2.1correct-multi sbatch --gres=gpu:1 --job-name=vl-cm-q-s7 scripts/mll_he_val_loss.sbatch 7
# Canary (cap items per class, fast):
#   MAXITEMS=15 OUTSUFFIX=_canary MODEL=gemma4 HE_TASK=humaneval-v2.1correct-upper sbatch scripts/mll_he_val_loss.sbatch 4
set -uo pipefail
S="${1:?usage: mll_he_val_loss.sbatch <1|2|3|4|7|13>}"
MODEL="${MODEL:?set MODEL=gemma4|qwen}"

REPO=/datastor2/jdr/rankalign
HE_TASK="${HE_TASK:-humaneval-v2.1correct-upper}"
DS=$(case "$HE_TASK" in *correct-upper) echo upper;; *correct-multi) echo multi;; esac)
WTAG=$(case "$DS" in upper) echo cu;; multi) echo cm;; esac)
TEST_TASK="humaneval-${DS}-all"
OUTDIR="${OUTDIR:-$REPO/outputs-he-valloss}"
LOGDIR=/datastor2/jdr/logs
OUTPUT="$OUTDIR/he_val_loss_${MODEL}_${WTAG}_s${S}${OUTSUFFIX:-}.csv"

case "$MODEL" in
  gemma4)
    VENV=/datastor2/jdr/venvs/gemma4; BASE="google/gemma-4-31B-it"; MERGED=""; DIRPREFIX="v7-google--gemma-4-31B-it-delta"
    # s2/s4 live in models2-rerun-wandb (moved); s1/s3/s7/s13 in the per-dataset tmp dirs
    case "$DS" in upper) MODELS_DIR=$REPO/gemma-4-models-mll-tmp;; multi) MODELS_DIR=$REPO/gemma-4-models-mll-tmp-multi;; esac
    case "$S" in 2|4) MODELS_DIR=$REPO/models2-rerun-wandb;; esac ;;
  qwen)
    VENV=/datastor2/jdr/venvs/qwen35; BASE="Qwen/Qwen3.5-9B"; MERGED="_merged"; DIRPREFIX="v7-Qwen--Qwen3.5-9B-delta"
    MODELS_DIR=$REPO/models2-rerun-wandb ;;
  *) echo "FATAL MODEL"; exit 2 ;;
esac

# Exact per-setting checkpoint suffixes (authoritative; verified against the live dirs).
# NB: scripts_settings.get_setting_suffix is STALE for HE (omits vallogodds from s1) — do not use it.
case "$S" in
  1)  NAME=SFT;            SUFFIX="-all--d2g--random--alpha1.0--full-completion--pref0.0--nllv1.0--nllg1.0--vallogodds--labelonly0.1--fix1" ;;
  13) NAME=ConsistencyFT;  SUFFIX="-all--d2g--random--alpha1.0--full-completion--pref0.0--nllv1.0--nllg1.0--cft--labelonly0.1--fix1" ;;
  2)  NAME=RankAlign;      SUFFIX="-all--d2g--random--alpha1.0--full-completion--semi0.1--fix1" ;;
  3)  NAME="New+fsx";      SUFFIX="-all--d2g--random--alpha1.0--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1" ;;
  4)  NAME="New+fsx+selfTC"; SUFFIX="-all--d2g--random--alpha1.0--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1" ;;
  7)  NAME="New+fsx+negTC";  SUFFIX="-all--d2g--random--alpha1.0--tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1" ;;
  *) echo "FATAL S (must be 1,2,3,4,7,13)"; exit 2 ;;
esac

export HF_HOME=/datastor2/jdr/.cache/huggingface HF_HUB_CACHE=/datastor2/jdr/.cache/huggingface/hub
export TRANSFORMERS_CACHE=$HF_HUB_CACHE HF_HUB_DISABLE_XET=1
export HF_TOKEN=$(cat /datastor2/jdr/.hf_token 2>/dev/null || cat /datastor2/jdr/.hftoken 2>/dev/null || true)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
source "$VENV/bin/activate"
cd "$REPO"
mkdir -p "$OUTDIR" "$LOGDIR"

# Locate epoch checkpoint dir for a given epoch (proven glob; same as the trpp eval).
ckpt_path() { local c="$1"; ls -d "${MODELS_DIR}/${DIRPREFIX}"*"-epoch${c}--${HE_TASK}${SUFFIX}${MERGED}" 2>/dev/null | head -1; }
EP0=$(ckpt_path 0); EP1=$(ckpt_path 1); EP2=$(ckpt_path 2)

echo "[$(date -u +%FT%TZ)] HELD-OUT VAL LOSS: $MODEL s$S ($NAME) $HE_TASK | test-task=$TEST_TASK"
echo "[$(date -u +%FT%TZ)] BASE=$BASE  MODELS_DIR=$MODELS_DIR  node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-?}"
echo "[$(date -u +%FT%TZ)] ep0=$EP0"
echo "[$(date -u +%FT%TZ)] ep1=$EP1"
echo "[$(date -u +%FT%TZ)] ep2=$EP2"
echo "[$(date -u +%FT%TZ)] OUTPUT=$OUTPUT"
for e in "$EP0" "$EP1" "$EP2"; do
  if [ -z "$e" ] || [ ! -d "$e" ]; then echo "[$(date -u +%FT%TZ)] FATAL: missing epoch checkpoint dir"; exit 3; fi
done

EP_ARGS=()
[ -n "$EP0" ] && EP_ARGS+=(--ep0-path "$EP0")
[ -n "$EP1" ] && EP_ARGS+=(--ep1-path "$EP1")
[ -n "$EP2" ] && EP_ARGS+=(--ep2-path "$EP2")
MAXITEMS_ARGS=()
[ -n "${MAXITEMS:-}" ] && MAXITEMS_ARGS+=(--max-items-per-class "$MAXITEMS")

python scripts/compute_val_loss_he.py \
  --model "$BASE" --base-model-path "$BASE" \
  --task "$HE_TASK" --test-task "$TEST_TASK" \
  --setting "s${S}" \
  "${EP_ARGS[@]}" "${MAXITEMS_ARGS[@]}" \
  --output "$OUTPUT"
RC=$?
echo "[$(date -u +%FT%TZ)] ======== VAL LOSS $MODEL ${WTAG} s${S} done (rc=$RC) ========"
exit $RC
