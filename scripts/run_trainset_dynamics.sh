#!/bin/bash
# Train-set dynamics eval.
#
# Evaluate the BASE model + each saved epoch checkpoint (ep0/ep1/ep2) on the
# TRAINING task itself (membership-sans-rosch-v0, --train split) to track how
# the generator / validator scores evolve over training. This is distinct from
# the usual eval, which scores the held-out rosch-* tasks.
#
# IMPORTANT mechanics (verified 2026-06-07):
#   * membership-sans-rosch-v0 load_data returns (sampled_train, []), i.e. the
#     whole 2000-item training set sits in L_train and L_test is EMPTY. So the
#     --train flag is REQUIRED here (without it the eval set is empty).
#   * The model dir token "...-v0-all..." comes from the trainer's --all flag
#     (use all examples), NOT a different task. Training used task
#     membership-sans-rosch-v0 (confirmed via training_run_logs JSON).
#   * The base-model point does not depend on SETTING (it is the untrained model).
#
# Usage:
#   bash scripts/run_trainset_dynamics.sh [MODEL] [SETTING]
#     MODEL    bare HF id or google/... (default gemma-2-9b-it)
#     SETTING  s1..s7, s11..s13 (default s4 = comb+fsx+tc-self)
#
# Env:
#   MODELS_DIR    default /datastor2/jdr/rankalign/models2
#   OUTPUTS_DIR   default /datastor2/jdr/rankalign/outputs-trainset-dynamics
#   EVAL_HOURS    default 6
#   EPOCHS        default "0 1 2"
#   NO_BASE=1     skip the base-model eval (e.g. when adding more settings, the
#                 base point is identical and only needs to be run once)
#   DRYRUN=1      print the sbatch commands but do not submit

set -euo pipefail

MODEL_NAME="${1:-gemma-2-9b-it}"
SETTING="${2:-s4}"
case "$MODEL_NAME" in
    google/*) MODEL="$MODEL_NAME" ;;
    *)        MODEL="google/$MODEL_NAME" ;;
esac

TASK="membership-sans-rosch-v0"
MODELS_DIR="${MODELS_DIR:-/datastor2/jdr/rankalign/models2}"
OUTPUTS_DIR="${OUTPUTS_DIR:-/datastor2/jdr/rankalign/outputs-trainset-dynamics}"
EVAL_HOURS="${EVAL_HOURS:-6}"
EVAL_MEM=48G
EPOCHS="${EPOCHS:-0 1 2}"

mkdir -p "$OUTPUTS_DIR"

# --- setting -> (model-dir suffix tokens, TC eval variant) ----------------
# Suffix mirrors the save-dir built by ranking_loss_ref_fix.py / _eval_only.sh.
# TC variant selects the eval typicality flags (matches _overnight_launch.sh
# build_eval_flags): "self" -> --self-typcorr --base-typcorr, "neg" -> --neg.
case "$SETTING" in
    s2)  SUFFIX="--full-completion--semi0.1--fix1"; TC="self" ;;
    s3)  SUFFIX="--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1"; TC="self" ;;
    s4)  SUFFIX="--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1"; TC="self" ;;
    s5)  SUFFIX="--tc-self--full-completion--force-same-x--ppd--semi0.1--fix1"; TC="self" ;;
    s6)  SUFFIX="--tc-self--full-completion--semi0.1--fix1"; TC="self" ;;
    s7)  SUFFIX="--tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1"; TC="neg" ;;
    s11) SUFFIX="--tc-self--full-completion--nllv1.0--nllg1.0--vallogodds--semi0.1--fix1"; TC="self" ;;
    s12) SUFFIX="--tc-neg--full-completion--nllv1.0--nllg1.0--vallogodds--semi0.1--fix1"; TC="neg" ;;
    s1)  SUFFIX="--full-completion--pref0.0--nllv1.0--nllg1.0--labelonly0.1--fix1"; TC="self" ;;
    s13) SUFFIX="--full-completion--pref0.0--nllv1.0--nllg1.0--cft--labelonly0.1--fix1"; TC="self" ;;
    *)   echo "FATAL: SETTING $SETTING not wired yet (add its suffix/TC mapping)"; exit 1 ;;
esac

# Allow env override of TC (for settings needing both self+neg evals).
TC="${TC_OVERRIDE:-$TC}"

# TC eval flag stack (the "usual" flags: typicality correction + validator
# log-odds, NO length-normalization). run_eval_semi.sh adds --gen-shots zero,
# --disc-shots few (default), and --save-scores-csv on top.
if [ "$TC" = "self" ]; then
    EVAL_FLAGS="--self-typcorr --base-typcorr --base-model $MODEL --log-odds"
else
    EVAL_FLAGS="--neg-typcorr --base-typcorr --base-model $MODEL --log-odds"
fi
EVAL_FLAGS="$EVAL_FLAGS --train --outputs-dir $OUTPUTS_DIR"

# gemma-2-2b* is full-FT (no LoRA merge); everything else evals the _merged dir.
MERGED_SUFFIX="_merged"
[[ "$MODEL" == *"-2b"* ]] && MERGED_SUFFIX=""

MODEL_REPL=$(echo "$MODEL" | sed 's|/|--|g')
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OVERNIGHT_DIR="$REPO_ROOT/overnight"
mkdir -p "$OVERNIGHT_DIR"
JOB_LOG="$OVERNIGHT_DIR/_trainset_dynamics_jobids.txt"

echo "========================================"
echo "Train-set dynamics eval"
echo "  MODEL:       $MODEL"
echo "  SETTING:     $SETTING (TC=$TC)"
echo "  TASK:        $TASK  (--train split)"
echo "  EVAL_FLAGS:  $EVAL_FLAGS"
echo "  OUTPUTS_DIR: $OUTPUTS_DIR"
echo "  EPOCHS:      $EPOCHS  (+ base unless NO_BASE)"
echo "========================================"

submit() {
    local model_arg="$1" jobtag="$2"
    local wrap_cmd="cd ${REPO_ROOT} && PYTHONUNBUFFERED=1 /usr/bin/time -v \
scripts/run_eval_semi.sh \"${model_arg}\" ${EVAL_FLAGS} -- ${TASK}"

    if [ -n "${DRYRUN:-}" ]; then
        echo "DRYRUN [$jobtag]:"
        echo "  sbatch --gres=gpu:1 --time=${EVAL_HOURS}:00:00 --mem=$EVAL_MEM --wrap=\"$wrap_cmd\""
        return 0
    fi

    local out
    out=$(sbatch \
        --partition=allnodes \
        --cpus-per-task=4 \
        --mem="$EVAL_MEM" \
        --gres=gpu:1 \
        --time="${EVAL_HOURS}:00:00" \
        --output=/datastor2/jdr/logs/%j.out \
        --error=/datastor2/jdr/logs/%j.err \
        --job-name="trdyn-${SETTING}-${jobtag}" \
        --wrap="$wrap_cmd" 2>&1) || true
    echo "$out"
    local jid
    jid=$(echo "$out" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+$' | head -1)
    if [ -z "$jid" ]; then
        echo "FAILED to submit [$jobtag]"
        echo "$(date -u +%FT%TZ)  trdyn $MODEL $SETTING $jobtag  SUBMIT_FAILED" >> "$JOB_LOG"
    else
        echo "Submitted [$jobtag]: jobid=$jid"
        echo "$(date -u +%FT%TZ)  trdyn $MODEL $SETTING $jobtag  JOB=$jid  MODEL_ARG=$model_arg" >> "$JOB_LOG"
    fi
}

# 1) Base model (untrained). Setting-independent.
if [ -z "${NO_BASE:-}" ]; then
    submit "$MODEL" "base"
fi

# 2) Each saved epoch checkpoint.
for ep in $EPOCHS; do
    GLOB="${MODELS_DIR}/v7-${MODEL_REPL}-delta*-epoch${ep}--${TASK}-all--d2g--random--alpha1.0${SUFFIX}${MERGED_SUFFIX}"
    MDIR=$(ls -dt $GLOB 2>/dev/null | head -1 || true)
    if [ -z "$MDIR" ] || [ ! -d "$MDIR" ]; then
        echo "WARN: no checkpoint for epoch $ep (glob: $GLOB) -- skipping"
        continue
    fi
    echo "epoch $ep -> $MDIR"
    submit "$MDIR" "ep${ep}"
done
