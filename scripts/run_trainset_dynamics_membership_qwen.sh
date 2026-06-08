#!/bin/bash
# Train-set dynamics eval for membership-sans-rosch-v0 (Qwen3.5-9B).
#
# Same structure as run_trainset_dynamics.sh but for Qwen.
# Key differences:
#   - MODEL=Qwen/Qwen3.5-9B
#   - VENV=/datastor2/jdr/venvs/qwen35 (transformers 5.x for qwen3_5_text)
#   - Different deltas (1.53 for s1, 1.54 for others)
#   - Slightly different suffixes (qwen s1/s2 include --vallogodds)
#   - No --disc-shots flag needed (membership uses default few-shot)
#
# Usage:
#   bash scripts/run_trainset_dynamics_membership_qwen.sh [SETTING]
#     SETTING in {s1, s2, s3, s4, s7}
#
# Env:
#   MODELS_DIR    default /datastor2/jdr/rankalign/models2-rerun-wandb
#   OUTPUTS_DIR   default /datastor2/jdr/rankalign/outputs-trainset-dynamics
#   EVAL_HOURS    default 6
#   EPOCHS        default "0 1 2"
#   NO_BASE=1     skip base model eval
#   TC_OVERRIDE   override the TC eval variant (self/neg)
#   DRYRUN=1      print but don't submit

set -euo pipefail

SETTING="${1:-s3}"
MODEL="Qwen/Qwen3.5-9B"

TASK="membership-sans-rosch-v0"
MODELS_DIR="${MODELS_DIR:-/datastor2/jdr/rankalign/models2-rerun-wandb}"
OUTPUTS_DIR="${OUTPUTS_DIR:-/datastor2/jdr/rankalign/outputs-trainset-dynamics}"
EVAL_HOURS="${EVAL_HOURS:-6}"
EVAL_MEM=48G
EPOCHS="${EPOCHS-0 1 2}"

# Qwen3.5 merged checkpoints need transformers >= 5.x
export VENV="/datastor2/jdr/venvs/qwen35"

mkdir -p "$OUTPUTS_DIR"

# --- setting -> (suffix, default TC) -----------------------------------------
case "$SETTING" in
    s1)  SUFFIX="--full-completion--pref0.0--nllv1.0--nllg1.0--vallogodds--labelonly0.1--fix1"; TC="self" ;;
    s2)  SUFFIX="--full-completion--vallogodds--semi0.1--fix1"; TC="self" ;;
    s3)  SUFFIX="--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1"; TC="self" ;;
    s4)  SUFFIX="--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1"; TC="self" ;;
    s7)  SUFFIX="--tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1"; TC="neg" ;;
    *)   echo "FATAL: SETTING $SETTING not wired for qwen membership"; exit 1 ;;
esac

# Allow env override of TC
TC="${TC_OVERRIDE:-$TC}"

# TC eval flags (no --disc-shots needed for membership)
if [ "$TC" = "self" ]; then
    EVAL_FLAGS="--self-typcorr --base-typcorr --base-model $MODEL --log-odds"
else
    EVAL_FLAGS="--neg-typcorr --base-typcorr --base-model $MODEL --log-odds"
fi
EVAL_FLAGS="$EVAL_FLAGS --train --outputs-dir $OUTPUTS_DIR"

MERGED_SUFFIX="_merged"
MODEL_REPL=$(echo "$MODEL" | sed 's|/|--|g')
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OVERNIGHT_DIR="$REPO_ROOT/overnight"
mkdir -p "$OVERNIGHT_DIR"
JOB_LOG="$OVERNIGHT_DIR/_trainset_dynamics_jobids.txt"

echo "========================================"
echo "Train-set dynamics eval (membership, Qwen)"
echo "  MODEL:       $MODEL"
echo "  SETTING:     $SETTING (TC=$TC)"
echo "  TASK:        $TASK  (--train split)"
echo "  EVAL_FLAGS:  $EVAL_FLAGS"
echo "  VENV:        $VENV"
echo "  MODELS_DIR:  $MODELS_DIR"
echo "  OUTPUTS_DIR: $OUTPUTS_DIR"
echo "  EPOCHS:      $EPOCHS  (+ base unless NO_BASE)"
echo "========================================"

submit() {
    local model_arg="$1" jobtag="$2"
    local wrap_cmd="cd ${REPO_ROOT} && VENV=${VENV} PYTHONUNBUFFERED=1 /usr/bin/time -v \
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
        --job-name="trdyn-qw-mem-${SETTING}-${jobtag}" \
        --wrap="$wrap_cmd" 2>&1) || true
    echo "$out"
    local jid
    jid=$(echo "$out" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+$' | head -1)
    if [ -z "$jid" ]; then
        echo "FAILED to submit [$jobtag]"
        echo "$(date -u +%FT%TZ)  trdyn-qw-membership $SETTING $TC $jobtag  SUBMIT_FAILED" >> "$JOB_LOG"
    else
        echo "Submitted [$jobtag]: jobid=$jid"
        echo "$(date -u +%FT%TZ)  trdyn-qw-membership $SETTING $TC $jobtag  JOB=$jid  MODEL_ARG=$model_arg" >> "$JOB_LOG"
    fi
}

# 1) Base model
if [ -z "${NO_BASE:-}" ]; then
    submit "$MODEL" "base"
fi

# 2) Each epoch checkpoint
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
