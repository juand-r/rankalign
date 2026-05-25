#!/bin/bash
# Eval-only launcher (no train, no afterany dependency).
#
# Mirrors the eval half of `_overnight_launch.sh` exactly, so the resulting
# `scores_*.csv` filenames match what the v7 table builders expect.
#
# Usage:
#   bash scripts/_eval_only.sh DATASET MODEL SETTING TC
#     DATASET in {membership, persona, ifeval, humaneval}
#     MODEL   bare HF id (e.g. gemma-2-2b-it) or google/gemma-2-9b-it
#     SETTING in {s1..s7, s11, s12, s13}
#     TC      in {self, neg}
#
# Env (same as _overnight_launch.sh):
#   MODELS_DIR     default /datastor2/jdr/rankalign/models2
#   DRYRUN=1       print only, don't submit
#   DEP_JOBID      if set, submit with --dependency=afterany:$DEP_JOBID
#                  (use when the matching train is still running)
#   NO_BASE=1      omit --base-typcorr from the eval flag stack. Gives
#                  CSVs with `self-` (TC=self) or `neg-` (TC=neg) prefix
#                  rather than `basetyp-` / `basetypneg-`. Use this to
#                  populate the "PMI self" / "Neg self" columns of the
#                  v7 GenROC tables.
#
# Created 2026-05-24 to backfill missing neg-TC evals for s1/s2/s3 cells
# whose trained models already exist on disk (the v7 dispatcher previously
# only ran self-TC for those; now fixed but pre-fix trained models still
# need their neg-TC evals submitted manually).

set -euo pipefail

DATASET="${1:?DATASET required (membership|persona|ifeval|humaneval)}"
MODEL_NAME="${2:?MODEL required}"
SETTING="${3:?SETTING required (s1..s7, s11, s12, s13)}"
TC="${4:?TC required (self|neg)}"

case "$TC" in
    self|neg) ;;
    *) echo "FATAL: TC must be 'self' or 'neg' (got $TC)"; exit 1 ;;
esac

case "$MODEL_NAME" in
    google/*) MODEL="$MODEL_NAME" ;;
    *)        MODEL="google/$MODEL_NAME" ;;
esac

# ---- Map DATASET (copied verbatim from _overnight_launch.sh §dataset) ----
case "$DATASET" in
    membership)
        TASK="membership-sans-rosch-v0"
        EVAL_TASKS="rosch-bird rosch-carpenters-tool rosch-clothing rosch-fruit rosch-furniture rosch-sport rosch-toy rosch-vehicle rosch-vegetable rosch-weapon"
        case "$MODEL" in
            *9b-it*) EVAL_HOURS=4 ;;
            *)       EVAL_HOURS=3 ;;
        esac
        EVAL_MEM=48G
        ;;
    persona)
        TASK="persona-v1"
        EVAL_TASKS="persona-v1-psychopathy persona-v1-machiavellianism persona-v1-narcissism persona-v1-desire-to-create-allies persona-v1-interest-in-music persona-v1-interest-in-science"
        case "$MODEL" in
            *9b-it*) EVAL_HOURS=2 ;;
            *)       EVAL_HOURS=2 ;;
        esac
        EVAL_MEM=48G
        ;;
    ifeval)
        TASK="ifeval-concat"
        EVAL_TASKS=""
        for n in $(seq 1 21); do EVAL_TASKS="$EVAL_TASKS ifeval-prompt_$n"; done
        EVAL_TASKS="${EVAL_TASKS# }"
        case "$MODEL" in
            *9b-it*) EVAL_HOURS=5 ;;
            *)       EVAL_HOURS=4 ;;
        esac
        EVAL_MEM=64G
        ;;
    humaneval)
        TASK="humaneval-v2.1correct-upper"
        DATASET_DIR="v2.1correct-upper"
        REPO_ROOT_FOR_TASKS="$(cd "$(dirname "$0")/.." && pwd)"
        EVAL_TASKS=$(ls "$REPO_ROOT_FOR_TASKS/data/humaneval/${DATASET_DIR}/humaneval_"*.csv 2>/dev/null \
            | xargs -n1 basename 2>/dev/null | sed 's/\.csv$//' \
            | sed "s/^/${TASK}-/" | tr '\n' ' ')
        EVAL_TASKS="${EVAL_TASKS% }"
        EVAL_HOURS=8
        EVAL_MEM=128G
        ;;
    *)
        echo "Unknown DATASET: $DATASET"; exit 1 ;;
esac

# ---- Setting dispatch (copied verbatim from _overnight_launch.sh) ----
build_setting() {
    PREF_STR=""; NLLV_STR=""; NLLG_STR=""; FSX_STR=""; PPD_STR=""
    VLO_STR=""; SEMI_STR=""; CFT_STR=""; TC_LABEL=""
    case "$SETTING" in
        s1)
            TC_LABEL=""
            PREF_STR="--pref0.0"; NLLV_STR="--nllv1.0"; NLLG_STR="--nllg1.0"
            FSX_STR=""; PPD_STR=""; VLO_STR=""
            SEMI_STR="--labelonly0.1"
            ;;
        s2)
            TC_LABEL=""
            PREF_STR=""; NLLV_STR=""; NLLG_STR=""
            FSX_STR=""; PPD_STR=""; VLO_STR=""
            SEMI_STR="--semi0.1"
            ;;
        s3)
            TC_LABEL=""
            PREF_STR=""; NLLV_STR="--nllv1.0"; NLLG_STR="--nllg1.0"
            FSX_STR="--force-same-x"; PPD_STR="--ppd"; VLO_STR="--vallogodds"
            SEMI_STR="--semi0.1"
            ;;
        s4)
            TC_LABEL="--tc-self"
            PREF_STR=""; NLLV_STR="--nllv1.0"; NLLG_STR="--nllg1.0"
            FSX_STR="--force-same-x"; PPD_STR="--ppd"; VLO_STR="--vallogodds"
            SEMI_STR="--semi0.1"
            ;;
        s5)
            TC_LABEL="--tc-self"
            PREF_STR=""; NLLV_STR=""; NLLG_STR=""
            FSX_STR="--force-same-x"; PPD_STR="--ppd"; VLO_STR=""
            SEMI_STR="--semi0.1"
            ;;
        s6)
            TC_LABEL="--tc-self"
            PREF_STR=""; NLLV_STR=""; NLLG_STR=""
            FSX_STR=""; PPD_STR=""; VLO_STR=""
            SEMI_STR="--semi0.1"
            ;;
        s7)
            TC_LABEL="--tc-neg"
            PREF_STR=""; NLLV_STR="--nllv1.0"; NLLG_STR="--nllg1.0"
            FSX_STR="--force-same-x"; PPD_STR="--ppd"; VLO_STR="--vallogodds"
            SEMI_STR="--semi0.1"
            ;;
        s11)
            TC_LABEL="--tc-self"
            PREF_STR=""; NLLV_STR="--nllv1.0"; NLLG_STR="--nllg1.0"
            FSX_STR=""; PPD_STR=""; VLO_STR="--vallogodds"
            SEMI_STR="--semi0.1"
            ;;
        s12)
            TC_LABEL="--tc-neg"
            PREF_STR=""; NLLV_STR="--nllv1.0"; NLLG_STR="--nllg1.0"
            FSX_STR=""; PPD_STR=""; VLO_STR="--vallogodds"
            SEMI_STR="--semi0.1"
            ;;
        s13)
            TC_LABEL=""
            PREF_STR="--pref0.0"; NLLV_STR="--nllv1.0"; NLLG_STR="--nllg1.0"
            FSX_STR=""; PPD_STR=""; VLO_STR=""
            CFT_STR="--cft"
            SEMI_STR="--labelonly0.1"
            ;;
        *)
            echo "Unknown SETTING: $SETTING"; exit 1 ;;
    esac
}
build_setting

MODELS_DIR="${MODELS_DIR:-/datastor2/jdr/rankalign/models2}"
# EPOCH_GLOB controls which epoch dir gets matched. Default [012] picks the
# latest epoch via `ls -dt | head -1`. Override (e.g. EPOCH_GLOB=0) to lock
# the eval to a specific epoch — useful when an in-flight train is about to
# write a newer checkpoint that you don't yet want to evaluate.
EPOCH_GLOB="${EPOCH_GLOB:-[012]}"
MODEL_REPL=$(echo "$MODEL" | sed 's|/|--|g')

USE_LORA=1
[[ "$MODEL" == *"-2b"* || "$MODEL" == *"-2b-"* ]] && USE_LORA=0

case "$MODEL" in
    *gemma-2-2b-it*)   MODEL_TAG="2b-it" ;;
    *gemma-2-2b*)      MODEL_TAG="2b" ;;
    *gemma-2-9b-it*)   MODEL_TAG="9b-it" ;;
    *gemma-2-9b*)      MODEL_TAG="9b" ;;
    *gemma-4-31B-it*|*gemma-4-31b-it*) MODEL_TAG="g431Bit" ;;
    *) MODEL_TAG=$(echo "$MODEL" | sed 's|.*/||; s|gemma-||') ;;
esac

USE_GEMMA4_LORA=0
case "$MODEL" in
    *gemma-4-31B-it*|*gemma-4-31b-it*) USE_GEMMA4_LORA=1 ;;
esac

MERGED_SUFFIX=""
if [ "$USE_LORA" -eq 1 ] && [ "$USE_GEMMA4_LORA" -eq 0 ]; then
    MERGED_SUFFIX="_merged"
fi

SUFFIX="${TC_LABEL}--full-completion${PREF_STR}${NLLV_STR}${NLLG_STR}${FSX_STR}${PPD_STR}${CFT_STR}${VLO_STR}${SEMI_STR}--fix1"

PATH_PREFIX="${MODELS_DIR}/v7-${MODEL_REPL}-delta"
GLOB_PATH="${PATH_PREFIX}*-epoch${EPOCH_GLOB}--${TASK}-all--d2g--random--alpha1.0${SUFFIX}${MERGED_SUFFIX}"

# Pick the latest epoch matching the glob (eval the most-trained checkpoint).
MATCHED=$(ls -dt ${GLOB_PATH} 2>/dev/null | head -1 || true)

label="$DATASET-$(basename $MODEL)-$SETTING-evaltc-${TC}-EVALONLY"

echo "========================================"
echo "Eval-only: $label"
echo "  TASK:           $TASK"
echo "  MODEL:          $MODEL"
echo "  SETTING:        $SETTING"
echo "  TC:             $TC"
echo "  Expected glob:  ${GLOB_PATH}"
echo "  Matched dir:    ${MATCHED:-NONE}"
echo "========================================"

if [ -z "$MATCHED" ]; then
    echo "FATAL: no model dir matching glob; skipping submission."
    exit 1
fi

VENV_OVERRIDE=""
[ "$USE_GEMMA4_LORA" -eq 1 ] && VENV_OVERRIDE="/datastor2/jdr/venvs/gemma4"

# /datastor2 outputs dir (keeps /datastor1 from filling up; both v7 table
# builders scan /datastor2/jdr/rankalign/outputs as a SEARCH_DIR).
OUTPUTS_DIR_FLAG="--outputs-dir ${OUTPUTS_DIR:-/datastor2/jdr/rankalign/outputs}"
if [ -n "${NO_BASE:-}" ]; then
    # No --base-typcorr → CSV prefix is "self-" or "neg-" (PMI/Neg-self columns).
    if [ "$TC" = "self" ]; then
        EVAL_FLAGS="--self-typcorr --log-odds $OUTPUTS_DIR_FLAG"
    else
        EVAL_FLAGS="--neg-typcorr --log-odds $OUTPUTS_DIR_FLAG"
    fi
    NO_BASE_TAG="-nobase"
else
    # With --base-typcorr → CSV prefix is "basetyp-" or "basetypneg-" (PMI/Neg-base).
    if [ "$TC" = "self" ]; then
        EVAL_FLAGS="--self-typcorr --base-typcorr --base-model $MODEL --log-odds $OUTPUTS_DIR_FLAG"
    else
        EVAL_FLAGS="--neg-typcorr --base-typcorr --base-model $MODEL --log-odds $OUTPUTS_DIR_FLAG"
    fi
    NO_BASE_TAG=""
fi

# ifeval requires --disc-shots-zero: src/utils.py:make_prompt_ifeval
# raises NotImplementedError for shots != "zero" in the discriminator branch
# (line ~817), so each per-prompt task silently fails if this flag is absent.
if [ "$DATASET" = "ifeval" ]; then
    EVAL_FLAGS="$EVAL_FLAGS --disc-shots-zero"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV_EXPORT=""
[ -n "$VENV_OVERRIDE" ] && VENV_EXPORT="export VENV='$VENV_OVERRIDE'; "

WRAP_CMD="cd ${REPO_ROOT} && ${VENV_EXPORT}\
MODEL_DIR=\$(ls -dt ${GLOB_PATH} 2>/dev/null | head -1); \
if [ -z \"\$MODEL_DIR\" ] || [ ! -d \"\$MODEL_DIR\" ]; then echo 'SKIP - no matching ${GLOB_PATH}'; exit 0; fi; \
echo \"Eval model: \$MODEL_DIR\"; \
PYTHONUNBUFFERED=1 /usr/bin/time -v scripts/run_eval_semi.sh \"\$MODEL_DIR\" $EVAL_FLAGS -- ${EVAL_TASKS}"

if [ -n "${DRYRUN:-}" ]; then
    echo "DRYRUN. Would submit:"
    echo "  sbatch --partition=allnodes --cpus-per-task=4 --mem=$EVAL_MEM --gres=gpu:1 --time=${EVAL_HOURS}:00:00 \\"
    echo "    --output=/datastor2/jdr/logs/%j.out --error=/datastor2/jdr/logs/%j.err \\"
    echo "    --job-name=\"eval-${SETTING}-${DATASET}-${MODEL_TAG}-${TC}${NO_BASE_TAG}-only\" --wrap=\"<cmd>\""
    echo "  WRAP: $WRAP_CMD"
    exit 0
fi

DEP_FLAG=""
if [ -n "${DEP_JOBID:-}" ]; then
    DEP_FLAG="--dependency=afterany:${DEP_JOBID}"
    echo "Will submit with dependency: afterany:${DEP_JOBID}"
fi

EVAL_OUT=$(sbatch \
    --partition=allnodes \
    --cpus-per-task=4 \
    --mem="$EVAL_MEM" \
    --gres=gpu:1 \
    --time="${EVAL_HOURS}:00:00" \
    --output=/datastor2/jdr/logs/%j.out \
    --error=/datastor2/jdr/logs/%j.err \
    ${DEP_FLAG} \
    --job-name="eval-${SETTING}-${DATASET}-${MODEL_TAG}-${TC}${NO_BASE_TAG}-only" \
    --wrap="$WRAP_CMD" 2>&1) || true
echo "$EVAL_OUT"
EVAL_JOBID=$(echo "$EVAL_OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+$' | head -1)

if [ -z "$EVAL_JOBID" ]; then
    echo "FAILED to submit eval-only for $label"
    exit 2
fi

OVERNIGHT_DIR="$(cd "$(dirname "$0")/.." && pwd)/overnight"
mkdir -p "$OVERNIGHT_DIR"
echo "$(date -u +%FT%TZ)  $label  EVAL=$EVAL_JOBID  TC=$TC  PATH=$GLOB_PATH" >> "$OVERNIGHT_DIR/_overnight_jobids.txt"
echo "Eval submitted: jobid=$EVAL_JOBID  (tc=$TC, eval-only, no dependency)"
