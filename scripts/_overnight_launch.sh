#!/bin/bash
# Overnight unified train+eval launcher (2026-05-24).
#
# Usage:
#   bash scripts/_overnight_launch.sh DATASET MODEL SETTING
#     DATASET in {membership, persona, ifeval}
#     MODEL   in {gemma-2-2b-it, gemma-2-9b-it} (or any HF id; just the bare name)
#     SETTING in {s1, s2, s3, s4, s5, s6, s7, s11, s12}
#
# Submits two slurm jobs:
#   1. Train job via ~/.local/bin/run -> scripts/run_train_semi.sh ...
#   2. Eval job via direct sbatch with --dependency=afterany:<train_jobid>
#
# Env:
#   MODELS_DIR     default /datastor2/jdr/rankalign/models2
#   OUTPUTS_DIR    default ../outputs
#   DRYRUN=1       print only, don't submit
#   FORCE_RETRAIN  if not set: skip launching train if model dir already exists
#   FORCE_REEVAL   if not set: skip launching eval if score CSV already exists
#                  (the eval script already skips per-task; this skips the SBATCH)
#
# Output:
#   - Records (train_jobid, eval_jobid, paths) into
#     overnight/_overnight_jobids.txt for the loop monitor.
#   - The train command goes verbatim into docs/overnight_progress.md.

set -euo pipefail

DATASET="${1:?DATASET required (membership|persona|ifeval)}"
MODEL_NAME="${2:?MODEL required (e.g. gemma-2-2b-it or google/gemma-2-9b-it)}"
SETTING="${3:?SETTING required (s1..s7, s11, s12)}"

# Normalize MODEL: allow either bare or google/... form.
case "$MODEL_NAME" in
    google/*) MODEL="$MODEL_NAME" ;;
    *)        MODEL="google/$MODEL_NAME" ;;
esac

# Map DATASET to (TASK, EVAL_TASKS, GPUS, TRAIN_HOURS, EVAL_HOURS, TRAIN_MEM, MAX_SEQ).
case "$DATASET" in
    membership)
        TASK="membership-sans-rosch-v0"
        EVAL_TASKS="rosch-bird rosch-carpenters-tool rosch-clothing rosch-fruit rosch-furniture rosch-sport rosch-toy rosch-vehicle rosch-vegetable rosch-weapon"
        GPUS=1
        # Observed ~2s/it for 2b on persona-v1 (similar size). Membership has
        # 5113 samples * 3 epochs ~= 8.5h on 2b. 9b is ~3x. Add slack.
        case "$MODEL" in
            *9b-it*) TRAIN_HOURS=24 ; EVAL_HOURS=4 ;;
            *)       TRAIN_HOURS=12 ; EVAL_HOURS=3 ;;
        esac
        TRAIN_MEM=64G
        EVAL_MEM=48G
        MAX_SEQ_FLAG=""
        ;;
    persona)
        TASK="persona-v1"
        EVAL_TASKS="persona-v1-psychopathy persona-v1-machiavellianism persona-v1-narcissism persona-v1-desire-to-create-allies persona-v1-interest-in-music persona-v1-interest-in-science"
        GPUS=1
        # 5110 samples * 3 epochs at 2s/it -> ~8.5h for 2b; ~2-3x for 9b LoRA.
        case "$MODEL" in
            *9b-it*) TRAIN_HOURS=20 ; EVAL_HOURS=2 ;;
            *)       TRAIN_HOURS=10 ; EVAL_HOURS=2 ;;
        esac
        TRAIN_MEM=64G
        EVAL_MEM=48G
        MAX_SEQ_FLAG=""
        ;;
    ifeval)
        TASK="ifeval-concat"
        EVAL_TASKS=""
        for n in $(seq 1 21); do EVAL_TASKS="$EVAL_TASKS ifeval-prompt_$n"; done
        # Trim leading space.
        EVAL_TASKS="${EVAL_TASKS# }"
        # ifeval-concat: ~5110 samples but longer prompts; longer wall.
        # 9b-it on 2 GPUs (model parallel) helps but still slow.
        case "$MODEL" in
            *9b-it*) GPUS=2 ; TRAIN_HOURS=30 ; EVAL_HOURS=5 ;;
            *)       GPUS=1 ; TRAIN_HOURS=14 ; EVAL_HOURS=4 ;;
        esac
        TRAIN_MEM=96G
        EVAL_MEM=64G
        # ifeval has long prompts; cap seq len to keep VRAM in check.
        MAX_SEQ_FLAG="--max-seq-len 1024"
        ;;
    *)
        echo "Unknown DATASET: $DATASET (membership|persona|ifeval)"
        exit 1
        ;;
esac

# Map SETTING to (LOSS, SEMI_MODE, FSX_FLAG, TC_FLAG, LOGODDS_FLAG, TC_EVAL_LIST,
# DIR_SUFFIX_FRAGMENTS).
#
# DIR_SUFFIX_FRAGMENTS is the part of the v7 dir name after the alpha-prefix and
# before --semi*--fix1, used to construct the expected save path. We assemble it
# in the same order as ranking_loss_ref_fix.py @ line 2182.
#
# The TC_EVAL_LIST is space-separated, each token is one of:
#   self  -> --self-typcorr
#   neg   -> --neg-typcorr
# and we always also include --base-typcorr for each.
build_setting() {
    SET_FLAGS=""
    TC_LABEL=""
    PREF_STR=""
    NLLV_STR=""
    NLLG_STR=""
    FSX_STR=""
    PPD_STR=""
    VLO_STR=""
    SEMI_STR=""

    case "$SETTING" in
        s1)   # SFT-lo: sft + labelonly + no fsx
            LOSS="sft" ; SEMI_MODE="labelonly" ; RATIO="0.1"
            FSX_FLAG="--no-force-same-x"
            TC_FLAG="" ; TC_LABEL=""
            LOGODDS_FLAG=""
            # Only "self" eval to cap queue size; backfill "neg" later if time.
            TC_EVAL_LIST="self"
            PREF_STR="--pref0.0" ; NLLV_STR="--nllv1.0" ; NLLG_STR="--nllg1.0"
            FSX_STR="" ; PPD_STR=""
            VLO_STR=""
            SEMI_STR="--labelonly0.1"
            ;;
        s2)   # RankAlign: pref-only + semi + no fsx
            LOSS="pref-only" ; SEMI_MODE="semi" ; RATIO="0.1"
            FSX_FLAG="--no-force-same-x"
            TC_FLAG="" ; TC_LABEL=""
            LOGODDS_FLAG=""
            TC_EVAL_LIST="self"
            PREF_STR="" ; NLLV_STR="" ; NLLG_STR=""
            FSX_STR="" ; PPD_STR=""
            VLO_STR=""
            SEMI_STR="--semi0.1"
            ;;
        s3)   # New+fsx: comb + semi + log-odds + fsx (+ ppd + sbm-global)
            LOSS="comb" ; SEMI_MODE="semi" ; RATIO="0.1"
            FSX_FLAG=""  # default ON
            TC_FLAG="" ; TC_LABEL=""
            LOGODDS_FLAG="--log-odds"
            TC_EVAL_LIST="self"
            PREF_STR="" ; NLLV_STR="--nllv1.0" ; NLLG_STR="--nllg1.0"
            FSX_STR="--force-same-x" ; PPD_STR="--ppd"
            VLO_STR="--vallogodds"
            SEMI_STR="--semi0.1"
            ;;
        s4)   # New+fsx+selfTC
            LOSS="comb" ; SEMI_MODE="semi" ; RATIO="0.1"
            FSX_FLAG=""
            TC_FLAG="--self-typcorr" ; TC_LABEL="--tc-self"
            LOGODDS_FLAG="--log-odds"
            TC_EVAL_LIST="self"
            PREF_STR="" ; NLLV_STR="--nllv1.0" ; NLLG_STR="--nllg1.0"
            FSX_STR="--force-same-x" ; PPD_STR="--ppd"
            VLO_STR="--vallogodds"
            SEMI_STR="--semi0.1"
            ;;
        s5)   # RankAlign+fsx+selfTC (per IRP §1: drop --validator-log-odds)
            LOSS="pref-only" ; SEMI_MODE="semi" ; RATIO="0.1"
            FSX_FLAG=""
            TC_FLAG="--self-typcorr" ; TC_LABEL="--tc-self"
            LOGODDS_FLAG=""  # explicitly NO vlo (was a historical bug)
            TC_EVAL_LIST="self"
            PREF_STR="" ; NLLV_STR="" ; NLLG_STR=""
            FSX_STR="--force-same-x" ; PPD_STR="--ppd"
            VLO_STR=""
            SEMI_STR="--semi0.1"
            ;;
        s6)   # RankAlign+selfTC (no fsx)
            LOSS="pref-only" ; SEMI_MODE="semi" ; RATIO="0.1"
            FSX_FLAG="--no-force-same-x"
            TC_FLAG="--self-typcorr" ; TC_LABEL="--tc-self"
            LOGODDS_FLAG=""
            TC_EVAL_LIST="self"
            PREF_STR="" ; NLLV_STR="" ; NLLG_STR=""
            FSX_STR="" ; PPD_STR=""
            VLO_STR=""
            SEMI_STR="--semi0.1"
            ;;
        s7)   # New+fsx+negTC
            LOSS="comb" ; SEMI_MODE="semi" ; RATIO="0.1"
            FSX_FLAG=""
            TC_FLAG="--neg-typcorr" ; TC_LABEL="--tc-neg"
            LOGODDS_FLAG="--log-odds"
            TC_EVAL_LIST="neg"
            PREF_STR="" ; NLLV_STR="--nllv1.0" ; NLLG_STR="--nllg1.0"
            FSX_STR="--force-same-x" ; PPD_STR="--ppd"
            VLO_STR="--vallogodds"
            SEMI_STR="--semi0.1"
            ;;
        s11)  # New+selfTC (no fsx)
            LOSS="comb" ; SEMI_MODE="semi" ; RATIO="0.1"
            FSX_FLAG="--no-force-same-x"
            TC_FLAG="--self-typcorr" ; TC_LABEL="--tc-self"
            LOGODDS_FLAG="--log-odds"
            TC_EVAL_LIST="self"
            PREF_STR="" ; NLLV_STR="--nllv1.0" ; NLLG_STR="--nllg1.0"
            FSX_STR="" ; PPD_STR=""
            VLO_STR="--vallogodds"
            SEMI_STR="--semi0.1"
            ;;
        s12)  # New+negTC (no fsx)
            LOSS="comb" ; SEMI_MODE="semi" ; RATIO="0.1"
            FSX_FLAG="--no-force-same-x"
            TC_FLAG="--neg-typcorr" ; TC_LABEL="--tc-neg"
            LOGODDS_FLAG="--log-odds"
            TC_EVAL_LIST="neg"
            PREF_STR="" ; NLLV_STR="--nllv1.0" ; NLLG_STR="--nllg1.0"
            FSX_STR="" ; PPD_STR=""
            VLO_STR="--vallogodds"
            SEMI_STR="--semi0.1"
            ;;
        *)
            echo "Unknown SETTING: $SETTING"
            exit 1
            ;;
    esac
}

build_setting

# Optional WALLTIME env var overrides the per-task heuristic. Useful when
# the dispatcher caller knows the cluster is contended and a shorter
# walltime is more likely to backfill into a small slot.
if [ -n "${WALLTIME:-}" ]; then
    TRAIN_HOURS="$WALLTIME"
fi

# All the universal flags for fix1 overnight runs.
COMMON_FLAGS=( --script ranking_loss_ref_fix.py
               --disc-shots few
               --delta-bins 10 )
[ -n "$MAX_SEQ_FLAG" ] && COMMON_FLAGS+=( $MAX_SEQ_FLAG )
[ -n "$FSX_FLAG" ]     && COMMON_FLAGS+=( $FSX_FLAG )
[ -n "$TC_FLAG" ]      && COMMON_FLAGS+=( $TC_FLAG )
[ -n "$LOGODDS_FLAG" ] && COMMON_FLAGS+=( $LOGODDS_FLAG )

# fsx settings get ppd + sbm-global. (Detect via FSX_STR which is set when the
# setting USES fsx.)
if [ -n "$FSX_STR" ]; then
    COMMON_FLAGS+=( --per-prompt-delta --shape-budget-mode global )
fi

# Use absolute /datastor2 models dir to avoid /datastor1 fill-up.
MODELS_DIR="${MODELS_DIR:-/datastor2/jdr/rankalign/models2}"
COMMON_FLAGS+=( --models-dir "$MODELS_DIR" )

# Persona-v1 task name in the dir is "persona-v1" (matches TASK var).
# Eval glob matches any saved epoch (0/1/2). The wrap uses `ls -dt | head -1`
# to pick the most recently saved epoch dir, so even a walltime-killed run
# (which only saved epoch0 or epoch1) is evaluable.
EPOCH_GLOB="[012]"
DELTA_PLACEHOLDER="DELTA"  # we don't know delta exactly until script runs;
                            # we'll glob-match instead.

# Build the variable-suffix string the python script appends.
# NOTE: order from ranking_loss_ref_fix.py L2182:
#   {tc}{lenorm}{single}{full-completion}{eos}{pref}{nllv}{nllg}{fsx}{ppd}{valboost}{vallogodds}{semi}{fix1}
# with single, lenorm, eos, valboost all empty in our case.
SUFFIX="${TC_LABEL}--full-completion${PREF_STR}${NLLV_STR}${NLLG_STR}${FSX_STR}${PPD_STR}${VLO_STR}${SEMI_STR}--fix1"

MODEL_REPL=$(echo "$MODEL" | sed 's|/|--|g')

# Determine LoRA merge suffix: matches the python script's logic in run_train_semi.sh
# (LoRA flag set when MODEL doesn't contain -2b- substring).
USE_LORA=1
if [[ "$MODEL" == *"-2b"* || "$MODEL" == *"-2b-"* ]]; then
    USE_LORA=0
fi
MERGED_SUFFIX=""
[ "$USE_LORA" -eq 1 ] && MERGED_SUFFIX="_merged"

PATH_PREFIX="${MODELS_DIR}/v7-${MODEL_REPL}-delta"

# We don't know the exact delta until --delta-bins computes it from the data,
# so for the expected eval path we use a glob. The eval wrapper uses `ls -d`
# at sbatch runtime to pick the right one.
GLOB_PATH="${PATH_PREFIX}*-epoch${EPOCH_GLOB}--${TASK}-all--d2g--random--alpha1.0${SUFFIX}${MERGED_SUFFIX}"

OVERNIGHT_DIR="$(cd "$(dirname "$0")/.." && pwd)/overnight"
mkdir -p "$OVERNIGHT_DIR"
JOB_LOG="$OVERNIGHT_DIR/_overnight_jobids.txt"

# Eval task list - quote-protected.
read -r -a EVAL_TASKS_ARR <<< "$EVAL_TASKS"

# Build eval flags per TC variant
build_eval_flags() {
    local tc="$1"
    if [ "$tc" = "self" ]; then
        echo "--self-typcorr --base-typcorr --base-model $MODEL --log-odds"
    elif [ "$tc" = "neg" ]; then
        echo "--neg-typcorr --base-typcorr --base-model $MODEL --log-odds"
    else
        echo "--base-typcorr --base-model $MODEL --log-odds"
    fi
}

label="$DATASET-$(basename $MODEL)-$SETTING"

echo "========================================"
echo "Overnight launch: $label"
echo "  TASK:          $TASK"
echo "  MODEL:         $MODEL  (LoRA=${USE_LORA})"
echo "  SETTING:       $SETTING (loss=$LOSS, semi_mode=$SEMI_MODE, fsx_flag='$FSX_FLAG', tc='$TC_FLAG', vlo='$LOGODDS_FLAG')"
echo "  GPUs:          $GPUS"
echo "  TRAIN_HOURS:   $TRAIN_HOURS"
echo "  EVAL_HOURS:    $EVAL_HOURS"
echo "  COMMON_FLAGS:  ${COMMON_FLAGS[*]}"
echo "  EVAL TC list:  $TC_EVAL_LIST"
echo "  Expected save: ${GLOB_PATH}"
echo "========================================"

if [ -n "${DRYRUN:-}" ]; then
    echo "DRYRUN. Would run:"
    echo "  run $GPUS $TRAIN_HOURS --cpu 4 --mem $TRAIN_MEM scripts/run_train_semi.sh $MODEL $TASK $LOSS $SEMI_MODE $RATIO ${COMMON_FLAGS[*]}"
    exit 0
fi

# 1) Submit train.
TRAIN_OUT=$(run "$GPUS" "$TRAIN_HOURS" --cpu 4 --mem "$TRAIN_MEM" \
    scripts/run_train_semi.sh "$MODEL" "$TASK" "$LOSS" "$SEMI_MODE" "$RATIO" "${COMMON_FLAGS[@]}" 2>&1) || true
echo "$TRAIN_OUT"
TRAIN_JOBID=$(echo "$TRAIN_OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+$' | head -1)

if [ -z "$TRAIN_JOBID" ]; then
    echo "FAILED to submit train job for $label"
    echo "$(date -u +%FT%TZ)  $label  TRAIN_FAILED  $TRAIN_OUT" >> "$JOB_LOG"
    exit 2
fi

echo "Train submitted: jobid=$TRAIN_JOBID"

# 2) Submit eval(s) chained on afterany.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

submit_eval() {
    local tc="$1"
    local eval_flags
    eval_flags=$(build_eval_flags "$tc")
    local eval_label="${label}-evaltc-${tc}"

    # WRAP_CMD picks the matching epoch2 dir at sbatch run-time. If multiple
    # match (sweep + main collision), pick the most recent by mtime.
    # If none match (training died early), exit 0 with SKIP message.
    local wrap_cmd
    wrap_cmd="cd ${REPO_ROOT} && \
MODEL_DIR=\$(ls -dt ${GLOB_PATH} 2>/dev/null | head -1); \
if [ -z \"\$MODEL_DIR\" ] || [ ! -d \"\$MODEL_DIR\" ]; then echo 'SKIP - no matching ${GLOB_PATH}'; exit 0; fi; \
echo \"Eval model: \$MODEL_DIR\"; \
PYTHONUNBUFFERED=1 /usr/bin/time -v scripts/run_eval_semi.sh \"\$MODEL_DIR\" $eval_flags -- ${EVAL_TASKS}"

    EVAL_OUT=$(sbatch \
        --partition=allnodes \
        --cpus-per-task=4 \
        --mem="$EVAL_MEM" \
        --gres=gpu:1 \
        --time="${EVAL_HOURS}:00:00" \
        --output=/datastor2/jdr/logs/%j.out \
        --error=/datastor2/jdr/logs/%j.err \
        --dependency=afterany:"$TRAIN_JOBID" \
        --job-name="eval-${SETTING}-${DATASET}-${tc}" \
        --wrap="$wrap_cmd" 2>&1) || true
    echo "$EVAL_OUT"
    EVAL_JOBID=$(echo "$EVAL_OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+$' | head -1)

    if [ -z "$EVAL_JOBID" ]; then
        echo "FAILED to submit eval ($tc) for $label"
        echo "$(date -u +%FT%TZ)  $eval_label  EVAL_FAILED  $EVAL_OUT" >> "$JOB_LOG"
    else
        echo "Eval submitted: jobid=$EVAL_JOBID  (tc=$tc, dep=afterany:$TRAIN_JOBID)"
        echo "$(date -u +%FT%TZ)  $eval_label  TRAIN=$TRAIN_JOBID  EVAL=$EVAL_JOBID  TC=$tc  PATH=$GLOB_PATH" >> "$JOB_LOG"
    fi
}

for tc in $TC_EVAL_LIST; do
    submit_eval "$tc"
done

echo "$(date -u +%FT%TZ)  $label  TRAIN=$TRAIN_JOBID  TC_EVAL_LIST=$TC_EVAL_LIST" >> "$JOB_LOG"
echo "Done: $label"
