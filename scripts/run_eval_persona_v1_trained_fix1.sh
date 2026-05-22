#!/bin/bash
# Post-train eval for FIX1 persona-v1 trained models (v7- in models2/).
#
# Sibling of run_eval_persona_v1_trained.sh. Differences:
#   - Looks in $MODELS_DIR (default: $REPO_ROOT/models2) for v7- checkpoints
#     with --fix1 suffix.
#   - Suffix list drops --force-same-x (fix1 launcher uses --no-force-same-x).
#   - Only handles variants 3, 4, 7 (the fix1 variants we care about).
#   - Inlines sbatch (rather than going through ~/.local/bin/run) so we can
#     pass --dependency=afterany:<TRAIN_JOBID> for auto-eval-on-training-done.
#
# Slurm dependency support:
#   TRAIN_JOBID=<jobid> bash scripts/run_eval_persona_v1_trained_fix1.sh ...
#     -> each submitted eval job uses --dependency=afterany:<TRAIN_JOBID>
#        so it sits in PD until the training job ends (success / fail /
#        walltime kill). Use 'afterany' so we still attempt eval if
#        training timed out -- if epoch2 dir is missing the script SKIPs.
#
# Usage:
#   bash scripts/run_eval_persona_v1_trained_fix1.sh [BASE_MODEL] [VARIANT...]
#
#   BASE_MODEL: HuggingFace id (default: google/gemma-2-9b-it).
#   VARIANT:    one or more of {3, 4, 7}. Default: 3 4 7.
#
# Env:
#   BASE_MODEL    - default base model when no positional argument given
#   HOURS         - Slurm walltime per job (default: 2)
#   CPUS          - CPUs per task (default: 6)
#   MEM           - memory per job (default: 60G)
#   PARTITION     - slurm partition (default: allnodes)
#   MODELS_DIR    - dir containing v7- trained models (default: models2/)
#   EPOCH         - which epoch to eval (default: 2 = final of 3-epoch training)
#   DISC_SHOTS    - 'zero' (default, matches 9b-it/2b-it training) or
#                   'few'/'auto' (matches 2b training)
#   TRAIN_JOBID   - optional: slurm jobid to depend on
#                   (afterany so we eval whatever epoch2 dir survives)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BASE_MODEL="${1:-${BASE_MODEL:-google/gemma-2-9b-it}}"
shift || true
HOURS="${HOURS:-2}"
CPUS="${CPUS:-6}"
MEM="${MEM:-60G}"
PARTITION="${PARTITION:-allnodes}"
MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models2}"
EPOCH="${EPOCH:-2}"
DISC_SHOTS="${DISC_SHOTS:-zero}"
TRAIN_JOBID="${TRAIN_JOBID:-}"

# Variants to run (positional; defaults to all 3 fix1 variants).
if [ "$#" -eq 0 ]; then
    VARIANTS=(3 4 7)
else
    VARIANTS=("$@")
fi

TASKS=(
    persona-v1-psychopathy
    persona-v1-machiavellianism
    persona-v1-narcissism
    persona-v1-desire-to-create-allies
    persona-v1-interest-in-music
    persona-v1-interest-in-science
)

# disc-shots flag selection -- mirrors the parent eval launcher's logic.
if [ "$DISC_SHOTS" = "few" ] || [ "$DISC_SHOTS" = "auto" ]; then
    EVAL_COMMON="--log-odds"
else
    EVAL_COMMON="--log-odds --disc-shots-zero"
fi

# Variant suffixes for FIX1: same as parent v6 except no --force-same-x and
# trailing --fix1.
declare -A SUFFIXES=(
    ["3.New(fix1)"]="--full-completion--nllv1.0--nllg1.0--vallogodds--semi0.1--fix1"
    ["4.New+selfTC(fix1)"]="--tc-self--full-completion--nllv1.0--nllg1.0--vallogodds--semi0.1--fix1"
    ["7.New+negTC(fix1)"]="--tc-neg--full-completion--nllv1.0--nllg1.0--vallogodds--semi0.1--fix1"
)
declare -A TC_POLICY=(
    ["3.New(fix1)"]="both"
    ["4.New+selfTC(fix1)"]="self"
    ["7.New+negTC(fix1)"]="neg"
)

# Map numeric arg -> variant key.
get_variant_key() {
    case "$1" in
        3) echo "3.New(fix1)" ;;
        4) echo "4.New+selfTC(fix1)" ;;
        7) echo "7.New+negTC(fix1)" ;;
        *) echo "" ;;
    esac
}

# LoRA-merged for non-2b models, plain for 2b.
USE_LORA=1
if [[ "$BASE_MODEL" == *"-2b"* || "$BASE_MODEL" == *"-2b-"* ]]; then
    USE_LORA=0
fi
MERGED_SUFFIX=""
[[ "$USE_LORA" -eq 1 ]] && MERGED_SUFFIX="_merged"

BASE_REPL=$(echo "$BASE_MODEL" | sed 's|/|--|g')
PATH_PREFIX="${MODELS_DIR}/v7-${BASE_REPL}-delta0.15-epoch${EPOCH}--persona-v1-all--d2g--random--alpha1.0"

OVERNIGHT_DIR="$(dirname "$0")/../overnight"
mkdir -p "$OVERNIGHT_DIR"
MODEL_TAG=$(basename "$BASE_MODEL" | sed 's|/|_|g; s|--|_|g')
DISC_TAG="_disc-${DISC_SHOTS}"
JOBID_FILE="$OVERNIGHT_DIR/persona_v1_eval_trained_jobids_${MODEL_TAG}${DISC_TAG}_fix1.txt"
# Append rather than truncate, so multiple invocations (one per training job)
# accumulate into one file.
touch "$JOBID_FILE"

DEP_FLAG=""
if [ -n "$TRAIN_JOBID" ]; then
    DEP_FLAG="--dependency=afterany:${TRAIN_JOBID}"
fi

echo "========================================"
echo "Persona-v1 FIX1 trained-model eval launcher"
echo "  BASE_MODEL: $BASE_MODEL"
echo "  EPOCH:      $EPOCH"
echo "  USE_LORA:   $USE_LORA  (merged suffix: '${MERGED_SUFFIX}')"
echo "  HOURS:      $HOURS / job"
echo "  CPUS / MEM: $CPUS / $MEM  on partition $PARTITION"
echo "  Tasks:      ${#TASKS[@]}  (${TASKS[*]})"
echo "  EVAL_COMMON: $EVAL_COMMON"
echo "  DISC_SHOTS: $DISC_SHOTS"
echo "  MODELS_DIR: $MODELS_DIR"
echo "  PATH_PREFIX: $PATH_PREFIX"
echo "  VARIANTS:   ${VARIANTS[*]}"
[ -n "$TRAIN_JOBID" ] && echo "  TRAIN_JOBID: $TRAIN_JOBID  (eval will run with --dependency=afterany:$TRAIN_JOBID)"
echo "  Jobid log:  $JOBID_FILE"
echo "========================================"

# Inline sbatch (mirrors ~/.local/bin/run but adds --dependency support).
# Args: <label> <model_path> <tc_flag> <tc_label>
submit() {
    local label="$1"; shift
    local model_path="$1"; shift
    local tc_flag="$1"; shift
    local tc_label="$1"; shift

    # NOTE: with TRAIN_JOBID, model_path may not exist YET (training is still
    # running). We let sbatch submit anyway; the actual run_eval_semi.sh
    # invocation at scheduling time will fail loudly if the dir is missing
    # post-training. Without TRAIN_JOBID we still gate on the dir existing
    # (the model is already on disk and we want a fast-fail).
    if [ -z "$TRAIN_JOBID" ] && [ ! -d "$model_path" ]; then
        echo ""
        echo "[$label / $tc_label] SKIP - model dir not found: $model_path"
        echo "(no jobid - SKIP)  $label  $tc_label  $model_path" >> "$JOBID_FILE"
        return
    fi

    echo ""
    echo ">>> [$label / $tc_label] $model_path"

    # Compose the wrapped command (same as run_eval_semi.sh expects).
    # Guard with `test -d`: if training got walltime-killed before epoch${EPOCH}
    # was saved, the eval job will gracefully exit 0 instead of crashing on a
    # missing checkpoint.
    WRAP_CMD="test -d ${model_path} || { echo 'SKIP - model dir not found: ${model_path}'; exit 0; }; PYTHONUNBUFFERED=1 /usr/bin/time -v scripts/run_eval_semi.sh ${model_path} ${tc_flag} --base-typcorr --base-model ${BASE_MODEL} ${EVAL_COMMON} -- ${TASKS[*]}"

    if [ -n "${DRYRUN:-}" ]; then
        echo "DRYRUN: sbatch --partition=$PARTITION --cpus-per-task=$CPUS --mem=$MEM --gres=gpu:1 --time=${HOURS}:00:00 --output=/datastor2/jdr/logs/%j.out --error=/datastor2/jdr/logs/%j.err $DEP_FLAG --wrap=\"$WRAP_CMD\""
        OUT="DRYRUN-no-submit"
    else
        OUT=$(sbatch \
            --partition="$PARTITION" \
            --cpus-per-task="$CPUS" \
            --mem="$MEM" \
            --gres=gpu:1 \
            --time="${HOURS}:00:00" \
            --output=/datastor2/jdr/logs/%j.out \
            --error=/datastor2/jdr/logs/%j.err \
            $DEP_FLAG \
            --wrap="$WRAP_CMD" 2>&1) || true
        echo "$OUT"
    fi
    JOBID=$(echo "$OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+$' | head -1)
    if [ -n "$JOBID" ]; then
        echo "$JOBID  $label  $tc_label  $model_path  dep=${TRAIN_JOBID:-none}" >> "$JOBID_FILE"
    else
        echo "(no jobid captured)  $label  $tc_label  $model_path" >> "$JOBID_FILE"
    fi
}

for v in "${VARIANTS[@]}"; do
    variant_key=$(get_variant_key "$v")
    if [ -z "$variant_key" ]; then
        echo "ERROR: variant $v not in {3,4,7}; skipping" >&2
        continue
    fi
    suffix="${SUFFIXES[$variant_key]}"
    policy="${TC_POLICY[$variant_key]}"
    model_path="${PATH_PREFIX}${suffix}${MERGED_SUFFIX}"
    case "$policy" in
        both)
            submit "$variant_key" "$model_path" "--self-typcorr" "self+base"
            submit "$variant_key" "$model_path" "--neg-typcorr"  "neg+base"
            ;;
        self)
            submit "$variant_key" "$model_path" "--self-typcorr" "self+base"
            ;;
        neg)
            submit "$variant_key" "$model_path" "--neg-typcorr"  "neg+base"
            ;;
        *)
            echo "ERROR: unknown TC_POLICY '$policy' for $variant_key" >&2
            ;;
    esac
done

echo ""
echo "========================================"
echo "Done. Jobid log:"
cat "$JOBID_FILE"
echo ""
echo "Tail logs with: tail -f /datastor2/jdr/logs/<JOBID>.out"
echo "Check progress with: squeue -u \$USER"
