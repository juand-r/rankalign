#!/bin/bash
# Post-train eval for the 9 main fix1 training runs (persona-v1).
#
# Globs models2/ for v7-<base>-delta*-epoch*--persona-v1-all--*--fix1 dirs
# (handling auto-deltas, _merged for 9b-it), picks the latest epoch dir per
# (base, variant), and submits the eval(s) the user specified:
#   variant 3 (no TC at train) -> --self-typcorr+base, --neg-typcorr+base (2 evals)
#   variant 4 (--tc-self at train) -> --self-typcorr+base only
#   variant 7 (--tc-neg at train) -> --neg-typcorr+base only
#
# Usage:
#   bash scripts/run_eval_main9_fix1.sh <BASE_MODEL> [VARIANT...]
#
# Args:
#   BASE_MODEL: HF id (e.g. google/gemma-2-9b-it)
#   VARIANT:    one or more of {3, 4, 7}. Default: 3 4 7.
#
# Env:
#   HOURS    - eval walltime (default: 2)
#   CPUS, MEM, PARTITION
#   MODELS_DIR - default ../models2 from repo root
#   DRYRUN=1 - print without submit

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BASE_MODEL="${1:?Usage: $0 <BASE_MODEL> [VARIANT...]}"
shift || true

HOURS="${HOURS:-2}"
CPUS="${CPUS:-6}"
MEM="${MEM:-60G}"
PARTITION="${PARTITION:-allnodes}"
MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models2}"

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

# Variant 3 has no TC training flag; 4 has --tc-self; 7 has --tc-neg.
declare -A SUFFIX_BY_VAR=(
    [3]="--full-completion--nllv1.0--nllg1.0--vallogodds--semi0.1--fix1"
    [4]="--tc-self--full-completion--nllv1.0--nllg1.0--vallogodds--semi0.1--fix1"
    [7]="--tc-neg--full-completion--nllv1.0--nllg1.0--vallogodds--semi0.1--fix1"
)

# 9b-it uses _merged; 2b/2b-it use plain LoRA dir.
case "$BASE_MODEL" in
    *9b-it*) USE_MERGED=1 ;;
    *)       USE_MERGED=0 ;;
esac
MERGED_SUFFIX=""
[ "$USE_MERGED" = "1" ] && MERGED_SUFFIX="_merged"

BASE_REPL=$(echo "$BASE_MODEL" | sed 's|/|--|g')

OVERNIGHT_DIR="${REPO_ROOT}/overnight"
mkdir -p "$OVERNIGHT_DIR"
MODEL_TAG=$(basename "$BASE_MODEL" | sed 's|/|_|g; s|--|_|g')
JOBID_FILE="$OVERNIGHT_DIR/persona_v1_eval_main9_${MODEL_TAG}.txt"
touch "$JOBID_FILE"

echo "========================================"
echo "main9 fix1 eval launcher"
echo "  BASE_MODEL:  $BASE_MODEL"
echo "  USE_MERGED:  $USE_MERGED  (suffix '${MERGED_SUFFIX}')"
echo "  VARIANTS:    ${VARIANTS[*]}"
echo "  MODELS_DIR:  $MODELS_DIR"
echo "  Jobid log:   $JOBID_FILE"
echo "========================================"

# Find latest epoch dir for a given variant suffix.
# Echoes path or empty if not found.
find_latest_epoch() {
    local var_suffix="$1"
    local pattern="${MODELS_DIR}/v7-${BASE_REPL}-delta*-epoch*--persona-v1-all--d2g--random--alpha1.0${var_suffix}${MERGED_SUFFIX}"
    # Sort by epoch number descending so the first match is the highest epoch.
    # We rely on lexical sort + a small awk to extract epoch.
    local best_path=""
    local best_epoch=-1
    shopt -s nullglob
    for path in $pattern; do
        # Extract epoch from the dirname
        local name=$(basename "$path")
        local ep=$(echo "$name" | grep -oE 'epoch[0-9]+' | head -1 | sed 's/epoch//')
        if [ -n "$ep" ] && [ "$ep" -gt "$best_epoch" ]; then
            best_epoch="$ep"
            best_path="$path"
        fi
    done
    shopt -u nullglob
    echo "$best_path"
}

submit_eval() {
    local label="$1"; shift
    local model_path="$1"; shift
    local tc_flag="$1"; shift

    EVAL_COMMON="--log-odds"
    WRAP_CMD="PYTHONUNBUFFERED=1 /usr/bin/time -v scripts/run_eval_semi.sh ${model_path} ${tc_flag} --base-typcorr --base-model ${BASE_MODEL} ${EVAL_COMMON} -- ${TASKS[*]}"

    if [ -n "${DRYRUN:-}" ]; then
        echo "DRYRUN: sbatch ... --wrap=\"$WRAP_CMD\""
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
            --wrap="$WRAP_CMD" 2>&1) || true
        echo "$OUT"
    fi
    JOBID=$(echo "$OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+$' | head -1)
    if [ -n "$JOBID" ]; then
        echo "$JOBID  $label  $model_path" >> "$JOBID_FILE"
    else
        echo "(no jobid)  $label  $model_path" >> "$JOBID_FILE"
    fi
}

for v in "${VARIANTS[@]}"; do
    suffix="${SUFFIX_BY_VAR[$v]}"
    if [ -z "$suffix" ]; then
        echo "ERROR: unknown variant $v (use 3, 4, or 7)" >&2
        continue
    fi
    model_path=$(find_latest_epoch "$suffix")
    if [ -z "$model_path" ]; then
        echo ""
        echo "[variant $v] NO MODEL DIR FOUND for suffix '${suffix}'"
        echo "(skip - no model dir)  variant=$v  base=$BASE_MODEL" >> "$JOBID_FILE"
        continue
    fi

    epoch=$(basename "$model_path" | grep -oE 'epoch[0-9]+' | head -1)
    echo ""
    echo ">>> variant=$v base=$BASE_MODEL ${epoch}"
    echo "    path: $model_path"

    case "$v" in
        3)
            submit_eval "v3.New ${epoch} self+base" "$model_path" "--self-typcorr"
            submit_eval "v3.New ${epoch} neg+base"  "$model_path" "--neg-typcorr"
            ;;
        4)
            submit_eval "v4.New+selfTC ${epoch} self+base" "$model_path" "--self-typcorr"
            ;;
        7)
            submit_eval "v7.New+negTC ${epoch} neg+base" "$model_path" "--neg-typcorr"
            ;;
    esac
done

echo ""
echo "========================================"
echo "Done. Jobid log:"
cat "$JOBID_FILE"
