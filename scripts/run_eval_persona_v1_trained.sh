#!/bin/bash
# Post-train eval for persona-v1 trained models.
#
# Sibling of run_eval_persona_v0_trained.sh. Differences vs v0:
#   - Training task name in model paths: persona-v1-all (was persona-v0-all)
#   - 6 test tasks (DROPS subscribes-to-moral-nihilism + believes-life-has-no-meaning).
#     See docs/datasets/persona_v1_notes.md.
#   - Trained models in this batch were trained with --disc-shots zero, so eval
#     also uses --disc-shots-zero (matched).
#
# All eval jobs use --base-typicality with the *original instruct model* as the
# typicality reference (matches the offline-TC training convention; see
# docs/IMPORTANT-RESEARCH-PLAN.md §3 / "Methodology fix"):
#
#   self+base: --self-typcorr --base-typcorr --base-model <BASE>   -> scores prefix: basetyp-
#   neg+base:  --neg-typcorr  --base-typcorr --base-model <BASE>   -> scores prefix: basetypneg-
#
# TC-flavor selection per variant (matched eval per research-plan §3):
#   #1 SFT-lo,    #2 RankAlign,         #3 New+fsx          -> BOTH self+base and neg+base
#   #4 New+fsx+selfTC, #5 RankAlign+fsx+selfTC, #6 RankAlign+selfTC -> self+base ONLY
#   #7 New+fsx+negTC,  #8 RankAlign+fsx+negTC,  #9 RankAlign+negTC  -> neg+base ONLY
#
# So per BASE: 3 variants × 2 + 6 variants × 1 = 12 jobs.
#
# Usage:
#   bash scripts/run_eval_persona_v1_trained.sh [BASE_MODEL]
#
#   BASE_MODEL: HuggingFace id of the base model used during training
#               (default: google/gemma-2-9b-it).
#
# Examples:
#   bash scripts/run_eval_persona_v1_trained.sh
#   bash scripts/run_eval_persona_v1_trained.sh google/gemma-2-2b-it
#
# Env:
#   BASE_MODEL  - default base model when no positional argument given
#   HOURS       - Slurm walltime per job (default: 2)
#   CPUS        - CPUs per task (default: 6)
#   MEM         - memory per job (default: 60G)
#   MODELS_DIR  - absolute or relative path to the trained-models dir
#                 (default: <repo_root>/models, resolved from the script's
#                 own location so it works regardless of cwd).
#   EPOCH       - which epoch checkpoint to eval (default: 2)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BASE_MODEL="${1:-${BASE_MODEL:-google/gemma-2-9b-it}}"
HOURS="${HOURS:-2}"
CPUS="${CPUS:-6}"
MEM="${MEM:-60G}"
MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models}"
EPOCH="${EPOCH:-2}"

TASKS=(
    persona-v1-psychopathy
    persona-v1-machiavellianism
    persona-v1-narcissism
    persona-v1-desire-to-create-allies
    persona-v1-interest-in-music
    persona-v1-interest-in-science
)

# Common eval flags. --log-odds matches the project convention.
# DISC_SHOTS env var controls disc-shots at eval time:
#   'zero' (default) -> --disc-shots-zero, matches v1 9b-it/2b-it training.
#   'few'            -> omit the flag (run_eval_semi.sh defaults to few-shot),
#                       matches the v1 gemma-2-2b training launched
#                       2026-05-21 with DISC_SHOTS=auto (= few for base models).
DISC_SHOTS="${DISC_SHOTS:-zero}"
if [ "$DISC_SHOTS" = "few" ] || [ "$DISC_SHOTS" = "auto" ]; then
    EVAL_COMMON="--log-odds"
else
    EVAL_COMMON="--log-odds --disc-shots-zero"
fi

# Variant suffixes (without trailing _merged; LoRA models append _merged automatically below).
declare -A SUFFIXES=(
    ["1.SFT-lo"]="--full-completion--pref0.0--nllv1.0--nllg1.0--labelonly0.1"
    ["2.RankAlign"]="--full-completion--semi0.1"
    ["3.New+fsx"]="--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1"
    ["4.New+fsx+selfTC"]="--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1"
    ["5.RankAlign+fsx+selfTC"]="--tc-self--full-completion--force-same-x--semi0.1"
    ["6.RankAlign+selfTC"]="--tc-self--full-completion--semi0.1"
    ["7.New+fsx+negTC"]="--tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1"
    ["8.RankAlign+fsx+negTC"]="--tc-neg--full-completion--force-same-x--semi0.1"
    ["9.RankAlign+negTC"]="--tc-neg--full-completion--semi0.1"
)

declare -A TC_POLICY=(
    ["1.SFT-lo"]="both"
    ["2.RankAlign"]="both"
    ["3.New+fsx"]="both"
    ["4.New+fsx+selfTC"]="self"
    ["5.RankAlign+fsx+selfTC"]="self"
    ["6.RankAlign+selfTC"]="self"
    ["7.New+fsx+negTC"]="neg"
    ["8.RankAlign+fsx+negTC"]="neg"
    ["9.RankAlign+negTC"]="neg"
)

VARIANT_ORDER=(
    "1.SFT-lo"
    "2.RankAlign"
    "3.New+fsx"
    "4.New+fsx+selfTC"
    "5.RankAlign+fsx+selfTC"
    "6.RankAlign+selfTC"
    "7.New+fsx+negTC"
    "8.RankAlign+fsx+negTC"
    "9.RankAlign+negTC"
)

# Decide whether models are LoRA-merged based on the base model name. Mirrors the
# logic in run_train_semi.sh.
USE_LORA=1
if [[ "$BASE_MODEL" == *"-2b"* || "$BASE_MODEL" == *"-2b-"* ]]; then
    USE_LORA=0
fi
MERGED_SUFFIX=""
[[ "$USE_LORA" -eq 1 ]] && MERGED_SUFFIX="_merged"

BASE_REPL=$(echo "$BASE_MODEL" | sed 's|/|--|g')
PATH_PREFIX="${MODELS_DIR}/v6-${BASE_REPL}-delta0.15-epoch${EPOCH}--persona-v1-all--d2g--random--alpha1.0"

OVERNIGHT_DIR="$(dirname "$0")/../overnight"
mkdir -p "$OVERNIGHT_DIR"
MODEL_TAG=$(basename "$BASE_MODEL" | sed 's|/|_|g; s|--|_|g')
DISC_TAG="_disc-${DISC_SHOTS}"
JOBID_FILE="$OVERNIGHT_DIR/persona_v1_eval_trained_jobids_${MODEL_TAG}${DISC_TAG}.txt"
: > "$JOBID_FILE"

echo "========================================"
echo "Persona-v1 trained-model eval launcher"
echo "  BASE_MODEL: $BASE_MODEL"
echo "  EPOCH:      $EPOCH"
echo "  USE_LORA:   $USE_LORA  (merged suffix: '${MERGED_SUFFIX}')"
echo "  HOURS:      $HOURS  (each of 12 jobs = 3 variants × 2 TC flavors + 6 variants × 1 matched flavor)"
echo "  CPUS / MEM: $CPUS / $MEM"
echo "  Tasks:      ${#TASKS[@]}  (${TASKS[*]})"
echo "  EVAL_COMMON: $EVAL_COMMON"
echo "  DISC_SHOTS: $DISC_SHOTS"
echo "  Jobid log:  $JOBID_FILE"
echo "========================================"

submit() {
    local label="$1"; shift
    local model_path="$1"; shift
    local tc_flag="$1"; shift
    local tc_label="$1"; shift

    if [ ! -d "$model_path" ]; then
        echo ""
        echo "[$label / $tc_label] SKIP - model dir not found: $model_path"
        echo "(no jobid - SKIP)  $label  $tc_label  $model_path" >> "$JOBID_FILE"
        return
    fi

    echo ""
    echo ">>> [$label / $tc_label] $model_path"
    OUT=$(run 1 "$HOURS" --cpu "$CPUS" --mem "$MEM" \
        scripts/run_eval_semi.sh "$model_path" \
        $tc_flag --base-typcorr --base-model "$BASE_MODEL" \
        $EVAL_COMMON \
        -- "${TASKS[@]}" 2>&1) || true
    echo "$OUT"
    JOBID=$(echo "$OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+$' | head -1)
    if [ -n "$JOBID" ]; then
        echo "$JOBID  $label  $tc_label  $model_path" >> "$JOBID_FILE"
    else
        echo "(no jobid captured)  $label  $tc_label  $model_path" >> "$JOBID_FILE"
    fi
}

for variant in "${VARIANT_ORDER[@]}"; do
    suffix="${SUFFIXES[$variant]}"
    policy="${TC_POLICY[$variant]}"
    model_path="${PATH_PREFIX}${suffix}${MERGED_SUFFIX}"
    case "$policy" in
        both)
            submit "$variant" "$model_path" "--self-typcorr" "self+base"
            submit "$variant" "$model_path" "--neg-typcorr"  "neg+base"
            ;;
        self)
            submit "$variant" "$model_path" "--self-typcorr" "self+base"
            ;;
        neg)
            submit "$variant" "$model_path" "--neg-typcorr"  "neg+base"
            ;;
        *)
            echo "ERROR: unknown TC_POLICY '$policy' for variant $variant" >&2
            exit 1
            ;;
    esac
done

echo ""
echo "========================================"
echo "All 12 submissions attempted. Jobid log:"
cat "$JOBID_FILE"
echo ""
echo "Tail logs with: tail -f /datastor2/jdr/logs/<JOBID>.out"
echo "Check progress with: squeue -u \$USER"
