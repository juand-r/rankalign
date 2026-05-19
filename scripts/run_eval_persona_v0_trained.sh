#!/bin/bash
# Post-train eval for persona-v0 trained models.
#
# Each of the 9 training variants (see run_train_persona_v0.sh) gets two
# evaluations, both using --base-typicality with the *original instruct
# model* as the typicality reference (matches the offline-TC training
# convention; see docs/IMPORTANT-RESEARCH-PLAN.md §3 / "Methodology fix"):
#
#   1. --self-typcorr --base-typcorr --base-model <BASE>   -> scores prefix: basetyp-
#   2. --neg-typcorr  --base-typcorr --base-model <BASE>   -> scores prefix: basetypneg-
#
# Each eval job runs all 8 persona-v0-<slug> test tasks sequentially via
# run_eval_semi.sh. With --disc-shots-zero --log-odds (matching the eval
# baseline of the persona task and the research-plan eval recipe).
#
# So per BASE: 9 variants × 2 TC flavors = 18 jobs. Each job runs 8 tasks.
#
# Usage:
#   bash scripts/run_eval_persona_v0_trained.sh [BASE_MODEL]
#
#   BASE_MODEL: HuggingFace id of the base model used during training
#               (default: google/gemma-2-9b-it).
#
# Examples:
#   bash scripts/run_eval_persona_v0_trained.sh
#   bash scripts/run_eval_persona_v0_trained.sh google/gemma-2-2b-it
#
# Env:
#   BASE_MODEL  - default base model when no positional argument given
#   HOURS       - Slurm walltime per job (default: 2)
#   CPUS        - CPUs per task (default: 6)
#   MEM         - memory per job (default: 60G)
#   MODELS_DIR  - directory containing trained models (default: ../models, i.e. ../models from scripts/)
#   EPOCH       - which epoch checkpoint to eval (default: 2)

set -e

BASE_MODEL="${1:-${BASE_MODEL:-google/gemma-2-9b-it}}"
HOURS="${HOURS:-2}"
CPUS="${CPUS:-6}"
MEM="${MEM:-60G}"
MODELS_DIR="${MODELS_DIR:-../models}"
EPOCH="${EPOCH:-2}"

TASKS=(
    persona-v0-psychopathy
    persona-v0-machiavellianism
    persona-v0-narcissism
    persona-v0-subscribes-to-moral-nihilism
    persona-v0-believes-life-has-no-meaning
    persona-v0-desire-to-create-allies
    persona-v0-interest-in-music
    persona-v0-interest-in-science
)

# Common eval flags. --log-odds matches the project convention; --disc-shots-zero
# matches the persona-v0 eval baseline (see docs/datasets/persona_v0_notes.md).
EVAL_COMMON="--log-odds --disc-shots-zero"

# Construct expected model directory suffixes per training variant. These mirror
# the save_directory built in scripts/ranking_loss_ref.py (~line 2757).
#
# Common path prefix (everything up to alpha1.0):
#   v6-{base_repl}-delta0.15-epoch{N}--persona-v0-all--d2g--random--alpha1.0
# where base_repl = "google--gemma-2-9b-it" or "google--gemma-2-2b-it".
#
# The suffix per variant (everything after alpha1.0) is what changes:

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

# Order matters for predictable jobid sequence; bash assoc arrays don't preserve order.
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
# logic in run_train_semi.sh (~line 145).
USE_LORA=1
if [[ "$BASE_MODEL" == *"-2b"* || "$BASE_MODEL" == *"-2b-"* ]]; then
    USE_LORA=0
fi
MERGED_SUFFIX=""
[[ "$USE_LORA" -eq 1 ]] && MERGED_SUFFIX="_merged"

BASE_REPL=$(echo "$BASE_MODEL" | sed 's|/|--|g')
PATH_PREFIX="${MODELS_DIR}/v6-${BASE_REPL}-delta0.15-epoch${EPOCH}--persona-v0-all--d2g--random--alpha1.0"

OVERNIGHT_DIR="$(dirname "$0")/../overnight"
mkdir -p "$OVERNIGHT_DIR"
MODEL_TAG=$(basename "$BASE_MODEL" | sed 's|/|_|g; s|--|_|g')
JOBID_FILE="$OVERNIGHT_DIR/persona_v0_eval_trained_jobids_${MODEL_TAG}.txt"
: > "$JOBID_FILE"

echo "========================================"
echo "Persona-v0 trained-model eval launcher"
echo "  BASE_MODEL: $BASE_MODEL"
echo "  EPOCH:      $EPOCH"
echo "  USE_LORA:   $USE_LORA  (merged suffix: '${MERGED_SUFFIX}')"
echo "  HOURS:      $HOURS  (each of 18 jobs = 9 variants × 2 TC flavors)"
echo "  CPUS / MEM: $CPUS / $MEM"
echo "  Tasks:      ${#TASKS[@]}  (${TASKS[*]})"
echo "  EVAL_COMMON: $EVAL_COMMON"
echo "  Jobid log:  $JOBID_FILE"
echo "========================================"

submit() {
    local label="$1"; shift
    local model_path="$1"; shift
    local tc_flag="$1"; shift   # --self-typcorr or --neg-typcorr
    local tc_label="$1"; shift  # "selfTC" / "negTC"

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
    model_path="${PATH_PREFIX}${suffix}${MERGED_SUFFIX}"
    submit "$variant" "$model_path" "--self-typcorr" "self+base"
    submit "$variant" "$model_path" "--neg-typcorr"  "neg+base"
done

echo ""
echo "========================================"
echo "All 18 submissions attempted. Jobid log:"
cat "$JOBID_FILE"
echo ""
echo "Tail logs with: tail -f ~/logs/<JOBID>.out"
echo "Check progress with: squeue -u \$USER"
