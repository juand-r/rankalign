#!/bin/bash
#
# Evaluate 4 base models on all test tasks using eval_by_claude.py --self-typicality.
#
# Usage:
#   ./run_base_model_evals.sh                  # Run all models
#   ./run_base_model_evals.sh gemma-2b         # Run only Gemma-2-2b
#   ./run_base_model_evals.sh qwen llama       # Run Qwen and Llama
#
# Models run 2 at a time (parallel) to maximize GPU throughput without saturating it.
# Small models (2b) pair with large models (9b); medium models (7b, 8b) pair together.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
source ../.venv/bin/activate

LOGDIR="../logs/base_evals"
mkdir -p "$LOGDIR"

# ============================================================================
# MODELS
# ============================================================================
declare -A MODELS
MODELS[gemma-2b]="google/gemma-2-2b"
MODELS[gemma-9b]="google/gemma-2-9b"
MODELS[llama-8b]="meta-llama/Llama-3.1-8B"
MODELS[qwen-7b]="Qwen/Qwen2.5-7B"

# ============================================================================
# TASKS — extracted from task registry
# ============================================================================

HYPERNYM_TASKS=(
    hypernym-bananas hypernym-bazookas hypernym-cabinets hypernym-cars
    hypernym-chairs hypernym-crows hypernym-diapers hypernym-dogs
    hypernym-dolls hypernym-ducklings hypernym-elephants hypernym-guns
    hypernym-hammers hypernym-helmets hypernym-jackets hypernym-kayaks
    hypernym-kites "hypernym-magnifying glasses" hypernym-mirrors
    hypernym-nuts hypernym-olives hypernym-oysters hypernym-penguins
    hypernym-puppies "hypernym-rocking horses" hypernym-scallions
    hypernym-spatulas hypernym-spinach hypernym-strollers hypernym-swords
    hypernym-turkeys hypernym-wagons
)

PLAUSIBLEQA_TASKS=($(python -c "
import sys; sys.path.insert(0, '../src')
import tasks
from task_registry import get_all_task_names
names = get_all_task_names([])
for n in sorted(names):
    if n.startswith('plausibleqa-') and 'train' not in n and n != 'plausibleqa-all':
        print(n)
" 2>/dev/null))

AMBIGQA_TASKS=(
    ambigqa-american ambigqa-danube ambigqa-executed ambigqa-gives
    ambigqa-harry ambigqa-involved ambigqa-jack ambigqa-plays
    ambigqa-received ambigqa-sang ambigqa-soccer ambigqa-used
    ambigqa-voice ambigqa-winter ambigqa-won ambigqa-world ambigqa-year
)

IFEVAL_TASKS=($(python -c "
import sys; sys.path.insert(0, '../src')
import tasks
from task_registry import get_all_task_names
names = get_all_task_names([])
for n in sorted(names):
    if n.startswith('ifeval-prompt_'):
        print(n)
" 2>/dev/null))

# ============================================================================
# EVAL FUNCTION
# ============================================================================

run_eval() {
    local MODEL_KEY=$1
    local MODEL_PATH=${MODELS[$MODEL_KEY]}
    local TASK=$2
    local LOG_FILE="$LOGDIR/eval_${MODEL_KEY}_${TASK// /_}.log"

    # Skip if scores file already exists for this model+task
    # Filename pattern: scores_self-v6-{model_path_with_underscores}_{task}_test_*
    local MODEL_SLUG=$(echo "$MODEL_PATH" | sed 's|/|_|g')
    local TASK_SLUG="${TASK// /_}"
    local EXISTING=$(ls ../outputs/scores_self-v6-${MODEL_SLUG}_${TASK_SLUG}_test_*log-odds*.csv 2>/dev/null | head -1)
    if [ -n "$EXISTING" ]; then
        echo "  [SKIP] $MODEL_KEY / $TASK — scores file exists"
        return 0
    fi

    echo "  [RUN] $MODEL_KEY / $TASK"
    python eval_by_claude.py \
        --model "$MODEL_PATH" \
        --task "$TASK" \
        --split_type random \
        --disc-shots few \
        --gen-shots zero \
        --validator-log-odds \
        --self-typicality \
        --save-scores-csv \
        > "$LOG_FILE" 2>&1

    local EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "  [FAIL] $MODEL_KEY / $TASK (exit $EXIT_CODE) — see $LOG_FILE"
    else
        echo "  [DONE] $MODEL_KEY / $TASK"
    fi
    return $EXIT_CODE
}

run_model() {
    local MODEL_KEY=$1
    local MODEL_PATH=${MODELS[$MODEL_KEY]}

    echo ""
    echo "========================================"
    echo "MODEL: $MODEL_KEY ($MODEL_PATH)"
    echo "========================================"

    local TOTAL=0
    local FAILED=0

    echo "--- Hypernym tasks (${#HYPERNYM_TASKS[@]}) ---"
    for TASK in "${HYPERNYM_TASKS[@]}"; do
        run_eval "$MODEL_KEY" "$TASK" || ((FAILED++))
        ((TOTAL++))
    done

    echo "--- PlausibleQA tasks (${#PLAUSIBLEQA_TASKS[@]}) ---"
    for TASK in "${PLAUSIBLEQA_TASKS[@]}"; do
        run_eval "$MODEL_KEY" "$TASK" || ((FAILED++))
        ((TOTAL++))
    done

    echo "--- AmbigQA tasks (${#AMBIGQA_TASKS[@]}) ---"
    for TASK in "${AMBIGQA_TASKS[@]}"; do
        run_eval "$MODEL_KEY" "$TASK" || ((FAILED++))
        ((TOTAL++))
    done

    echo "--- IFEval tasks (${#IFEVAL_TASKS[@]}) ---"
    for TASK in "${IFEVAL_TASKS[@]}"; do
        run_eval "$MODEL_KEY" "$TASK" || ((FAILED++))
        ((TOTAL++))
    done

    echo ""
    echo "[$MODEL_KEY] Completed: $((TOTAL - FAILED))/$TOTAL succeeded, $FAILED failed"
}

# ============================================================================
# MAIN — parse args and run
# ============================================================================

# Determine which models to run
if [ $# -eq 0 ]; then
    SELECTED_MODELS=(gemma-2b gemma-9b llama-8b qwen-7b)
else
    SELECTED_MODELS=("$@")
fi

echo "=========================================="
echo "BASE MODEL EVALUATION"
echo "Models: ${SELECTED_MODELS[*]}"
echo "Tasks per model: ${#HYPERNYM_TASKS[@]} hypernym + ${#PLAUSIBLEQA_TASKS[@]} plausibleqa + ${#AMBIGQA_TASKS[@]} ambigqa + ${#IFEVAL_TASKS[@]} ifeval"
echo "=========================================="

# Run models — 2 at a time for throughput
# Pair small (2b) with large (9b), medium with medium
RUNNING_PIDS=()

for MODEL_KEY in "${SELECTED_MODELS[@]}"; do
    run_model "$MODEL_KEY" &
    RUNNING_PIDS+=($!)

    # Run at most 2 in parallel
    if [ ${#RUNNING_PIDS[@]} -ge 2 ]; then
        wait "${RUNNING_PIDS[0]}"
        RUNNING_PIDS=("${RUNNING_PIDS[@]:1}")
    fi
done

# Wait for remaining
for PID in "${RUNNING_PIDS[@]}"; do
    wait "$PID"
done

echo ""
echo "=========================================="
echo "ALL EVALUATIONS COMPLETE"
echo "=========================================="
