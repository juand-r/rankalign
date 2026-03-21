#!/bin/bash
#
# Evaluate base models on all test tasks using eval_by_claude.py --self-typicality.
#
# Usage:
#   ./run_base_model_evals.sh gemma-2b         # Run only Gemma-2-2b
#   ./run_base_model_evals.sh qwen-7b          # Run only Qwen
#   ./run_base_model_evals.sh gemma-2b qwen-7b # Run both (sequentially)
#
# Available model keys: gemma-2b, gemma-9b, llama-8b, qwen-7b

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
# TASKS — hardcoded to avoid stdout pollution from task registration
# ============================================================================

HYPERNYM_TASKS=(
    hypernym-bananas hypernym-bazookas hypernym-cabinets hypernym-cars
    hypernym-chairs hypernym-crows hypernym-diapers hypernym-dogs
    hypernym-dolls hypernym-ducklings hypernym-elephants hypernym-guns
    hypernym-hammers hypernym-helmets hypernym-jackets hypernym-kayaks
    hypernym-kites hypernym-mirrors
)
# NOTE: removed hypernyms without v2 grammar-corrected data:
# magnifying glasses, nuts, olives, oysters, penguins, puppies,
# rocking horses, scallions, spatulas, spinach, strollers, swords, turkeys, wagons

PLAUSIBLEQA_TASKS=(
    plausibleqa-nq_1114 plausibleqa-nq_1324 plausibleqa-nq_1328 plausibleqa-nq_1369
    plausibleqa-nq_1394 plausibleqa-nq_1438 plausibleqa-nq_1663 plausibleqa-nq_2031
    plausibleqa-nq_207 plausibleqa-nq_2174 plausibleqa-nq_2281 plausibleqa-nq_2421
    plausibleqa-nq_2436 plausibleqa-nq_2535 plausibleqa-nq_2622 plausibleqa-nq_2637
    plausibleqa-nq_2759 plausibleqa-nq_2824 plausibleqa-nq_2856 plausibleqa-nq_2867
    plausibleqa-nq_2876 plausibleqa-nq_3004 plausibleqa-nq_3015 plausibleqa-nq_3068
    plausibleqa-nq_3099 plausibleqa-nq_3127 plausibleqa-nq_3137 plausibleqa-nq_316
    plausibleqa-nq_3276 plausibleqa-nq_54 plausibleqa-nq_562 plausibleqa-nq_709
    plausibleqa-nq_958
    plausibleqa-trivia_1655 plausibleqa-trivia_2984 plausibleqa-trivia_3035
    plausibleqa-trivia_3043 plausibleqa-trivia_3180 plausibleqa-trivia_3245
    plausibleqa-trivia_3433 plausibleqa-trivia_3492 plausibleqa-trivia_3599
    plausibleqa-trivia_4009 plausibleqa-trivia_4234 plausibleqa-trivia_4489
    plausibleqa-trivia_4697 plausibleqa-trivia_5003 plausibleqa-trivia_560
    plausibleqa-trivia_5675 plausibleqa-trivia_6317 plausibleqa-trivia_6777
    plausibleqa-trivia_7272 plausibleqa-trivia_7579 plausibleqa-trivia_9589
    plausibleqa-webq_1000 plausibleqa-webq_1046 plausibleqa-webq_1086
    plausibleqa-webq_1097 plausibleqa-webq_1163 plausibleqa-webq_1187
    plausibleqa-webq_1278 plausibleqa-webq_1307 plausibleqa-webq_1310
    plausibleqa-webq_1338 plausibleqa-webq_134 plausibleqa-webq_1383
    plausibleqa-webq_141 plausibleqa-webq_1421 plausibleqa-webq_1442
    plausibleqa-webq_1476 plausibleqa-webq_1498 plausibleqa-webq_15
    plausibleqa-webq_1584 plausibleqa-webq_1613 plausibleqa-webq_1668
    plausibleqa-webq_1714 plausibleqa-webq_1723 plausibleqa-webq_1836
    plausibleqa-webq_1972 plausibleqa-webq_212 plausibleqa-webq_299
    plausibleqa-webq_342 plausibleqa-webq_373 plausibleqa-webq_428
    plausibleqa-webq_435 plausibleqa-webq_520 plausibleqa-webq_611
    plausibleqa-webq_650 plausibleqa-webq_669 plausibleqa-webq_672
    plausibleqa-webq_713 plausibleqa-webq_744 plausibleqa-webq_749
    plausibleqa-webq_760 plausibleqa-webq_77 plausibleqa-webq_803
    plausibleqa-webq_84 plausibleqa-webq_88 plausibleqa-webq_882
    plausibleqa-webq_898
)

AMBIGQA_TASKS=(
    ambigqa-american ambigqa-danube ambigqa-executed ambigqa-gives
    ambigqa-harry ambigqa-involved ambigqa-jack ambigqa-plays
    ambigqa-received ambigqa-sang ambigqa-soccer ambigqa-used
    ambigqa-voice ambigqa-winter ambigqa-won ambigqa-world ambigqa-year
)

IFEVAL_TASKS=(
    ifeval-prompt_1 ifeval-prompt_10 ifeval-prompt_100 ifeval-prompt_102
    ifeval-prompt_103 ifeval-prompt_104 ifeval-prompt_105 ifeval-prompt_106
    ifeval-prompt_107 ifeval-prompt_108 ifeval-prompt_109 ifeval-prompt_11
    ifeval-prompt_12 ifeval-prompt_13 ifeval-prompt_15 ifeval-prompt_16
    ifeval-prompt_17 ifeval-prompt_18 ifeval-prompt_19 ifeval-prompt_2
    ifeval-prompt_20 ifeval-prompt_21 ifeval-prompt_22 ifeval-prompt_23
    ifeval-prompt_24 ifeval-prompt_25 ifeval-prompt_26 ifeval-prompt_27
    ifeval-prompt_28 ifeval-prompt_29 ifeval-prompt_3 ifeval-prompt_30
    ifeval-prompt_32 ifeval-prompt_33 ifeval-prompt_34 ifeval-prompt_35
    ifeval-prompt_36 ifeval-prompt_37 ifeval-prompt_38 ifeval-prompt_39
    ifeval-prompt_4 ifeval-prompt_40 ifeval-prompt_41 ifeval-prompt_42
    ifeval-prompt_43 ifeval-prompt_44 ifeval-prompt_45 ifeval-prompt_46
    ifeval-prompt_47 ifeval-prompt_48 ifeval-prompt_49 ifeval-prompt_5
    ifeval-prompt_50 ifeval-prompt_51 ifeval-prompt_52 ifeval-prompt_53
    ifeval-prompt_54 ifeval-prompt_55 ifeval-prompt_56 ifeval-prompt_57
    ifeval-prompt_58 ifeval-prompt_59 ifeval-prompt_6 ifeval-prompt_61
    ifeval-prompt_63 ifeval-prompt_64 ifeval-prompt_65 ifeval-prompt_66
    ifeval-prompt_67 ifeval-prompt_68 ifeval-prompt_7 ifeval-prompt_70
    ifeval-prompt_72 ifeval-prompt_73 ifeval-prompt_74 ifeval-prompt_75
    ifeval-prompt_76 ifeval-prompt_77 ifeval-prompt_78 ifeval-prompt_79
    ifeval-prompt_8 ifeval-prompt_80 ifeval-prompt_82 ifeval-prompt_83
    ifeval-prompt_84 ifeval-prompt_85 ifeval-prompt_87 ifeval-prompt_88
    ifeval-prompt_89 ifeval-prompt_9 ifeval-prompt_90 ifeval-prompt_91
    ifeval-prompt_92 ifeval-prompt_93 ifeval-prompt_94 ifeval-prompt_95
    ifeval-prompt_96 ifeval-prompt_97 ifeval-prompt_98 ifeval-prompt_99
)

# ============================================================================
# EVAL FUNCTION
# ============================================================================

run_eval() {
    local MODEL_KEY=$1
    local MODEL_PATH=${MODELS[$MODEL_KEY]}
    local TASK=$2
    local LOG_FILE="$LOGDIR/eval_${MODEL_KEY}_${TASK// /_}.log"

    # Skip if scores file already exists for this model+task
    local MODEL_SLUG=$(echo "$MODEL_PATH" | sed 's|/|_|g')
    local TASK_SLUG="${TASK// /_}"
    local EXISTING=$(ls ../outputs/scores_self-v6-${MODEL_SLUG}_${TASK_SLUG}_test_*log-odds*.csv 2>/dev/null | head -1)
    if [ -n "$EXISTING" ]; then
        echo "  [SKIP] $MODEL_KEY / $TASK — scores file exists"
        return 0
    fi

    # IFEval uses zero-shot discriminator (few-shot not implemented)
    local DISC_SHOTS="few"
    if [[ "$TASK" == ifeval-* ]]; then
        DISC_SHOTS="zero"
    fi

    echo "  [RUN] $MODEL_KEY / $TASK (disc=$DISC_SHOTS)"
    python eval_by_claude.py \
        --model "$MODEL_PATH" \
        --task "$TASK" \
        --split_type random \
        --disc-shots "$DISC_SHOTS" \
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
    return 0  # Always return 0 so script continues on failure
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
        run_eval "$MODEL_KEY" "$TASK"
        ((TOTAL++))
    done

    echo "--- PlausibleQA tasks (${#PLAUSIBLEQA_TASKS[@]}) ---"
    for TASK in "${PLAUSIBLEQA_TASKS[@]}"; do
        run_eval "$MODEL_KEY" "$TASK"
        ((TOTAL++))
    done

    echo "--- AmbigQA tasks (${#AMBIGQA_TASKS[@]}) ---"
    for TASK in "${AMBIGQA_TASKS[@]}"; do
        run_eval "$MODEL_KEY" "$TASK"
        ((TOTAL++))
    done

    echo "--- IFEval tasks (${#IFEVAL_TASKS[@]}) ---"
    for TASK in "${IFEVAL_TASKS[@]}"; do
        run_eval "$MODEL_KEY" "$TASK"
        ((TOTAL++))
    done

    echo ""
    echo "[$MODEL_KEY] Completed: $TOTAL tasks processed"
}

# ============================================================================
# MAIN
# ============================================================================

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_key> [model_key2 ...]"
    echo "Available: gemma-2b, gemma-9b, llama-8b, qwen-7b"
    exit 1
fi

SELECTED_MODELS=("$@")

echo "=========================================="
echo "BASE MODEL EVALUATION"
echo "Models: ${SELECTED_MODELS[*]}"
echo "Tasks per model: ${#HYPERNYM_TASKS[@]} hypernym + ${#PLAUSIBLEQA_TASKS[@]} plausibleqa + ${#AMBIGQA_TASKS[@]} ambigqa + ${#IFEVAL_TASKS[@]} ifeval"
echo "=========================================="

for MODEL_KEY in "${SELECTED_MODELS[@]}"; do
    run_model "$MODEL_KEY"
done

echo ""
echo "=========================================="
echo "ALL EVALUATIONS COMPLETE"
echo "=========================================="
