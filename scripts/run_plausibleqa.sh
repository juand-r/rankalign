#!/bin/bash
set -euo pipefail

# Train and evaluate models on the plausibleqa task (same layout as run_ifeval_concat.sh).
#
# Spec format (colon-separated): NAME:PREF:NLLV:NLLG:VALLOGODDS:LENORM
# Typicality is controlled globally via --tc (self-typicality).
#
# Evaluates each variant on the explicit plausibleqa test task list from
# scripts/semi_supervised_eval_runs.sh (avoids globbing other plausibleqa-* tasks).
#
# Usage:
#   ./run_plausibleqa.sh <NUM_GPUS> [--model MODEL] [--task TASK] [--tc]
#     [--semi-mode none|labelonly|semi] [--semi-ratio RATIO]
#     [--variant NAME|all] [--train-only|--eval-only]
# Variants: base, sft, pref_only, all_terms, all

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source /lusr/opt/miniconda/bin/activate /datastor2/jocelyn/.conda/nlp

export HF_HOME=/datastor2/jocelyn/.cache/huggingface
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

export RANKALIGN_RUN_CACHE=/datastor2/jocelyn/.cache/rankalign-run
mkdir -p \
    "$HF_HOME" \
    "$HF_HUB_CACHE" \
    "$HF_DATASETS_CACHE" \
    "$TRANSFORMERS_CACHE" \
    "$RANKALIGN_RUN_CACHE/tmp" \
    "$RANKALIGN_RUN_CACHE/xdg_cache" \
    "$RANKALIGN_RUN_CACHE/torch_extensions" \
    "$RANKALIGN_RUN_CACHE/triton" \
    "$RANKALIGN_RUN_CACHE/torchinductor" \
    "$RANKALIGN_RUN_CACHE/numba" \
    "$RANKALIGN_RUN_CACHE/matplotlib"

export TMPDIR="$RANKALIGN_RUN_CACHE/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export XDG_CACHE_HOME="$RANKALIGN_RUN_CACHE/xdg_cache"
export TORCH_EXTENSIONS_DIR="$RANKALIGN_RUN_CACHE/torch_extensions"
export TRITON_CACHE_DIR="$RANKALIGN_RUN_CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR="$RANKALIGN_RUN_CACHE/torchinductor"
export NUMBA_CACHE_DIR="$RANKALIGN_RUN_CACHE/numba"
export MPLCONFIGDIR="$RANKALIGN_RUN_CACHE/matplotlib"
export PIP_CACHE_DIR=/datastor2/jocelyn/.cache/pip
mkdir -p "$PIP_CACHE_DIR"

NUM_GPUS_INPUT=${1:-}
MODE="both"
VARIANT_FILTER="all"
MODEL="google/gemma-2-9b-it"
TASK_CONCAT="plausibleqa"
USE_TC=0
SEMI_MODE="none"
SEMI_RATIO="0.1"
shift || true

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --task)
            TASK_CONCAT="$2"
            shift 2
            ;;
        --tc)
            USE_TC=1
            shift
            ;;
        --variant)
            VARIANT_FILTER="$2"
            shift 2
            ;;
        --semi-mode)
            SEMI_MODE="$2"
            shift 2
            ;;
        --semi-ratio)
            SEMI_RATIO="$2"
            shift 2
            ;;
        --train-only)
            MODE="train"
            shift
            ;;
        --eval-only)
            MODE="eval"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

TC_TAG=$([ "$USE_TC" = 1 ] && echo self || echo plain)

VALID_VARIANTS="all base sft pref_only all_terms"
if [[ ! " $VALID_VARIANTS " =~ " $VARIANT_FILTER " ]]; then
    echo "Invalid --variant: $VARIANT_FILTER (must be one of: $VALID_VARIANTS)"
    exit 1
fi

if [ -z "$NUM_GPUS_INPUT" ]; then
    echo "Usage: $0 <NUM_GPUS> [--model MODEL] [--task TASK] [--tc] [--semi-mode none|labelonly|semi] [--semi-ratio RATIO] [--variant NAME|all] [--train-only|--eval-only]"
    echo "  NUM_GPUS: number of GPUs to use, assumed IDs 0..N-1 (e.g., 4 -> 0,1,2,3)"
    echo "  --model: HF model id (default: google/gemma-2-9b-it)"
    echo "  --task: training task name (default: plausibleqa)"
    echo "  --tc: enable self-typicality for train/eval (plain if omitted)"
    echo "  --semi-mode: none | labelonly | semi (default: none)"
    echo "  --semi-ratio: labeled prompt ratio in [0,1] (default: 0.1)"
    echo "  --variant: $VALID_VARIANTS"
    echo "  --train-only: run only training"
    echo "  --eval-only: run only evaluation"
    exit 1
fi

if [[ "$SEMI_MODE" != "none" && "$SEMI_MODE" != "labelonly" && "$SEMI_MODE" != "semi" ]]; then
    echo "--semi-mode must be one of: none, labelonly, semi (got: $SEMI_MODE)"
    exit 1
fi
if ! [[ "$SEMI_RATIO" =~ ^0(\.[0-9]+)?$|^1(\.0+)?$ ]]; then
    echo "--semi-ratio expects a ratio in [0, 1] (got: $SEMI_RATIO)"
    exit 1
fi

if ! [[ "$NUM_GPUS_INPUT" =~ ^[0-9]+$ ]] || [ "$NUM_GPUS_INPUT" -le 0 ]; then
    echo "NUM_GPUS must be a positive integer (got: $NUM_GPUS_INPUT)"
    exit 1
fi

NUM_GPUS=$NUM_GPUS_INPUT
GPUS=()
for ((gpu_id = 0; gpu_id < NUM_GPUS; gpu_id++)); do
    GPUS+=("$gpu_id")
done
GPU_LIST=$(IFS=,; echo "${GPUS[*]}")

GPUS_PER_JOB=2
NUM_WORKERS=$(((NUM_GPUS + GPUS_PER_JOB - 1) / GPUS_PER_JOB))
GPU_PAIRS=()
for ((i = 0; i < NUM_GPUS; i += GPUS_PER_JOB)); do
    pair="${GPUS[i]}"
    for ((j = 1; j < GPUS_PER_JOB && i + j < NUM_GPUS; j++)); do
        pair="${pair},${GPUS[i+j]}"
    done
    GPU_PAIRS+=("$pair")
done
TRAIN_GPU="${GPU_PAIRS[0]}"

NUM_EPOCHS=3
TOTAL_SAMPLES=5110
DELTA=0.15
ALPHA=1.0
USE_LORA=1
if [[ "${MODEL,,}" == *"2b"* ]]; then
    USE_LORA=0
fi

mkdir -p logs

# scripts/semi_supervised_eval_runs.sh TASKS= (explicit test split)
EVAL_TASKS_STR="plausibleqa-nq_1114 plausibleqa-nq_1324 plausibleqa-nq_1328 plausibleqa-nq_1369 plausibleqa-nq_1394 plausibleqa-nq_1438 plausibleqa-nq_1663 plausibleqa-nq_2031 plausibleqa-nq_207 plausibleqa-nq_2174 plausibleqa-nq_2281 plausibleqa-nq_2421 plausibleqa-nq_2436 plausibleqa-nq_2535 plausibleqa-nq_2622 plausibleqa-nq_2637 plausibleqa-nq_2759 plausibleqa-nq_2824 plausibleqa-nq_2856 plausibleqa-nq_2867 plausibleqa-nq_2876 plausibleqa-nq_3004 plausibleqa-nq_3015 plausibleqa-nq_3068 plausibleqa-nq_3099 plausibleqa-nq_3127 plausibleqa-nq_3137 plausibleqa-nq_316 plausibleqa-nq_3276 plausibleqa-nq_54 plausibleqa-nq_562 plausibleqa-nq_709 plausibleqa-nq_958 plausibleqa-trivia_1655 plausibleqa-trivia_2984 plausibleqa-trivia_3035 plausibleqa-trivia_3043 plausibleqa-trivia_3180 plausibleqa-trivia_3245 plausibleqa-trivia_3433 plausibleqa-trivia_3492 plausibleqa-trivia_3599 plausibleqa-trivia_4009 plausibleqa-trivia_4234 plausibleqa-trivia_4489 plausibleqa-trivia_4697 plausibleqa-trivia_5003 plausibleqa-trivia_560 plausibleqa-trivia_5675 plausibleqa-trivia_6317 plausibleqa-trivia_6777 plausibleqa-trivia_7272 plausibleqa-trivia_7579 plausibleqa-trivia_9589 plausibleqa-webq_1000 plausibleqa-webq_1046 plausibleqa-webq_1086 plausibleqa-webq_1097 plausibleqa-webq_1163 plausibleqa-webq_1187 plausibleqa-webq_1278 plausibleqa-webq_1307 plausibleqa-webq_1310 plausibleqa-webq_1338 plausibleqa-webq_134 plausibleqa-webq_1383 plausibleqa-webq_141 plausibleqa-webq_1421 plausibleqa-webq_1442 plausibleqa-webq_1476 plausibleqa-webq_1498 plausibleqa-webq_15 plausibleqa-webq_1584 plausibleqa-webq_1613 plausibleqa-webq_1668 plausibleqa-webq_1714 plausibleqa-webq_1723 plausibleqa-webq_1836 plausibleqa-webq_1972 plausibleqa-webq_212 plausibleqa-webq_299 plausibleqa-webq_342 plausibleqa-webq_373 plausibleqa-webq_428 plausibleqa-webq_435 plausibleqa-webq_520 plausibleqa-webq_611 plausibleqa-webq_650 plausibleqa-webq_669 plausibleqa-webq_672 plausibleqa-webq_713 plausibleqa-webq_744 plausibleqa-webq_749 plausibleqa-webq_760 plausibleqa-webq_77 plausibleqa-webq_803 plausibleqa-webq_84 plausibleqa-webq_88 plausibleqa-webq_882 plausibleqa-webq_898"
read -ra EVAL_TASKS <<< "$EVAL_TASKS_STR"

if [ ${#EVAL_TASKS[@]} -eq 0 ]; then
    echo "No plausibleqa eval tasks configured"
    exit 1
fi

echo "EVAL_TASKS: ${EVAL_TASKS[@]}"
echo "Typicality: ${TC_TAG}"
echo "Semi mode: ${SEMI_MODE} (ratio=${SEMI_RATIO})"
echo "LoRA: $([ "$USE_LORA" = "1" ] && echo enabled || echo disabled)"

run_train_variant() {
    local NAME=$1
    local PREF_W=$2
    local NLLV_W=$3
    local NLLG_W=$4
    local USE_VALIDATOR_LOG_ODDS=${5:-0}
    local USE_LENORM=${6:-0}
    local GPU_PAIR=${7:-$TRAIN_GPU}
    local GPU_LOG=$(echo "$GPU_PAIR" | tr ',' '-')
    local LOG_FILE="logs/train_${TASK_CONCAT}_${NAME}_${TC_TAG}_gpu${GPU_LOG}.log"
    local tc_args=()
    local semi_args=()
    local lora_args=()
    if [ "$USE_TC" = "1" ]; then
        tc_args+=(--self-typicality)
    fi
    if [ "$SEMI_MODE" = "labelonly" ]; then
        semi_args+=(--labeled-only "$SEMI_RATIO")
    elif [ "$SEMI_MODE" = "semi" ]; then
        semi_args+=(--semi-supervised "$SEMI_RATIO")
    fi
    if [ "$USE_LORA" = "1" ]; then
        lora_args+=(--lora)
    fi

    echo "[GPUs $GPU_PAIR] Training ${NAME} ${TC_TAG} (log: $LOG_FILE)"
    {
        echo "========================================"
        echo "Task: $TASK_CONCAT | Variant: $NAME | typicality=$TC_TAG"
        echo "pref=$PREF_W nllv=$NLLV_W nllg=$NLLG_W validator_log_odds=$USE_VALIDATOR_LOG_ODDS"
        echo "semi_mode=$SEMI_MODE ratio=$SEMI_RATIO"
        echo "========================================"

        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="$GPU_PAIR" python ranking_loss_ref.py \
            --model "$MODEL" \
            --num_epochs "$NUM_EPOCHS" \
            --task "$TASK_CONCAT" \
            --train_g_or_d g \
            --nll_validator_weight "$NLLV_W" \
            --nll_generator_weight "$NLLG_W" \
            --preference_loss_weight "$PREF_W" \
            --all \
            --delta "$DELTA" \
            --split_type random \
            --alpha "$ALPHA" \
            --total_samples "$TOTAL_SAMPLES" \
            --save_steps 1 \
            --force-same-x \
            --no-wandb \
            --disc-shots few \
            --no-upload-hf \
            $([ "$USE_LENORM" = "1" ] && echo "--length-normalize") \
            "${lora_args[@]}" \
            "${tc_args[@]}" \
            "${semi_args[@]}" \
            $([ "$USE_VALIDATOR_LOG_ODDS" = "1" ] && echo "--validator-log-odds")
    } >> "$LOG_FILE" 2>&1
}

build_model_dir() {
    local PREF_W=$1
    local NLLV_W=$2
    local NLLG_W=$3
    local USE_VALLOGODDS=${4:-0}
    local USE_LENORM=${5:-0}
    local SEMI_MODE_LOCAL=${6:-none}
    local SEMI_RATIO_LOCAL=${7:-0.1}

    local model_tag="${MODEL//\//--}"
    local typcorr_str=""
    local lenorm_str=""
    local pref_str=""
    local nllv_str=""
    local nllg_str=""
    local vallogodds_str=""
    local semi_str=""
    local LAST_EPOCH=$((NUM_EPOCHS - 1))

    if [ "$USE_TC" = "1" ]; then
        typcorr_str="--tc-self"
    fi
    if [ "$USE_LENORM" = "1" ]; then
        lenorm_str="--lenorm"
    fi
    if [ "$PREF_W" != "1" ]; then
        pref_str="--pref${PREF_W}"
    fi
    if [ "$NLLV_W" != "0" ]; then
        if [ "$NLLV_W" = "1" ] || [ "$NLLV_W" = "1.0" ]; then
            nllv_str="--nllv1.0"
        else
            nllv_str="--nllv${NLLV_W}"
        fi
    fi
    if [ "$NLLG_W" != "0" ]; then
        if [ "$NLLG_W" = "1" ] || [ "$NLLG_W" = "1.0" ]; then
            nllg_str="--nllg1.0"
        else
            nllg_str="--nllg${NLLG_W}"
        fi
    fi
    if [ "$USE_VALLOGODDS" = "1" ]; then
        vallogodds_str="--vallogodds"
    fi
    if [ "$SEMI_MODE_LOCAL" = "labelonly" ]; then
        semi_str="--labelonly${SEMI_RATIO_LOCAL}"
    elif [ "$SEMI_MODE_LOCAL" = "semi" ]; then
        semi_str="--semi${SEMI_RATIO_LOCAL}"
    fi
    local force_same_x_str="--force-same-x"

    local lora_suffix=""
    if [ "$USE_LORA" = "1" ]; then
        lora_suffix="_merged"
    fi
    echo "../models/v6-${model_tag}-delta${DELTA}-epoch${LAST_EPOCH}--${TASK_CONCAT}-all--d2g--random--alpha${ALPHA}${typcorr_str}${lenorm_str}--full-completion${pref_str}${nllv_str}${nllg_str}${force_same_x_str}${vallogodds_str}${semi_str}${lora_suffix}"
}

run_eval_variant() {
    local NAME=$1
    local PREF_W=$2
    local NLLV_W=$3
    local NLLG_W=$4
    local USE_VALLOGODDS=$5
    local USE_LENORM=$6
    local TASK=$7
    local GPU=$8

    local MODEL_DIR
    local LOG_FILE="logs/eval_${TASK}_${NAME}_${TC_TAG}_gpu${GPU}.log"
    {
        echo "========================================"
        echo "Task: $TASK | Variant: $NAME | GPU: $GPU | typicality=$TC_TAG"
        echo "========================================"
    } >> "$LOG_FILE"

    if [ "$NAME" = "base" ]; then
        MODEL_DIR="$MODEL"
    else
        MODEL_DIR=$(build_model_dir "$PREF_W" "$NLLV_W" "$NLLG_W" "$USE_VALLOGODDS" "$USE_LENORM" "$SEMI_MODE" "$SEMI_RATIO")
        if [ ! -d "$MODEL_DIR" ]; then
            echo "  [SKIP] Model not found: $MODEL_DIR" >> "$LOG_FILE"
            return
        fi
    fi

    CUDA_VISIBLE_DEVICES="$GPU" python eval_by_claude.py \
        --model "$MODEL_DIR" \
        --task "$TASK" \
        --split_type random \
        --validator-log-odds \
        --save-scores-csv \
        --disc-shots few \
        --length-normalize \
        --self-typicality \
        >> "$LOG_FILE" 2>&1
}

ALL_TRAIN_SPECS=(
    "sft:0.0:1:1:0:0"
    "pref_only:1:0:0:0:0"
    "all_terms:1:1:1:1:0"
)

TRAIN_SPECS=()
train_spec_selected() {
    local name=$1
    case "$VARIANT_FILTER" in
        base)
            return 1
            ;;
        *)
            [[ "$name" == "$VARIANT_FILTER" ]] && return 0
            return 1
            ;;
    esac
}

if [ "$VARIANT_FILTER" = "all" ]; then
    TRAIN_SPECS=("${ALL_TRAIN_SPECS[@]}")
else
    for spec in "${ALL_TRAIN_SPECS[@]}"; do
        name="${spec%%:*}"
        if train_spec_selected "$name"; then
            TRAIN_SPECS+=("$spec")
        fi
    done
fi

run_train_worker() {
    local worker_idx=$1
    local gpu_pair="${GPU_PAIRS[$worker_idx]}"
    local idx=0
    for spec in "${TRAIN_SPECS[@]}"; do
        if [ $((idx % NUM_WORKERS)) -eq "$worker_idx" ]; then
            IFS=':' read -r NAME PREF_W NLLV_W NLLG_W USE_VALIDATOR_LOG_ODDS USE_LENORM < <(printf '%s\n' "$spec")
            run_train_variant "$NAME" "$PREF_W" "$NLLV_W" "$NLLG_W" "$USE_VALIDATOR_LOG_ODDS" "$USE_LENORM" "$gpu_pair"
        fi
        idx=$((idx + 1))
    done
}

if [ "$MODE" = "train" ] || [ "$MODE" = "both" ]; then
    if [ ${#TRAIN_SPECS[@]} -gt 0 ]; then
        echo "========================================"
        echo "Training variants on $TASK_CONCAT (GPUs: $GPU_LIST, $GPUS_PER_JOB GPUs per job, $NUM_WORKERS parallel workers)"
        echo "========================================"

        for worker_idx in $(seq 0 $((NUM_WORKERS - 1))); do
            run_train_worker "$worker_idx" &
        done
        wait
        echo "All training runs completed."
    else
        echo "No training for variant=$VARIANT_FILTER (base has no training)."
    fi
fi

ALL_VARIANTS=(
    "base:0:0:0:0:0"
    "sft:0.0:1:1:0:0"
    "pref_only:1:0:0:0:0"
    "all_terms:1:1:1:1:0"
)

VARIANTS=()
eval_variant_selected() {
    local name=$1
    [[ "$name" == "$VARIANT_FILTER" ]] && return 0
    return 1
}

if [ "$VARIANT_FILTER" = "all" ]; then
    VARIANTS=("${ALL_VARIANTS[@]}")
else
    for v in "${ALL_VARIANTS[@]}"; do
        name="${v%%:*}"
        if eval_variant_selected "$name"; then
            VARIANTS+=("$v")
        fi
    done
fi

EVAL_JOBS=()
for TASK in "${EVAL_TASKS[@]}"; do
    for V in "${VARIANTS[@]}"; do
        EVAL_JOBS+=("${TASK}|${V}")
    done
done

echo "Prepared ${#EVAL_JOBS[@]} eval jobs across ${#VARIANTS[@]} variant(s) and ${#EVAL_TASKS[@]} task(s)."
if [ ${#VARIANTS[@]} -eq 0 ]; then
    echo "No eval variants selected for --variant=$VARIANT_FILTER"
    exit 1
fi
if [ ${#EVAL_JOBS[@]} -eq 0 ]; then
    echo "No eval jobs were generated."
    exit 1
fi

run_eval_worker() {
    local gpu_idx=$1
    local gpu_id=${GPUS[$gpu_idx]}
    local idx=0
    local assigned=0
    echo "[worker $gpu_idx gpu $gpu_id] starting"
    for job in "${EVAL_JOBS[@]}"; do
        if [ $((idx % NUM_GPUS)) -eq "$gpu_idx" ]; then
            TASK="${job%%|*}"
            V="${job#*|}"
            IFS=':' read -r NAME PREF_W NLLV_W NLLG_W USE_VALLOGODDS USE_LENORM < <(printf '%s\n' "$V")
            run_eval_variant "$NAME" "$PREF_W" "$NLLV_W" "$NLLG_W" "$USE_VALLOGODDS" "$USE_LENORM" "$TASK" "$gpu_id"
            assigned=$((assigned + 1))
        fi
        idx=$((idx + 1))
    done
    echo "[worker $gpu_idx gpu $gpu_id] completed $assigned job(s)"
}

if [ "$MODE" = "eval" ] || [ "$MODE" = "both" ]; then
    {
        echo "========================================"
        echo "Evaluating variants on ${TASK_CONCAT} + per-example test tasks (one task per GPU at a time)"
        echo "========================================"
        echo "  Eval logs:  $SCRIPT_DIR/logs/eval_<TASK>_<VARIANT>_gpu<N>.log"
        echo "  CSV files:  $SCRIPT_DIR/../outputs/scores_*.csv"
    }
    mkdir -p "$SCRIPT_DIR/../outputs"

    eval_pids=()
    for gpu_idx in $(seq 0 $((NUM_GPUS - 1))); do
        run_eval_worker "$gpu_idx" &
        eval_pids+=("$!")
    done
    for pid in "${eval_pids[@]}"; do
        wait "$pid"
    done
    echo "All evaluations completed."
fi
