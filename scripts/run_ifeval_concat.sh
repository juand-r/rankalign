#!/bin/bash
set -euo pipefail

# Train and evaluate models on the ifeval-concat task.
# Current: all-terms w/o val log odds (with and without tc)
#   - all_terms_no_vallogodds (pref=1, nllv=1, nllg=1, no tc, no vallogodds)
#   - all_terms_no_vallogodds_tc (pref=1, nllv=1, nllg=1, tc, no vallogodds)
# Commented out: SFT, pref_only, all_terms (with vallogodds), all_terms_tc (with vallogodds)
#
# Evaluates each variant on:
#   - each ifeval-prompt_* task
#
# Usage:
#   ./run_ifeval_concat.sh <GPU_LIST> [--variant NAME|all] [--train-only|--eval-only]
# Variants: base, sft, pref_only, all_terms, all_terms_tc, pref_tc, all_terms_tc_lenorm (or all)
# Example:
#   ./run_ifeval_concat.sh 0,1,2,3
#   ./run_ifeval_concat.sh 0,1 --variant sft
#   ./run_ifeval_concat.sh 0 --variant all_terms_tc --eval-only

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Avoid Hugging Face Xet downloader "Background writer channel closed" on NFS/Slurm
export HF_HUB_DISABLE_XET=1
# Put hub cache (model/dataset downloads) on local disk to avoid "Disk quota exceeded".
# Leave HF_HOME default so ~/.cache/huggingface (and login token) is still used.
HF_HUB_CACHE_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/.cache/huggingface/hub"
export HF_HUB_CACHE="$HF_HUB_CACHE_ROOT"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE_ROOT"
mkdir -p "$HF_HUB_CACHE_ROOT"

GPU_LIST=${1:-}
MODE="both"
VARIANT_FILTER="all"
shift || true

while [[ $# -gt 0 ]]; do
    case $1 in
        --variant)
            VARIANT_FILTER="$2"
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

VALID_VARIANTS="base sft pref_only all_terms all_terms_tc pref_tc all_terms_tc_lenorm all"
if [[ ! " $VALID_VARIANTS " =~ " $VARIANT_FILTER " ]]; then
    echo "Invalid --variant: $VARIANT_FILTER (must be: $VALID_VARIANTS)"
    exit 1
fi

if [ -z "$GPU_LIST" ]; then
    echo "Usage: $0 <GPU_LIST> [--variant NAME|all] [--train-only|--eval-only]"
    echo "  GPU_LIST: comma-separated GPU IDs (e.g., 0,1,2,3)"
    echo "  --variant: base, sft, pref_only, all_terms, all_terms_tc, pref_tc, all_terms_tc_lenorm, or all (default)"
    echo "  --train-only: run only training"
    echo "  --eval-only: run only evaluation"
    exit 1
fi

IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}
GPUS_PER_JOB=2
NUM_WORKERS=$((NUM_GPUS / GPUS_PER_JOB))
# Build GPU pairs for model parallelism: "0,1", "2,3", etc.
GPU_PAIRS=()
for ((i = 0; i < NUM_GPUS; i += GPUS_PER_JOB)); do
    pair="${GPUS[i]}"
    for ((j = 1; j < GPUS_PER_JOB && i + j < NUM_GPUS; j++)); do
        pair="${pair},${GPUS[i+j]}"
    done
    GPU_PAIRS+=("$pair")
done
TRAIN_GPU="${GPU_PAIRS[0]}"

MODEL="google/gemma-2-9b-it"
NUM_EPOCHS=2
TOTAL_SAMPLES=5110
DELTA=0.15
ALPHA=1.0
TASK_CONCAT="ifeval-concat"

DATA_DIR="/datastor2/jocelyn/rankalign/data/fixed-prompts-ifeval"

mkdir -p logs
STATUS_LOG="$SCRIPT_DIR/logs/job_status.log"

TASKS=()
for f in "$DATA_DIR"/gpt_ifeval_results_*.jsonl; do
    [ -e "$f" ] || continue
    base=$(basename "$f")
    name="${base#gpt_ifeval_results_}"
    name="${name%.jsonl}"
    TASKS+=("ifeval-$name")
done

if [ ${#TASKS[@]} -eq 0 ]; then
    echo "No ifeval prompt datasets found in $DATA_DIR"
    exit 1
fi

EVAL_TASKS=("${TASKS[@]}")

echo "EVAL_TASKS: ${EVAL_TASKS[@]}" >> "$STATUS_LOG"

run_train_variant() {
    local NAME=$1
    local PREF_W=$2
    local NLLV_W=$3
    local NLLG_W=$4
    local USE_TC=$5
    local USE_VALIDATOR_LOG_ODDS=${6:-0}
    local USE_LENORM=${7:-0}
    local GPU_PAIR=${8:-$TRAIN_GPU}
    local GPU_LOG=$(echo "$GPU_PAIR" | tr ',' '-')
    local LOG_FILE="logs/train_${TASK_CONCAT}_${NAME}_gpu${GPU_LOG}.log"

    echo "[GPUs $GPU_PAIR] Training ${NAME} (log: $LOG_FILE)" >> "$STATUS_LOG"
    {
        echo "========================================"
        echo "Task: $TASK_CONCAT | Variant: $NAME"
        echo "pref=$PREF_W nllv=$NLLV_W nllg=$NLLG_W tc=$USE_TC validator_log_odds=$USE_VALIDATOR_LOG_ODDS"
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
            --lora \
            --force-same-x \
            --no-wandb \
            $([ "$USE_LENORM" = "1" ] && echo "--length-normalize") \
            $([ "$USE_TC" = "1" ] && echo "--typicality-correction") \
            $([ "$USE_VALIDATOR_LOG_ODDS" = "1" ] && echo "--validator-log-odds")
    } >> "$LOG_FILE" 2>&1
}

build_model_dir() {
    local PREF_W=$1
    local NLLV_W=$2
    local NLLG_W=$3
    local USE_TC=$4
    local USE_VALLOGODDS=${5:-0}
    local USE_LENORM=${6:-0}

    local model_tag="${MODEL//\//--}"
    local typcorr_str=""
    local lenorm_str=""
    local pref_str=""
    local nllv_str=""
    local nllg_str=""
    local vallogodds_str=""
    # Training saves with 0-based epoch index; NUM_EPOCHS=2 -> last save at epoch 1
    local LAST_EPOCH=$((NUM_EPOCHS - 1))

    if [ "$USE_TC" = "1" ]; then
        typcorr_str="--tc-online"
    fi
    if [ "$USE_LENORM" = "1" ]; then
        lenorm_str="--lenorm"
    fi
    if [ "$PREF_W" != "1" ]; then
        pref_str="--pref${PREF_W}"
    fi
    # Training script (Python) formats weights as float, so 1 -> "1.0" in dir name
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
    # Training uses --force-same-x; model save path includes it, so we must too for eval to find the dir
    local force_same_x_str="--force-same-x"

    echo "../models/v6-${model_tag}-delta${DELTA}-epoch${LAST_EPOCH}--${TASK_CONCAT}-all--d2g--random--alpha${ALPHA}${typcorr_str}${lenorm_str}--full-completion${pref_str}${nllv_str}${nllg_str}${force_same_x_str}${vallogodds_str}_merged"
}

run_eval_variant() {
    local NAME=$1
    local PREF_W=$2
    local NLLV_W=$3
    local NLLG_W=$4
    local USE_TC=$5
    local USE_VALLOGODDS=$6
    local USE_LENORM=$7
    local TASK=$8
    local GPU=$9

    local MODEL_DIR
    local LOG_FILE="logs/eval_${TASK}_${NAME}_gpu${GPU}.log"

    if [ "$NAME" = "base" ]; then
        MODEL_DIR="$MODEL"
    else
        MODEL_DIR=$(build_model_dir "$PREF_W" "$NLLV_W" "$NLLG_W" "$USE_TC" "$USE_VALLOGODDS" "$USE_LENORM")
        if [ ! -d "$MODEL_DIR" ]; then
            echo "  [SKIP] Model not found: $MODEL_DIR" >> "$LOG_FILE"
            return
        fi
    fi

    CUDA_VISIBLE_DEVICES="$GPU" python eval.py \
        --model "$MODEL_DIR" \
        --task "$TASK" \
        --split_type random \
        --validator-log-odds \
        --save-scores-csv \
        --disc-shots zero \
        --length-normalize \
        --typicality-correction \
        >> "$LOG_FILE" 2>&1
}

# Training task specs: NAME:PREF_W:NLLV_W:NLLG_W:USE_TC:USE_VALLOGODDS:USE_LENORM
ALL_TRAIN_SPECS=(
    "sft:0.0:1:1:0:0:0"
    "pref_only:1:0:0:0:0:0"
    "all_terms:1:1:1:0:1:0"
    "all_terms_tc:1:1:1:1:1:0"
    "pref_tc:1:0:0:1:0:0"
    "all_terms_tc_lenorm:1:1:1:1:1:1"
)

# Filter by variant (base has no training)
TRAIN_SPECS=()
if [ "$VARIANT_FILTER" = "all" ]; then
    TRAIN_SPECS=("${ALL_TRAIN_SPECS[@]}")
elif [ "$VARIANT_FILTER" != "base" ]; then
    for spec in "${ALL_TRAIN_SPECS[@]}"; do
        name="${spec%%:*}"
        if [ "$name" = "$VARIANT_FILTER" ]; then
            TRAIN_SPECS+=("$spec")
            break
        fi
    done
fi


# Run one GPU worker: execute all tasks whose index % NUM_WORKERS == worker_idx, sequentially.
# Each worker uses GPUS_PER_JOB GPUs for model parallelism.
run_train_worker() {
    local worker_idx=$1
    local gpu_pair="${GPU_PAIRS[$worker_idx]}"
    local idx=0
    for spec in "${TRAIN_SPECS[@]}"; do
        if [ $((idx % NUM_WORKERS)) -eq "$worker_idx" ]; then
            IFS=':' read -r NAME PREF_W NLLV_W NLLG_W USE_TC USE_VALLOGODDS USE_LENORM <<< "$spec"
            run_train_variant "$NAME" "$PREF_W" "$NLLV_W" "$NLLG_W" "$USE_TC" "$USE_VALLOGODDS" "$USE_LENORM" "$gpu_pair"
        fi
        idx=$((idx + 1))
    done
}

if [ "$MODE" = "train" ] || [ "$MODE" = "both" ]; then
    if [ ${#TRAIN_SPECS[@]} -gt 0 ]; then
        {
            echo "========================================"
            echo "Training variants on $TASK_CONCAT (GPUs: $GPU_LIST, $GPUS_PER_JOB GPUs per job, $NUM_WORKERS parallel workers)"
            echo "========================================"
        } >> "$STATUS_LOG"

        for worker_idx in $(seq 0 $((NUM_WORKERS - 1))); do
            run_train_worker "$worker_idx" &
        done
        wait
        echo "All training runs completed." >> "$STATUS_LOG"
    else
        echo "No training for variant=$VARIANT_FILTER (base has no training)." >> "$STATUS_LOG"
    fi
fi

# Eval variant specs: NAME:PREF_W:NLLV_W:NLLG_W:USE_TC:USE_VALLOGODDS:USE_LENORM
ALL_VARIANTS=(
    "base:0:0:0:0:0:0"
    "sft:0.0:1:1:0:0:0"
    "pref_only:1:0:0:0:0:0"
    "all_terms:1:1:1:0:1:0"
    "all_terms_tc:1:1:1:1:1:0"
    "pref_tc:1:0:0:1:0:0"
    "all_terms_tc_lenorm:1:1:1:1:1:1"
)

# Filter by variant
VARIANTS=()
if [ "$VARIANT_FILTER" = "all" ]; then
    VARIANTS=("${ALL_VARIANTS[@]}")
else
    for v in "${ALL_VARIANTS[@]}"; do
        name="${v%%:*}"
        if [ "$name" = "$VARIANT_FILTER" ]; then
            VARIANTS+=("$v")
            break
        fi
    done
fi

# Build flat list of eval jobs: each element is "TASK|NAME:PREF_W:NLLV_W:NLLG_W:USE_TC:USE_VALLOGODDS"
EVAL_JOBS=()
for TASK in "${EVAL_TASKS[@]}"; do
    for V in "${VARIANTS[@]}"; do
        EVAL_JOBS+=("${TASK}|${V}")
    done
done

# Run one GPU worker: execute all eval jobs whose index % NUM_GPUS == gpu_idx, sequentially.
run_eval_worker() {
    local gpu_idx=$1
    local gpu_id=${GPUS[$gpu_idx]}
    local idx=0
    for job in "${EVAL_JOBS[@]}"; do
        if [ $((idx % NUM_GPUS)) -eq "$gpu_idx" ]; then
            TASK="${job%%|*}"
            V="${job#*|}"
            IFS=':' read -r NAME PREF_W NLLV_W NLLG_W USE_TC USE_VALLOGODDS USE_LENORM <<< "$V"
            run_eval_variant "$NAME" "$PREF_W" "$NLLV_W" "$NLLG_W" "$USE_TC" "$USE_VALLOGODDS" "$USE_LENORM" "$TASK" "$gpu_id"
        fi
        idx=$((idx + 1))
    done
}

if [ "$MODE" = "eval" ] || [ "$MODE" = "both" ]; then
    {
        echo "========================================"
        echo "Evaluating variants on concat + per-prompt tasks (one task per GPU at a time)"
        echo "========================================"
        echo "  Eval logs:  $SCRIPT_DIR/logs/eval_<TASK>_<VARIANT>_gpu<N>.log"
        echo "  CSV files:  $SCRIPT_DIR/../outputs/scores_*.csv"
    } >> "$STATUS_LOG"
    mkdir -p "$SCRIPT_DIR/../outputs"

    for gpu_idx in $(seq 0 $((NUM_GPUS - 1))); do
        run_eval_worker "$gpu_idx" &
    done
    wait
    echo "All evaluations completed." >> "$STATUS_LOG"
fi
