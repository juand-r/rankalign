#!/bin/bash
#
# Submit negtc + selftc evals on all gsm8k-v2 per-problem test tasks (100 problems).
# 3 base models × 2 TC modes × 8 task groups = 48 Slurm jobs.
#
# Mirrors the v1.1 launcher — uses sbatch directly, auto-discovers task IDs
# from the data dir, no external `run` helper.
#
# Usage (on mll):
#   bash scripts/run_eval_gsm8k_v2_base_models.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RANKALIGN_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$RANKALIGN_DIR/slurm_logs/v2_eval"
mkdir -p "$LOG_DIR"

# Auto-discover v2 test task IDs from the data dir
V2_TEST_DIR="$(dirname "$SCRIPT_DIR")/data/gsm8k/v2/test"
if [ ! -d "$V2_TEST_DIR" ]; then
    echo "ERROR: v2 test dir not found at $V2_TEST_DIR" >&2
    exit 1
fi
ALL_TIDS=( $(ls "$V2_TEST_DIR"/gsm8k_test_*.csv 2>/dev/null | xargs -n1 basename | sed 's/\.csv$//' | sort) )
N_TIDS=${#ALL_TIDS[@]}
echo "Discovered $N_TIDS v2 test task IDs" >&2

# Round-robin into 8 groups (12-13 each for 100 problems)
TASKS_G1="" TASKS_G2="" TASKS_G3="" TASKS_G4="" TASKS_G5="" TASKS_G6="" TASKS_G7="" TASKS_G8=""
for i in "${!ALL_TIDS[@]}"; do
    g=$(( (i % 8) + 1 ))
    var="TASKS_G${g}"
    eval "${var}=\"\${${var}}\${${var}:+ }gsm8k-v2-${ALL_TIDS[i]}\""
done

MODELS_BIG="google/gemma-2-9b-it"
MODELS_SMALL="google/gemma-2-2b-it google/gemma-2-2b"

submit() {
    local hours="$1"; shift
    local gpus="$1"; shift
    local model="$1"; shift
    local tc="$1"; shift
    local gidx="$1"; shift
    local tasks="$*"

    local model_short
    model_short=$(basename "$model" | sed 's/--/_/g')
    local tc_short="${tc#--}"
    local jobname="v2_${model_short}_${tc_short}_g${gidx}"

    sbatch \
        --job-name="$jobname" \
        --partition=allnodes \
        --gres=gpu:nvidia-A40:${gpus} \
        --cpus-per-task=4 \
        --mem=32G \
        --time=${hours}:00:00 \
        --output="${LOG_DIR}/${jobname}-%j.out" \
        --wrap="bash -c 'set -eo pipefail; source ~/venvs/venv_lexcons/bin/activate; cd ${SCRIPT_DIR}/..; bash scripts/run_eval_semi.sh \"$model\" $tc --log-odds -- $tasks'"
}

N_SUBMITTED=0
for G in 1 2 3 4 5 6 7 8; do
    TASKS_VAR="TASKS_G${G}"
    TASKS="${!TASKS_VAR}"

    for MODEL in $MODELS_BIG; do
        for TC in --self-typcorr --neg-typcorr; do
            submit 2 1 "$MODEL" "$TC" "$G" $TASKS
            N_SUBMITTED=$((N_SUBMITTED + 1))
        done
    done

    for MODEL in $MODELS_SMALL; do
        for TC in --self-typcorr --neg-typcorr; do
            submit 1 1 "$MODEL" "$TC" "$G" $TASKS
            N_SUBMITTED=$((N_SUBMITTED + 1))
        done
    done
done

echo ""
echo "Submitted $N_SUBMITTED jobs (3 models × 2 TC × 8 groups = 48)."
echo "Logs in $LOG_DIR/"
echo "Score CSVs land in $RANKALIGN_DIR/outputs/ with prefix self- or neg-."
echo "Reruns skip already-completed tasks."
