#!/bin/bash
#
# Submit negtc + selftc evals on all 111 v1.1 per-problem tasks.
# 3 base models × 2 TC modes × 8 task groups = 48 Slurm jobs.
#
# Walltimes: 2h for 9b-it, 1h for 2b-it / 2b (matched to v1).
# Score CSVs land in private_projects/rankalign/outputs/ with prefix
# self- or neg- depending on TC mode. eval_by_claude.py skips tasks
# whose score file already exists, so reruns are cheap.
#
# Usage (on mll):
#   bash scripts/run_eval_gsm8k_v1_1_base_models.sh
#
# This launcher is SELF-CONTAINED — uses sbatch directly, no external
# `run` helper. Mirrors run_eval_gsm8k_v1_base_models.sh in groupings.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RANKALIGN_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$RANKALIGN_DIR/slurm_logs/v1_1_eval"
mkdir -p "$LOG_DIR"

# ============================================================================
# Task groups — auto-discovered from the actual v1.1 dataset directory so the
# launcher never drifts from the data. (The v1 launcher was hardcoded and
# had 23 stale task IDs that no longer matched the data.)
# Splits into 8 groups round-robin for even sizes.
# ============================================================================
V1_1_DIR="$(dirname "$SCRIPT_DIR")/data/gsm8k/v1.1"
if [ ! -d "$V1_1_DIR" ]; then
    echo "ERROR: v1.1 data dir not found at $V1_1_DIR" >&2
    exit 1
fi
ALL_TIDS=( $(ls "$V1_1_DIR"/gsm8k_test_*.csv 2>/dev/null | xargs -n1 basename | sed 's/\.csv$//' | sort) )
N_TIDS=${#ALL_TIDS[@]}
echo "Discovered $N_TIDS v1.1 task IDs in $V1_1_DIR" >&2

# Round-robin into 8 groups for even sizes
TASKS_G1="" TASKS_G2="" TASKS_G3="" TASKS_G4="" TASKS_G5="" TASKS_G6="" TASKS_G7="" TASKS_G8=""
for i in "${!ALL_TIDS[@]}"; do
    g=$(( (i % 8) + 1 ))
    var="TASKS_G${g}"
    eval "${var}=\"\${${var}}\${${var}:+ }gsm8k-v1.1-${ALL_TIDS[i]}\""
done

MODELS_BIG="google/gemma-2-9b-it"
MODELS_SMALL="google/gemma-2-2b-it google/gemma-2-2b"

submit() {
    # submit <hours> <gpu_count> <model> <tc_flag> <group_idx> <tasks>
    local hours="$1"; shift
    local gpus="$1"; shift
    local model="$1"; shift
    local tc="$1"; shift
    local gidx="$1"; shift
    local tasks="$*"

    local model_short
    model_short=$(basename "$model" | sed 's/--/_/g')
    local tc_short="${tc#--}"
    local jobname="v1.1_${model_short}_${tc_short}_g${gidx}"

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
