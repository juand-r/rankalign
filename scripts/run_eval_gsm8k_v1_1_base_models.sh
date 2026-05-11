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
# Task groups (mirrors v1 split — 111 problems into 8 groups of ~14).
# ============================================================================
TASKS_G1="gsm8k-v1.1-gsm8k_test_1000 gsm8k-v1.1-gsm8k_test_1003 gsm8k-v1.1-gsm8k_test_1018 gsm8k-v1.1-gsm8k_test_1021 gsm8k-v1.1-gsm8k_test_1022 gsm8k-v1.1-gsm8k_test_1025 gsm8k-v1.1-gsm8k_test_1027 gsm8k-v1.1-gsm8k_test_1058 gsm8k-v1.1-gsm8k_test_1073 gsm8k-v1.1-gsm8k_test_1080 gsm8k-v1.1-gsm8k_test_1084 gsm8k-v1.1-gsm8k_test_1087 gsm8k-v1.1-gsm8k_test_1092 gsm8k-v1.1-gsm8k_test_1097"

TASKS_G2="gsm8k-v1.1-gsm8k_test_1110 gsm8k-v1.1-gsm8k_test_1113 gsm8k-v1.1-gsm8k_test_1120 gsm8k-v1.1-gsm8k_test_1124 gsm8k-v1.1-gsm8k_test_1146 gsm8k-v1.1-gsm8k_test_115 gsm8k-v1.1-gsm8k_test_1168 gsm8k-v1.1-gsm8k_test_1170 gsm8k-v1.1-gsm8k_test_1185 gsm8k-v1.1-gsm8k_test_1203 gsm8k-v1.1-gsm8k_test_1216 gsm8k-v1.1-gsm8k_test_1226 gsm8k-v1.1-gsm8k_test_124 gsm8k-v1.1-gsm8k_test_1240"

TASKS_G3="gsm8k-v1.1-gsm8k_test_1244 gsm8k-v1.1-gsm8k_test_1252 gsm8k-v1.1-gsm8k_test_1258 gsm8k-v1.1-gsm8k_test_1262 gsm8k-v1.1-gsm8k_test_1273 gsm8k-v1.1-gsm8k_test_1282 gsm8k-v1.1-gsm8k_test_129 gsm8k-v1.1-gsm8k_test_1300 gsm8k-v1.1-gsm8k_test_1311 gsm8k-v1.1-gsm8k_test_1313 gsm8k-v1.1-gsm8k_test_1316 gsm8k-v1.1-gsm8k_test_1317 gsm8k-v1.1-gsm8k_test_145 gsm8k-v1.1-gsm8k_test_16"

TASKS_G4="gsm8k-v1.1-gsm8k_test_166 gsm8k-v1.1-gsm8k_test_168 gsm8k-v1.1-gsm8k_test_193 gsm8k-v1.1-gsm8k_test_205 gsm8k-v1.1-gsm8k_test_218 gsm8k-v1.1-gsm8k_test_227 gsm8k-v1.1-gsm8k_test_252 gsm8k-v1.1-gsm8k_test_257 gsm8k-v1.1-gsm8k_test_270 gsm8k-v1.1-gsm8k_test_282 gsm8k-v1.1-gsm8k_test_296 gsm8k-v1.1-gsm8k_test_303 gsm8k-v1.1-gsm8k_test_307 gsm8k-v1.1-gsm8k_test_321"

TASKS_G5="gsm8k-v1.1-gsm8k_test_360 gsm8k-v1.1-gsm8k_test_363 gsm8k-v1.1-gsm8k_test_386 gsm8k-v1.1-gsm8k_test_387 gsm8k-v1.1-gsm8k_test_390 gsm8k-v1.1-gsm8k_test_4 gsm8k-v1.1-gsm8k_test_427 gsm8k-v1.1-gsm8k_test_431 gsm8k-v1.1-gsm8k_test_459 gsm8k-v1.1-gsm8k_test_462 gsm8k-v1.1-gsm8k_test_466 gsm8k-v1.1-gsm8k_test_485 gsm8k-v1.1-gsm8k_test_538 gsm8k-v1.1-gsm8k_test_541"

TASKS_G6="gsm8k-v1.1-gsm8k_test_547 gsm8k-v1.1-gsm8k_test_554 gsm8k-v1.1-gsm8k_test_574 gsm8k-v1.1-gsm8k_test_609 gsm8k-v1.1-gsm8k_test_622 gsm8k-v1.1-gsm8k_test_633 gsm8k-v1.1-gsm8k_test_650 gsm8k-v1.1-gsm8k_test_656 gsm8k-v1.1-gsm8k_test_660 gsm8k-v1.1-gsm8k_test_677 gsm8k-v1.1-gsm8k_test_680 gsm8k-v1.1-gsm8k_test_697 gsm8k-v1.1-gsm8k_test_701 gsm8k-v1.1-gsm8k_test_707"

TASKS_G7="gsm8k-v1.1-gsm8k_test_714 gsm8k-v1.1-gsm8k_test_742 gsm8k-v1.1-gsm8k_test_761 gsm8k-v1.1-gsm8k_test_784 gsm8k-v1.1-gsm8k_test_798 gsm8k-v1.1-gsm8k_test_807 gsm8k-v1.1-gsm8k_test_809 gsm8k-v1.1-gsm8k_test_821 gsm8k-v1.1-gsm8k_test_839 gsm8k-v1.1-gsm8k_test_866 gsm8k-v1.1-gsm8k_test_878 gsm8k-v1.1-gsm8k_test_882 gsm8k-v1.1-gsm8k_test_885 gsm8k-v1.1-gsm8k_test_903"

TASKS_G8="gsm8k-v1.1-gsm8k_test_912 gsm8k-v1.1-gsm8k_test_916 gsm8k-v1.1-gsm8k_test_918 gsm8k-v1.1-gsm8k_test_922 gsm8k-v1.1-gsm8k_test_924 gsm8k-v1.1-gsm8k_test_927 gsm8k-v1.1-gsm8k_test_934 gsm8k-v1.1-gsm8k_test_939 gsm8k-v1.1-gsm8k_test_956 gsm8k-v1.1-gsm8k_test_966 gsm8k-v1.1-gsm8k_test_971 gsm8k-v1.1-gsm8k_test_988 gsm8k-v1.1-gsm8k_test_996"

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
