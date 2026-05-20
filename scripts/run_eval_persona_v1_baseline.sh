#!/bin/bash
# Persona-v1 baseline evaluation across the gemma-2 model family.
#
# 6 slurm jobs total: 3 models × {neg-typcorr, self-typcorr}, all with
# --disc-shots-zero. Each job runs all 6 persona-v1-<slug> eval tasks
# sequentially via run_eval_semi.sh.
#
# Each --neg-typcorr / --self-typcorr eval pass writes a single scores_*.csv
# per task, capturing all of gen_score, gen_score_typcorr, gen_score_lenorm,
# gen_score_typcorr_lenorm — so raw-vs-TC and lenorm comparisons come for free.
#
# Usage:
#   bash scripts/run_eval_persona_v1_baseline.sh
#
# Captures jobids to overnight/persona_v1_baseline_eval_jobids.txt.

set -e

PERSONAS=(
    persona-v1-psychopathy
    persona-v1-machiavellianism
    persona-v1-narcissism
    persona-v1-desire-to-create-allies
    persona-v1-interest-in-music
    persona-v1-interest-in-science
)

MODELS=(
    google/gemma-2-2b
    google/gemma-2-2b-it
    google/gemma-2-9b-it
)

TC_FLAVORS=(--neg-typcorr --self-typcorr)
# `run` writes Slurm logs to $HOME/logs. /u/jdr is currently quota-limited for
# new log growth, so default to a high-space HOME for job submission.
RUN_HOME="${RUN_HOME:-/datastor2/jdr}"

OVERNIGHT_DIR="$(dirname "$0")/../overnight"
mkdir -p "$OVERNIGHT_DIR"
JOBID_FILE="$OVERNIGHT_DIR/persona_v1_baseline_eval_jobids.txt"
: > "$JOBID_FILE"

echo "Submitting persona-v1 baseline evals: ${#MODELS[@]} models × ${#TC_FLAVORS[@]} TC flavors = $((${#MODELS[@]} * ${#TC_FLAVORS[@]})) jobs"
echo "Tasks per job (${#PERSONAS[@]}): ${PERSONAS[*]}"
echo "RUN_HOME: $RUN_HOME (run logs -> $RUN_HOME/logs)"
echo "Logging jobids to: $JOBID_FILE"
echo "============================================================"

for MODEL in "${MODELS[@]}"; do
    for TC in "${TC_FLAVORS[@]}"; do
        echo ""
        echo ">>> $MODEL  $TC"
        OUT=$(HOME="$RUN_HOME" run 1 2 scripts/run_eval_semi.sh "$MODEL" "$TC" --log-odds --disc-shots-zero -- "${PERSONAS[@]}" 2>&1) || true
        echo "$OUT"
        JOBID=$(echo "$OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+$' | head -1)
        if [ -n "$JOBID" ]; then
            echo "$JOBID  $MODEL  $TC" >> "$JOBID_FILE"
        else
            echo "(no jobid captured)  $MODEL  $TC" >> "$JOBID_FILE"
        fi
    done
done

echo ""
echo "============================================================"
echo "All submissions attempted. Jobid log:"
cat "$JOBID_FILE"
echo ""
echo "Tail logs with: tail -f ~/logs/<JOBID>.out  (or .err)"
echo "Check progress with: squeue -u \$USER"
