#!/bin/bash
# Persona-v0 baseline evaluation across the gemma-2 model family.
#
# 6 slurm jobs total: 3 models × {neg-typcorr, self-typcorr}, all with
# --disc-shots-zero (matches Perez et al. 2022 methodology and the ifeval
# precedent in this repo). Each job runs all 8 persona-v0-<slug> eval tasks
# sequentially via run_eval_semi.sh.
#
# Each --neg-typcorr or --self-typcorr eval pass writes a single scores_*.csv
# per task, capturing all of: gen_score, gen_score_typcorr,
# gen_score_lenorm, gen_score_typcorr_lenorm — so we get the lenorm and
# raw-vs-TC comparisons "for free" without extra runs.
#
# Usage:
#   bash scripts/run_eval_persona_v0_baseline.sh
#
# Captures jobids to overnight/persona_v0_baseline_eval_jobids.txt for later
# tailing of slurm logs.

set -e

PERSONAS=(
    persona-v0-psychopathy
    persona-v0-machiavellianism
    persona-v0-narcissism
    persona-v0-subscribes-to-moral-nihilism
    persona-v0-believes-life-has-no-meaning
    persona-v0-desire-to-create-allies
    persona-v0-interest-in-music
    persona-v0-interest-in-science
)

MODELS=(
    google/gemma-2-2b
    google/gemma-2-2b-it
    google/gemma-2-9b-it
)

TC_FLAVORS=(--neg-typcorr --self-typcorr)

OVERNIGHT_DIR="$(dirname "$0")/../overnight"
mkdir -p "$OVERNIGHT_DIR"
JOBID_FILE="$OVERNIGHT_DIR/persona_v0_baseline_eval_jobids.txt"
: > "$JOBID_FILE"

echo "Submitting persona-v0 baseline evals: ${#MODELS[@]} models × ${#TC_FLAVORS[@]} TC flavors = $((${#MODELS[@]} * ${#TC_FLAVORS[@]})) jobs"
echo "Tasks per job (${#PERSONAS[@]}): ${PERSONAS[*]}"
echo "Logging jobids to: $JOBID_FILE"
echo "============================================================"

for MODEL in "${MODELS[@]}"; do
    for TC in "${TC_FLAVORS[@]}"; do
        echo ""
        echo ">>> $MODEL  $TC"
        # Capture stdout so we can extract the jobid from the run wrapper.
        OUT=$(run 1 2 scripts/run_eval_semi.sh "$MODEL" "$TC" --log-odds --disc-shots-zero -- "${PERSONAS[@]}" 2>&1) || true
        echo "$OUT"
        # Best-effort jobid capture (sbatch usually prints "Submitted batch job <ID>").
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
