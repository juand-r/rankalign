#!/bin/bash
# Schedule persona-v0 trained-model eval to auto-fire after training jobs finish.
#
# Submits two CPU-only wrapper slurm jobs, each with --dependency=afterany on
# the corresponding training job set:
#
#   wrapper-9b -> waits on overnight/persona_v0_train_jobids_gemma-2-9b-it.txt
#                 then runs: scripts/run_eval_persona_v0_trained.sh google/gemma-2-9b-it
#   wrapper-2b -> waits on overnight/persona_v0_train_jobids_gemma-2-2b-it.txt
#                 then runs: scripts/run_eval_persona_v0_trained.sh google/gemma-2-2b-it
#
# Each wrapper is tiny (1 CPU, 4G, 30min). Its only job is to call the eval
# launcher, which submits 12 separate GPU eval jobs. afterany (not afterok)
# means a failed training variant won't block its base's eval batch -- the
# eval launcher already skips missing model dirs.
#
# Usage:
#   bash scripts/schedule_evals_after_training.sh
#
# Env:
#   DRY_RUN=1   - print sbatch commands but don't submit
#
# Output:
#   overnight/persona_v0_eval_wrapper_jobids.txt -- the two wrapper job IDs.

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OVERNIGHT_DIR="$REPO_ROOT/overnight"
LOG_DIR="$HOME/logs"
mkdir -p "$LOG_DIR"

WRAPPER_JOBID_FILE="$OVERNIGHT_DIR/persona_v0_eval_wrapper_jobids.txt"
: > "$WRAPPER_JOBID_FILE"

submit_wrapper() {
    local base_model="$1"
    local train_jobid_file="$2"
    local label="$3"

    if [ ! -s "$train_jobid_file" ]; then
        echo "ERROR: training jobid file is missing or empty: $train_jobid_file" >&2
        return 1
    fi

    # Extract jobids (first whitespace-separated column), drop any "(no jobid captured)" rows.
    local jids
    jids=$(awk '{print $1}' "$train_jobid_file" | grep -E '^[0-9]+$' | paste -sd: -)
    if [ -z "$jids" ]; then
        echo "ERROR: no numeric jobids parsed from $train_jobid_file" >&2
        return 1
    fi
    local n_deps
    n_deps=$(echo "$jids" | tr ':' '\n' | wc -l)

    echo ""
    echo "=== Wrapper for $label ==="
    echo "  base_model:        $base_model"
    echo "  train jobids file: $train_jobid_file"
    echo "  num dependencies:  $n_deps  (afterany)"
    echo "  dep string (head): $(echo "$jids" | cut -c1-100)..."

    # Use cd $REPO_ROOT && so the eval launcher's relative path defaults
    # (../models, etc.) resolve correctly.
    local cmd="cd '$REPO_ROOT' && bash scripts/run_eval_persona_v0_trained.sh '$base_model'"

    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "  [DRY_RUN] would submit:"
        echo "    sbatch --dependency=afterany:$jids \\"
        echo "           --partition=allnodes --cpus-per-task=1 --mem=4G --time=00:30:00 \\"
        echo "           --output=$LOG_DIR/%j.out --error=$LOG_DIR/%j.err \\"
        echo "           --wrap=\"$cmd\""
        return 0
    fi

    local OUT
    OUT=$(sbatch \
        --dependency=afterany:"$jids" \
        --partition=allnodes \
        --cpus-per-task=1 \
        --mem=4G \
        --time=00:30:00 \
        --job-name="persona_v0_eval_wrap_${label}" \
        --output="$LOG_DIR/%j.out" \
        --error="$LOG_DIR/%j.err" \
        --wrap="$cmd" 2>&1)
    echo "$OUT"
    local WRAP_JOBID
    WRAP_JOBID=$(echo "$OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+$' | head -1)
    if [ -n "$WRAP_JOBID" ]; then
        echo "$WRAP_JOBID  $label  base=$base_model  deps=$n_deps" >> "$WRAPPER_JOBID_FILE"
    else
        echo "(no jobid captured)  $label  base=$base_model  deps=$n_deps" >> "$WRAPPER_JOBID_FILE"
    fi
}

echo "========================================"
echo "Persona-v0 eval auto-fire scheduler"
echo "  REPO_ROOT: $REPO_ROOT"
echo "  Wrapper jobid log: $WRAPPER_JOBID_FILE"
echo "========================================"

submit_wrapper "google/gemma-2-9b-it" "$OVERNIGHT_DIR/persona_v0_train_jobids_gemma-2-9b-it.txt" "9b-it"
submit_wrapper "google/gemma-2-2b-it" "$OVERNIGHT_DIR/persona_v0_train_jobids_gemma-2-2b-it.txt" "2b-it"

echo ""
echo "========================================"
echo "Wrapper jobs submitted:"
cat "$WRAPPER_JOBID_FILE"
echo ""
echo "Each wrapper will fire its eval launcher when its training set finishes"
echo "(afterany dependency). Then the eval launcher submits 12 GPU eval jobs"
echo "(per base model = 24 GPU jobs total)."
echo ""
echo "Check wrapper status: squeue -u \$USER --name=persona_v0_eval_wrap_*"
echo "Check eval jobs after they fire: squeue -u \$USER"
