#!/bin/bash
# Fan out the membership->rosch eval as 10 short Slurm jobs, one per rosch
# category. Each job runs scripts/run_eval_one_rosch_for_membership.sh.
#
# All 10 rosch categories are evaluated (we want stratified-by-overlap
# reporting, so even high-leakage rosch-bird is included).
#
# This script reads JIDs from overnight/membership_train_jobids.txt and
# submits each eval with --dependency=afterany:<all 9 train JIDs> so they
# start after all training jobs have finished (success or failure).
#
# Submits via `sbatch` directly (NOT via `~/.local/bin/run`) because `run`
# does not pass --dependency through to sbatch.
#
# Usage:
#   bash scripts/launch_membership_to_rosch_evals.sh

set -e

cd "$(dirname "$0")/.."

JIDS_FILE=overnight/membership_train_jobids.txt
if [[ ! -f "$JIDS_FILE" ]]; then
    echo "ERROR: $JIDS_FILE not found. Run scripts/run_train_membership_quickiter.sh first." >&2
    exit 1
fi

TRAIN_JIDS=$(awk '{print $1}' "$JIDS_FILE" | tr '\n' ':' | sed 's/:$//')
if [[ -z "$TRAIN_JIDS" ]]; then
    echo "ERROR: No training JIDs parsed from $JIDS_FILE." >&2
    exit 1
fi

DEP_STR="afterany:$TRAIN_JIDS"

TASKS=(
    rosch-bird
    rosch-carpenters-tool
    rosch-clothing
    rosch-fruit
    rosch-furniture
    rosch-sport
    rosch-toy
    rosch-vegetable
    rosch-vehicle
    rosch-weapon
)

mkdir -p overnight ~/logs
EVAL_JIDS_FILE=overnight/membership_eval_jobids.txt
: > "$EVAL_JIDS_FILE"

echo "Training JIDs: $TRAIN_JIDS"
echo "Dependency: $DEP_STR"
echo ""

for T in "${TASKS[@]}"; do
    echo ">>> Submitting eval: $T"
    out=$(sbatch \
        --partition=allnodes \
        --cpus-per-task=12 \
        --mem=120G \
        --gres=gpu:1 \
        --time=1:00:00 \
        --dependency="$DEP_STR" \
        --output="$HOME/logs/%j.out" \
        --error="$HOME/logs/%j.err" \
        --wrap="PYTHONUNBUFFERED=1 /usr/bin/time -v bash scripts/run_eval_one_rosch_for_membership.sh $T" 2>&1)
    echo "$out"
    jid=$(echo "$out" | grep -oE 'Submitted batch job [0-9]+' | awk '{print $4}')
    if [[ -n "$jid" ]]; then
        echo "$jid  $T" >> "$EVAL_JIDS_FILE"
    fi
done

echo ""
echo "============================================================"
echo "Submitted ${#TASKS[@]} eval jobs (will start after training)."
echo "Eval JIDs / tasks recorded in $EVAL_JIDS_FILE"
echo "============================================================"
cat "$EVAL_JIDS_FILE"
