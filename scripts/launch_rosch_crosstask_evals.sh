#!/bin/bash
# Fan out the cross-task eval as 8 short Slurm jobs (one per non-train rosch
# category). Each job runs run_eval_rosch_crosstask_one.sh on a single
# rosch-<category> task, evaluating all 10 trained variants + base.
#
# Tasks excluded: rosch-bird and rosch-furniture (the two categories that
# went into the rosch-furniture-and-bird training set).
#
# Walltime: 1 h per job (each job is ~25 evals × ~30-90s = ~15-30 min).
# 1 GPU each. Many small jobs is intentional — better cluster scheduling
# than one fat job.
#
# Usage:
#   bash scripts/launch_rosch_crosstask_evals.sh

set -e

TASKS=(
    rosch-carpenters-tool
    rosch-clothing
    rosch-fruit
    rosch-sport
    rosch-toy
    rosch-vegetable
    rosch-vehicle
    rosch-weapon
)

cd "$(dirname "$0")/.."

mkdir -p overnight
JOBIDS_FILE=overnight/crosstask_jobids.txt
: > "$JOBIDS_FILE"

for T in "${TASKS[@]}"; do
    echo ""
    echo ">>> Submitting cross-task eval: $T"
    out=$(run 1 1 "bash scripts/run_eval_rosch_crosstask_one.sh $T" 2>&1)
    echo "$out"
    jid=$(echo "$out" | grep -oE 'Submitted batch job [0-9]+' | awk '{print $4}')
    if [[ -n "$jid" ]]; then
        echo "$jid  $T" >> "$JOBIDS_FILE"
    fi
done

echo ""
echo "============================================================"
echo "Submitted ${#TASKS[@]} cross-task eval jobs."
echo "Job IDs / tasks recorded in $JOBIDS_FILE"
echo "============================================================"
cat "$JOBIDS_FILE"
