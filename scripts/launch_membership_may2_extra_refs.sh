#!/bin/bash
# Launch one short Slurm job per rosch task to fill in the missing eval-ref
# combinations on the 3 May-2 membership-sans-rosch-v0 pref-loss-only
# checkpoints (see scripts/run_eval_membership_may2_extra_refs.sh for the
# checkpoint × ref matrix).
#
# Each job runs ~3-7 minutes per checkpoint × ref pair, total ~7 evals per
# task across 10 tasks => 10 short jobs.
#
# Usage:
#   bash scripts/launch_membership_may2_extra_refs.sh

set -e
cd "$(dirname "$0")/.."

mkdir -p overnight
JOBIDS_FILE=overnight/membership_may2_extra_eval_jobids.txt
: > "$JOBIDS_FILE"

ROSCH_TASKS=(
    rosch-bird rosch-carpenters-tool rosch-clothing rosch-fruit
    rosch-furniture rosch-sport rosch-toy rosch-vegetable
    rosch-vehicle rosch-weapon
)

for TASK in "${ROSCH_TASKS[@]}"; do
    echo ""
    echo ">>> Submitting eval for $TASK"
    out=$(run 1 1 "bash scripts/run_eval_membership_may2_extra_refs.sh $TASK" 2>&1)
    echo "$out"
    jid=$(echo "$out" | grep -oE 'Submitted batch job [0-9]+' | awk '{print $4}')
    if [[ -n "$jid" ]]; then
        echo "$jid  may2_extra_refs__$TASK" >> "$JOBIDS_FILE"
    fi
done

echo ""
echo "============================================================"
echo "Submitted ${#ROSCH_TASKS[@]} eval jobs."
echo "JIDs / labels in $JOBIDS_FILE"
echo "============================================================"
cat "$JOBIDS_FILE"
