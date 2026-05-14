#!/bin/bash
# Fan out evals for the ambigqa delta sweep. Submits one Slurm job per delta,
# each calling scripts/run_eval_ambigqa_2b_one_delta.sh and depending on the
# corresponding training job (afterany) so it starts as soon as that train
# finishes (success or failure).
#
# Walltime is 2 h per eval (vs 1 h for the original 38009, which TIMEOUT'd
# before basetyp/basetypneg were written).
#
# Reads JIDs from overnight/ambigqa_deltasweep_jobids.txt (written by
# scripts/run_train_ambigqa_2b_deltasweep.sh) and matches them by delta.
#
# Submits via sbatch directly (NOT via ~/.local/bin/run) because run does
# not pass --dependency through.
#
# Usage:
#   bash scripts/launch_ambigqa_deltasweep_evals.sh

set -e

cd "$(dirname "$0")/.."

JIDS_FILE=overnight/ambigqa_deltasweep_jobids.txt
if [[ ! -f "$JIDS_FILE" ]]; then
    echo "ERROR: $JIDS_FILE not found. Run scripts/run_train_ambigqa_2b_deltasweep.sh first." >&2
    exit 1
fi

mkdir -p overnight ~/logs
EVAL_JIDS_FILE=overnight/ambigqa_deltasweep_eval_jobids.txt
: > "$EVAL_JIDS_FILE"

echo "Train JIDs / deltas:"
cat "$JIDS_FILE"
echo ""

while read -r jid label; do
    if [[ -z "$jid" ]]; then continue; fi
    # label is e.g. "delta=0.3"
    delta=$(echo "$label" | sed -E 's/^delta=//')
    if [[ -z "$delta" ]]; then
        echo "WARN: could not parse delta from line: $jid $label"
        continue
    fi

    echo ">>> Submitting eval: delta=$delta (depends on train JID $jid)"
    out=$(sbatch \
        --partition=allnodes \
        --cpus-per-task=12 \
        --mem=120G \
        --gres=gpu:1 \
        --time=2:00:00 \
        --dependency="afterany:$jid" \
        --output="$HOME/logs/%j.out" \
        --error="$HOME/logs/%j.err" \
        --wrap="PYTHONUNBUFFERED=1 /usr/bin/time -v bash scripts/run_eval_ambigqa_2b_one_delta.sh $delta" 2>&1)
    echo "$out"
    eval_jid=$(echo "$out" | grep -oE 'Submitted batch job [0-9]+' | awk '{print $4}')
    if [[ -n "$eval_jid" ]]; then
        echo "$eval_jid  delta=$delta" >> "$EVAL_JIDS_FILE"
    fi
done < "$JIDS_FILE"

echo ""
echo "============================================================"
echo "Eval JIDs / deltas in $EVAL_JIDS_FILE:"
cat "$EVAL_JIDS_FILE"
echo "============================================================"
