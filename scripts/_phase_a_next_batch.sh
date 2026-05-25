#!/bin/bash
# _phase_a_next_batch.sh — autonomous batch submitter for Phase A.
#
# Run on each wake tick. It:
#   1) Polls squeue, computes free_slots = max(0, 32 - active_jobs).
#   2) Reads scripts/_eval_coverage_matrix.py output's "Gaps requiring
#      submission" table (regenerates fresh).
#   3) Submits up to free_slots gaps in priority order:
#         priority A: mem × 2b-it gaps   (finishing the mem table)
#         priority B: mem × 9b-it gaps
#         priority C: persona × 2b-it gaps
#         priority D: persona × 9b-it gaps  (afterany-eligible)
#         priority E: humaneval × s13   (afterany on 42114)
#   4) Logs to docs/phase_a_logs/auto_<timestamp>.log + returns.
#
# Intentionally stupid-simple. No retries on FATAL; we just skip missing-glob
# cells and keep going (those are the s1 cells whose train hasn't saved yet).
set -uo pipefail
cd /datastor1/jdr/gv-gap/rankalign

CAP="${CAP:-32}"
ACTIVE=$(squeue -u "$USER" --noheader 2>/dev/null | wc -l)
FREE=$(( CAP - ACTIVE ))
if [ "$FREE" -le 0 ]; then
    echo "queue at $ACTIVE/$CAP — no free slots, exiting"
    exit 0
fi
echo "queue at $ACTIVE/$CAP — $FREE slots free"

mkdir -p docs/phase_a_logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="docs/phase_a_logs/auto_${TS}.log"
echo "# Auto batch at $(date -u +%FT%TZ)  ACTIVE=$ACTIVE FREE=$FREE" | tee "$LOG"

source /u/jdr/venvs/venv_lexcons/bin/activate
# Regenerate the coverage matrix (writes docs/eval_coverage_matrix.md AND
# prints to stdout).
python scripts/_eval_coverage_matrix.py > docs/eval_coverage_matrix.md

# Extract the "Gaps requiring submission" rows. Format:
# | dataset | model | setting | column | csv prefix | NO_BASE? |
# We want: dataset, model, setting, NO_BASE?, and TC inferred from column.
# column is "PMI base"/"PMI self" (→ tc=self) or "Neg base"/"Neg self" (→ tc=neg).
# csv prefix is one of basetyp-/self-/basetypneg-/neg-.
# We submit as: NO_BASE=$nb bash scripts/_eval_only.sh $ds $model $setting $tc

# Priority sort: mem×2b-it, mem×9b-it, persona×2b-it, persona×9b-it, humaneval
priority () {
    local ds="$1" mdl="$2"
    case "$ds-$mdl" in
        membership-gemma-2-2b-it) echo 1 ;;
        membership-gemma-2-9b-it) echo 2 ;;
        persona-gemma-2-2b-it)    echo 3 ;;
        persona-gemma-2-9b-it)    echo 4 ;;
        humaneval-gemma-4-31B-it) echo 5 ;;
        *) echo 9 ;;
    esac
}

# Parse the gap table from the markdown.
TMPGAPS=$(mktemp)
awk '/^## Gaps requiring submission/,0' docs/eval_coverage_matrix.md \
    | awk -F'|' 'NF>=7 && $2 ~ /^ (membership|persona|humaneval|ifeval) / {
        ds=$2; mdl=$3; setting=$4; col=$5; pfx=$6; nb=$7;
        gsub(/^ +| +$/,"",ds); gsub(/^ +| +$/,"",mdl);
        gsub(/^ +| +$/,"",setting); gsub(/^ +| +$/,"",col);
        gsub(/^ +| +$/,"",nb);
        # Skip ifeval (no models trained yet, would FATAL).
        if (ds == "ifeval") next;
        tc = (col ~ /^Neg/) ? "neg" : "self";
        # Print: priority dataset model setting tc nb
        print ds, mdl, setting, tc, nb
    }' > "$TMPGAPS"

if [ ! -s "$TMPGAPS" ]; then
    echo "no gaps in matrix — Phase A appears complete!"
    rm -f "$TMPGAPS"
    exit 0
fi

# Sort by priority, then take top FREE rows.
SORTED=$(mktemp)
while read -r ds mdl setting tc nb; do
    p=$(priority "$ds" "$mdl")
    echo "$p $ds $mdl $setting $tc $nb"
done < "$TMPGAPS" | sort -k1,1n -k2,2 -k3,3 -k4,4 -k5,5 > "$SORTED"
rm -f "$TMPGAPS"

echo "" | tee -a "$LOG"
echo "# Top $FREE gap rows (of $(wc -l < $SORTED) total):" | tee -a "$LOG"

submitted=0
while read -r p ds mdl setting tc nb; do
    if [ "$submitted" -ge "$FREE" ]; then break; fi
    echo "" | tee -a "$LOG"
    echo "[$((submitted+1))/$FREE]  $ds × $mdl × $setting × $tc × NO_BASE=$nb" | tee -a "$LOG"
    NO_BASE_ARG=""
    [ "$nb" = "1" ] && NO_BASE_ARG="NO_BASE=1"
    OUT=$(env $NO_BASE_ARG bash scripts/_eval_only.sh "$ds" "$mdl" "$setting" "$tc" 2>&1)
    echo "$OUT" | tail -3 | tee -a "$LOG"
    if echo "$OUT" | grep -q "FATAL"; then
        echo "  -> FATAL (likely missing model dir; will retry on next tick)" | tee -a "$LOG"
        # Don't count toward submitted (no slot consumed)
    else
        submitted=$((submitted + 1))
    fi
done < "$SORTED"
rm -f "$SORTED"

echo "" | tee -a "$LOG"
echo "# Done. Submitted=$submitted. Final queue:" | tee -a "$LOG"
squeue -u "$USER" --noheader | wc -l | tee -a "$LOG"
