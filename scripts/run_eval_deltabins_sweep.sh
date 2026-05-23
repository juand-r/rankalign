#!/bin/bash
# Launch evaluation for the 15-model delta-bins sweep on persona-v1.
#
# Strategy: glob v7-*--tc-self--*--fix1 dirs in models2/, group by
# (base_model, delta_value), pick latest epoch dir, and submit one eval job
# covering all 6 persona-v1 personas (3 ID + 3 OOD).
#
# Eval flags: --self-typcorr --base-typcorr --base-model <BASE> --log-odds
#             disc-shots=few (default of run_eval_semi.sh, matches training)
#             --save-scores-csv (always set in run_eval_semi.sh)
#
# Usage: bash scripts/run_eval_deltabins_sweep.sh
#
# Env:
#   HOURS    - walltime per eval job (default: 2)
#   CPUS     - cpus per task (default: 6)
#   MEM      - memory per job (default: 60G)
#   PARTITION - slurm partition (default: allnodes)
#   MODELS_DIR - directory with v7- model dirs (default: ../models2 from repo root)
#   DRYRUN=1 - print sbatch commands without submitting

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

HOURS="${HOURS:-2}"
CPUS="${CPUS:-6}"
MEM="${MEM:-60G}"
PARTITION="${PARTITION:-allnodes}"
MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models2}"

if [ ! -d "$MODELS_DIR" ]; then
    echo "ERROR: MODELS_DIR not found: $MODELS_DIR" >&2
    exit 1
fi

TASKS=(
    persona-v1-psychopathy
    persona-v1-machiavellianism
    persona-v1-narcissism
    persona-v1-desire-to-create-allies
    persona-v1-interest-in-music
    persona-v1-interest-in-science
)

OVERNIGHT_DIR="${REPO_ROOT}/overnight"
mkdir -p "$OVERNIGHT_DIR"
JOBID_FILE="$OVERNIGHT_DIR/persona_v1_eval_deltabins_jobids.txt"
: > "$JOBID_FILE"

echo "========================================"
echo "Delta-bins sweep eval launcher"
echo "  MODELS_DIR: $MODELS_DIR"
echo "  HOURS:      $HOURS / job"
echo "  CPUS / MEM: $CPUS / $MEM  on partition $PARTITION"
echo "  Tasks:      ${#TASKS[@]}  (${TASKS[*]})"
echo "  Jobid log:  $JOBID_FILE"
echo "========================================"

submit_one() {
    local base_model="$1"
    local model_path="$2"
    local label="$3"

    if [ ! -d "$model_path" ]; then
        echo "[$label] SKIP - dir not found: $model_path"
        echo "(no jobid - SKIP)  $label  $model_path" >> "$JOBID_FILE"
        return
    fi

    EVAL_COMMON="--log-odds"
    WRAP_CMD="PYTHONUNBUFFERED=1 /usr/bin/time -v scripts/run_eval_semi.sh ${model_path} --self-typcorr --base-typcorr --base-model ${base_model} ${EVAL_COMMON} -- ${TASKS[*]}"

    if [ -n "${DRYRUN:-}" ]; then
        echo "DRYRUN: sbatch --partition=$PARTITION --cpus-per-task=$CPUS --mem=$MEM --gres=gpu:1 --time=${HOURS}:00:00 --output=/datastor2/jdr/logs/%j.out --error=/datastor2/jdr/logs/%j.err --wrap=\"$WRAP_CMD\""
        OUT="DRYRUN-no-submit"
    else
        OUT=$(sbatch \
            --partition="$PARTITION" \
            --cpus-per-task="$CPUS" \
            --mem="$MEM" \
            --gres=gpu:1 \
            --time="${HOURS}:00:00" \
            --output=/datastor2/jdr/logs/%j.out \
            --error=/datastor2/jdr/logs/%j.err \
            --wrap="$WRAP_CMD" 2>&1) || true
        echo "$OUT"
    fi
    JOBID=$(echo "$OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+$' | head -1)
    if [ -n "$JOBID" ]; then
        echo "$JOBID  $label  $model_path" >> "$JOBID_FILE"
    else
        echo "(no jobid)  $label  $model_path" >> "$JOBID_FILE"
    fi
}

# Use python to enumerate v7-*--tc-self--*--fix1 dirs and pick latest epoch
# per (base_model, delta_value). Outputs lines: <base_model> <delta> <model_path> <epoch>
python3 - "$MODELS_DIR" <<'PYEOF' > /tmp/_deltabins_evalplan.txt
import os, re, sys
from collections import defaultdict

models_dir = sys.argv[1]
# Match: v7-google--gemma-2-9b-it-delta0.498512...-epoch0--persona-v1-all--d2g--random--alpha1.0--tc-self--full-completion--nllv1.0--nllg1.0--vallogodds--semi0.1--fix1[_merged]
pat = re.compile(r'^v7-(google--gemma-2-(?:2b|2b-it|9b-it))-delta([0-9.]+)-epoch(\d+)--persona-v1-all--d2g--random--alpha1\.0--tc-self--full-completion--nllv1\.0--nllg1\.0--vallogodds--semi0\.1--fix1(_merged)?$')

# Group by (base_repl, delta_str) -> dict[epoch -> path]
groups = defaultdict(dict)
for name in os.listdir(models_dir):
    m = pat.match(name)
    if not m:
        continue
    base_repl, delta_str, epoch_str, merged = m.group(1), m.group(2), m.group(3), m.group(4) or ""
    epoch = int(epoch_str)
    full_path = os.path.join(models_dir, name)
    # Existing convention (run_eval_persona_v1_trained_fix1.sh):
    #   2b, 2b-it -> use plain dir (no _merged)
    #   9b-it     -> use _merged dir (LoRA-merged into base for eval)
    is_9b_it = ('9b-it' in base_repl)
    if is_9b_it and not merged:
        continue
    if (not is_9b_it) and merged:
        continue
    groups[(base_repl, delta_str)][epoch] = full_path

for (base_repl, delta_str), epoch_paths in sorted(groups.items()):
    base_model = base_repl.replace('--', '/')
    latest_epoch = max(epoch_paths.keys())
    path = epoch_paths[latest_epoch]
    print(f"{base_model}\t{delta_str}\t{latest_epoch}\t{path}")
PYEOF

echo ""
echo "Found $(wc -l < /tmp/_deltabins_evalplan.txt) (base, delta) combos with at least one epoch saved:"
cat /tmp/_deltabins_evalplan.txt
echo ""

# Also need a (base, delta) -> bins lookup, from auto_delta_log.csv if present.
LOG_FILE="${MODELS_DIR}/auto_delta_log.csv"
declare -A BINS_BY_KEY
if [ -f "$LOG_FILE" ]; then
    while IFS=, read -r ts model task metric n mn p5 p95 mx spread bins delta_used delta_arg; do
        [ "$ts" = "timestamp" ] && continue
        [ -z "$delta_used" ] && continue
        # Truncate delta_used to 6 decimals to make a robust lookup key
        # (auto_delta_log.csv writes 6 decimals; dirname uses full repr).
        key="${model}__$(printf '%.6f' "$delta_used")"
        BINS_BY_KEY[$key]="$bins"
    done < "$LOG_FILE"
fi

while IFS=$'\t' read -r base_model delta_str latest_epoch model_path; do
    [ -z "$base_model" ] && continue
    # Round delta to 6 decimals for lookup
    delta_rounded=$(python3 -c "print(f'{float(\"$delta_str\"):.6f}')")
    bins="${BINS_BY_KEY[${base_model}__${delta_rounded}]:-?}"
    LABEL="bins${bins}_${base_model}_ep${latest_epoch}_self+base"
    echo ""
    echo ">>> [$LABEL] (delta=$delta_str)"
    submit_one "$base_model" "$model_path" "$LABEL"
done < /tmp/_deltabins_evalplan.txt

rm -f /tmp/_deltabins_evalplan.txt

echo ""
echo "========================================"
echo "Done. Jobid log:"
cat "$JOBID_FILE"
