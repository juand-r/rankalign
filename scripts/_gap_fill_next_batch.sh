#!/bin/bash
# scripts/_gap_fill_next_batch.sh — autonomous gap-fill train submitter.
#
# Run by the wake loop every ~hour. Submits the next-priority train cells
# (and chained evals) so long as total slurm queue stays <= QUEUE_CAP.
#
# Two-pass logic:
#   Pass 1: Fire NO_BASE=1 evals for any *completed* non-TC train (s1/s2/s3/s13)
#           that doesn't yet have self-/neg- prefixed CSVs. This catches the
#           in-flight s13 trains' missing 2 prefix CSVs.
#   Pass 2: Fire next-priority train+eval bundles up to the queue cap.
#
# Priority order (highest first):
#   1. persona × 9b-it × s3        (deferred from Round 1)
#   2. ifeval × 2b-it × s1..s7     (fast 1-GPU trains, fills ifeval col for 2b-it)
#   3. ifeval × 2b × s1..s7        (1-GPU trains, fills ifeval col for 2b)
#   4. mem × 2b × s1..s7           (fills mem col for 2b)
#   5. persona × 2b × s1..s7       (fills persona col for 2b; s11/s12 already exist)
#   6. ifeval × 9b-it × s1..s7     (heavy 2-GPU trains)
#
# Each train fires 2 chained evals via _overnight_launch.sh (basetyp- + basetypneg-).
# For non-TC settings (s1/s2/s3) the batcher additionally schedules 2 NO_BASE
# evals (self- + neg- prefixes) on afterany dep of the train.

set -euo pipefail

cd /datastor1/jdr/gv-gap/rankalign
source /u/jdr/venvs/venv_lexcons/bin/activate

QUEUE_CAP="${QUEUE_CAP:-30}"          # leave 2 slots of headroom under the 32 hard cap
LOG=docs/gap_fill_logs/$(date +%Y%m%d_%H%M%S).log
mkdir -p docs/gap_fill_logs
exec > >(tee -a "$LOG") 2>&1

echo "=== gap-fill batcher tick @ $(date -u +%FT%TZ) ==="

# Current queue depth.
N_QUEUED=$(squeue -u "$USER" -h -o "%t" | wc -l)
SLOTS_FREE=$((QUEUE_CAP - N_QUEUED))
echo "queue depth: $N_QUEUED ; cap: $QUEUE_CAP ; slots free: $SLOTS_FREE"

if [ "$SLOTS_FREE" -le 0 ]; then
    echo "no slots free; exiting"
    exit 0
fi

# Priority gap list. Each line: DATASET MODEL SETTING WALLTIME EXTRA_NO_BASE_EVALS
#   EXTRA_NO_BASE_EVALS = 1 means schedule a NO_BASE=1 eval too (for s1/s2/s3 since
#   they get all 4 TC prefix variants; the chained _overnight_launch evals only
#   give the 2 basetyp- ones).
GAPS=$(cat <<'EOF'
persona      gemma-2-9b-it    s3   9   1
ifeval       gemma-2-2b-it    s1   6   1
ifeval       gemma-2-2b-it    s2   6   1
ifeval       gemma-2-2b-it    s3   7   1
ifeval       gemma-2-2b-it    s4   7   0
ifeval       gemma-2-2b-it    s7   7   0
ifeval       gemma-2-2b-it    s5   7   0
ifeval       gemma-2-2b-it    s6   7   0
ifeval       gemma-2-2b       s1   6   1
ifeval       gemma-2-2b       s2   6   1
ifeval       gemma-2-2b       s3   7   1
ifeval       gemma-2-2b       s4   7   0
ifeval       gemma-2-2b       s7   7   0
ifeval       gemma-2-2b       s5   7   0
ifeval       gemma-2-2b       s6   7   0
membership   gemma-2-2b       s1   6   1
membership   gemma-2-2b       s2   6   1
membership   gemma-2-2b       s3   7   1
membership   gemma-2-2b       s4   7   0
membership   gemma-2-2b       s7   7   0
membership   gemma-2-2b       s5   7   0
membership   gemma-2-2b       s6   7   0
persona      gemma-2-2b       s1   6   1
persona      gemma-2-2b       s2   6   1
persona      gemma-2-2b       s3   7   1
persona      gemma-2-2b       s4   7   0
persona      gemma-2-2b       s7   7   0
persona      gemma-2-2b       s5   7   0
persona      gemma-2-2b       s6   7   0
ifeval       gemma-2-9b-it    s1  10   1
ifeval       gemma-2-9b-it    s2  10   1
ifeval       gemma-2-9b-it    s3  12   1
ifeval       gemma-2-9b-it    s4  12   0
ifeval       gemma-2-9b-it    s7  12   0
ifeval       gemma-2-9b-it    s5  12   0
ifeval       gemma-2-9b-it    s6  12   0
EOF
)

# Build "already covered" set from on-disk models (epoch>=1) and queued trains.
# A cell counts as covered if EITHER:
#   (a) An adapter dir exists matching (model, dataset, setting) with epoch>=1.
#   (b) A train job is already queued/running (we approximate via squeue
#       scanning logs/<jobid>.out for "Model: google/<MODEL>" + dataset hint).
covered=$(python3 <<'PYEOF'
import os, re, sys
sys.path.insert(0, '/tmp')
sys.path.insert(0, '/datastor1/jdr/gv-gap/rankalign/scripts')
from _audit_v7_coverage import classify_setting, model_from_dirname, dataset_from_dirname
MODELS_DIR = '/datastor2/jdr/rankalign/models2'
LOGS_DIR = '/datastor2/jdr/logs'
covered = set()
# (a) on-disk
for d in os.listdir(MODELS_DIR):
    if not d.startswith('v7-google--') or d.endswith('_merged'):
        continue
    m = model_from_dirname(d)
    dsf, dss = dataset_from_dirname(d)
    s = classify_setting(d)
    m_ep = re.search(r'-epoch(\d+)--', d)
    ep = int(m_ep.group(1)) if m_ep else -1
    if m and dsf and s and ep >= 1:
        # store as DATASET key (long form) -> (DATASET, MODEL, s)
        # gap list uses dataset shorts: "membership", "persona", "ifeval", "humaneval"
        ds_short = {
            'membership-sans-rosch-v0': 'membership',
            'persona-v1': 'persona',
            'ifeval-concat': 'ifeval',
            'humaneval-v2.1correct-upper': 'humaneval',
        }.get(dsf, dsf)
        covered.add((ds_short, m, s))
# (b) queued/running trains (from log files of jobs in squeue)
import subprocess
sq = subprocess.run(['squeue', '-u', os.environ.get('USER', 'jdr'), '-h', '-o', '%i %j %t'],
                    capture_output=True, text=True).stdout
for line in sq.splitlines():
    parts = line.split()
    if not parts:
        continue
    jid, jname, st = parts[0], parts[1] if len(parts) > 1 else '', parts[2] if len(parts) > 2 else ''
    if not jname.startswith('wrap'):
        # eval jobs have specific names; skip
        continue
    # parse log for Model: google/...  and the dataset
    p = f'{LOGS_DIR}/{jid}.out'
    try:
        with open(p, 'r', errors='replace') as f:
            text = f.read(3000)  # first 3KB usually contains the banner
    except FileNotFoundError:
        continue
    mm = re.search(r'Model:\s*google/([^\s]+)', text)
    if not mm:
        continue
    mname = mm.group(1).strip()
    if 'persona-v1' in text:
        ds_short = 'persona'
    elif 'membership-sans-rosch' in text:
        ds_short = 'membership'
    elif 'ifeval' in text:
        ds_short = 'ifeval'
    elif 'humaneval' in text:
        ds_short = 'humaneval'
    else:
        continue
    # setting: look at flags in the command echo
    if '--cft' in text or '--consistency-ft' in text:
        s = 's13'
    elif '--labelonly0.1' in text and ('--pref0.0' in text or 'pref-only' not in text):
        s = 's1'
    elif '--self-typcorr' in text and '--force-same-x' in text and '--vallogodds' in text:
        s = 's4'
    elif '--self-typcorr' in text and '--force-same-x' in text:
        s = 's5'
    elif '--self-typcorr' in text and '--vallogodds' in text:
        s = 's11'
    elif '--self-typcorr' in text:
        s = 's6'
    elif '--neg-typcorr' in text and '--force-same-x' in text:
        s = 's7'
    elif '--neg-typcorr' in text:
        s = 's12'
    elif '--force-same-x' in text:
        s = 's3'
    else:
        s = 's2'
    covered.add((ds_short, mname, s))

for c in sorted(covered):
    print('|'.join(c))
PYEOF
)

declare -A COV
while IFS='|' read -r d m s; do
    [ -z "$d" ] && continue
    COV["$d $m $s"]=1
done <<< "$covered"

echo ""
echo "Already covered (on disk OR queued): ${#COV[@]} cells"

# Pass 2: submit next gaps until SLOTS_FREE reaches 0.
# A normal s4-s7,s11,s12 train fires 1 train + 2 evals = 3 jobs.
# A non-TC s1-s3,s13 train fires 1 train + 2 evals + 2 NO_BASE evals = 5 jobs.
SUBMITTED=0

while IFS= read -r line; do
    # skip blank / leading whitespace
    line=$(echo "$line" | sed -E 's/^[[:space:]]+//;s/[[:space:]]+/ /g')
    [ -z "$line" ] && continue
    read -r DATASET MODEL SETTING WALLTIME EXTRA <<<"$line"
    [ -z "$DATASET" ] && continue

    KEY="$DATASET $MODEL $SETTING"
    if [ -n "${COV[$KEY]:-}" ]; then
        # already trained or queued
        continue
    fi

    # Estimate cost (jobs to add).
    if [ "$EXTRA" = "1" ]; then
        COST=5    # train + 2 chained + 2 NO_BASE evals
    else
        COST=3    # train + 2 chained
    fi
    if [ "$SLOTS_FREE" -lt "$COST" ]; then
        echo "SKIP $KEY (cost=$COST, slots_free=$SLOTS_FREE)"
        # don't break; lighter cost cells later might still fit (though we ordered by priority)
        # but for simplicity, break now: priority order matters more than packing.
        break
    fi

    echo ""
    echo ">>> Launching: $DATASET × $MODEL × $SETTING (WALLTIME=$WALLTIME, extra_nobase=$EXTRA)"

    # 1) train + 2 chained evals via _overnight_launch.sh
    # Capture the train jobid to chain NO_BASE evals on.
    OUT=$(WALLTIME="$WALLTIME" bash scripts/_overnight_launch.sh "$DATASET" "$MODEL" "$SETTING" 2>&1)
    echo "$OUT"
    TRAIN_JID=$(echo "$OUT" | grep -oE 'Train submitted: jobid=[0-9]+' | head -1 | grep -oE '[0-9]+')
    if [ -z "$TRAIN_JID" ]; then
        echo "  WARN: failed to extract train jobid; skipping NO_BASE follow-up"
        SUBMITTED=$((SUBMITTED + 1))
        SLOTS_FREE=$((SLOTS_FREE - COST))
        continue
    fi
    echo "  train_jid=$TRAIN_JID"

    # 2) NO_BASE=1 evals for non-TC settings (only for s1/s2/s3/s13).
    if [ "$EXTRA" = "1" ]; then
        echo "  scheduling NO_BASE evals (self + neg) chained on afterany:$TRAIN_JID"
        # Map dataset short to MODEL_TAG / setting needs. _eval_only.sh expects
        # SETTING + DATASET + MODEL_TAG + EVAL_DEP env. Let me use it directly.
        for TC in self neg; do
            DEP_JOBID="$TRAIN_JID" NO_BASE=1 \
                bash scripts/_eval_only.sh "$DATASET" "$MODEL" "$SETTING" "$TC" 2>&1 | tail -3 || \
                echo "  WARN NO_BASE $TC failed"
        done
    fi

    SUBMITTED=$((SUBMITTED + 1))
    SLOTS_FREE=$((SLOTS_FREE - COST))
    COV["$KEY"]=1

    if [ "$SLOTS_FREE" -lt 3 ]; then
        echo ""
        echo "queue near cap; stopping batcher"
        break
    fi
done <<< "$GAPS"

echo ""
echo "=== batcher done @ $(date -u +%FT%TZ): submitted=$SUBMITTED, final_slots_free=$SLOTS_FREE ==="
