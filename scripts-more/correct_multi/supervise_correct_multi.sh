#!/bin/bash
# supervise_correct_multi.sh — SESSION-INDEPENDENT safety net.
#
# Run by the user's OS crontab (every ~17 min). Does NOT depend on any Claude
# session being alive. Guarantees the deterministic core completes: once the
# canary's kept_menu.json exists, the full v2.1correct-multi dataset gets
# built (idempotent) and a human-readable STATUS is written. This is the
# defense-in-depth that closes the "supervisor dies with the session" gap.
#
# It does the DETERMINISTIC step only (dataset build + status). Task
# registration / prose docs / the final summary remain agent/human work — but
# the science finishing no longer depends on an agent being awake.
set -uo pipefail

ROOT=/home/jdr/racas-more/llm-consistency-raca
PKG=$ROOT/private_projects/rankalign/scripts-more/correct_multi
NOTES=$ROOT/notes/log_P_diff_plots/humaneval-v2.1correct-multi
DATA=$ROOT/private_projects/rankalign/data/humaneval/v2.1correct-multi
PY=$ROOT/.tools-venv/bin/python
STATUS=$NOTES/STATUS.txt
LOG=$NOTES/supervise.log
LOCK=/tmp/.supervise_correct_multi.lock
stamp(){ date -u +%Y-%m-%dT%H:%M:%SZ; }
say(){ echo "[$(stamp)] $*" >> "$LOG"; }

# single-instance (cron overlap guard)
exec 9>"$LOCK"; flock -n 9 || { say "another run active; skip"; exit 0; }
mkdir -p "$NOTES"

# already done?
if [ -d "$DATA" ] && [ "$(ls "$DATA"/humaneval_*.csv 2>/dev/null | wc -l)" -ge 80 ] \
   && [ -f "$DATA/_BUILD_REPORT.json" ]; then
    echo "DONE: v2.1correct-multi built ($(ls "$DATA"/humaneval_*.csv|wc -l) csv). $(stamp)" > "$STATUS"
    say "dataset already built — nothing to do"
    exit 0
fi

MENU=$NOTES/kept_menu.json
if [ ! -f "$MENU" ]; then
    echo "WAITING: canary not finished — kept_menu.json not present yet. $(stamp)" > "$STATUS"
    say "kept_menu.json absent; canary still in progress (agent/heartbeat handles canary)"
    exit 0
fi
if [ "$($PY -c "import json;print(json.load(open('$MENU'))['premise_supported'])" 2>/dev/null)" != "True" ]; then
    echo "STOP: canary verdict = premise NOT supported. Dataset intentionally NOT built. $(stamp)" > "$STATUS"
    say "premise not supported — degenerate-outcome stop, not building"
    exit 0
fi

# build is not done but the menu exists → run the committed builder (idempotent-ish:
# it rewrites data/v2.1correct-multi/ deterministically from seed=42)
if pgrep -f build_v2_1_correct_multi.py >/dev/null 2>&1; then
    echo "BUILDING: build_v2_1_correct_multi.py in progress ($(ls "$DATA"/*.csv 2>/dev/null|wc -l) csv so far). $(stamp)" > "$STATUS"
    say "builder already running; let it finish"
    exit 0
fi

say "kept_menu present, dataset not built, no builder running -> launching builder"
echo "BUILDING (launched by OS-cron supervisor): $(stamp)" > "$STATUS"
"$PY" "$PKG/build_v2_1_correct_multi.py" --menu "$MENU" --seed 42 \
    >> "$NOTES/build_dataset.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ] && [ -f "$DATA/_BUILD_REPORT.json" ]; then
    n=$(ls "$DATA"/humaneval_*.csv 2>/dev/null | wc -l)
    echo "DATASET BUILT: $n csv at $DATA (built by OS-cron safety net). Remaining: task registration + docs (agent/human). $(stamp)" > "$STATUS"
    say "BUILD SUCCESS: $n csv"
else
    echo "BUILD FAILED rc=$rc — see build_dataset.log. Will retry next cron tick. $(stamp)" > "$STATUS"
    say "build failed rc=$rc; retry next tick"
fi
