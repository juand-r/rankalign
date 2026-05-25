#!/bin/bash
# Autonomous overnight worker. Runs every 20 min:
#   - master scheduler (Pass 1 eval gap fills + Pass 2 train gap fills)
#   - table regen at every 3rd tick (~1h)
# Logs each tick to docs/overnight_worker_logs/<UTCstamp>.log
#
# Designed to run in a background shell on the login node so progress
# continues even when the agent is not active.

set -u

cd /datastor1/jdr/gv-gap/rankalign
source /u/jdr/venvs/venv_lexcons/bin/activate

mkdir -p docs/overnight_worker_logs
LOG_BASE=docs/overnight_worker_logs
SUMMARY=$LOG_BASE/_summary.txt
echo "=== worker started @ $(date -u +%FT%TZ) (pid=$$) ===" >> "$SUMMARY"

ITER=0
while true; do
    ITER=$((ITER + 1))
    STAMP=$(date -u +%Y%m%d_%H%M%S)
    LOG=$LOG_BASE/${STAMP}_iter${ITER}.log

    {
        echo "=== iter $ITER @ $(date -u +%FT%TZ) ==="
        echo ""
        echo "=== queue ==="
        squeue -u jdr -h -o "%t" | sort | uniq -c
        echo ""

        echo "=== master scheduler ==="
        QUEUE_CAP=100 MAX_P1=12 python scripts/_overnight_master.py 2>&1 | tail -60
        echo ""

        # Every 3rd tick, regen tables
        if [ $((ITER % 3)) -eq 0 ]; then
            echo "=== refreshing tables (every 3rd tick) ==="
            T_STAMP=$(date -u +%Y%m%d_%H%M%S)
            for METRIC in gen_roc pearson spearman val_roc val_acc; do
                for MODEL in 2b 2b-it 9b-it; do
                    ROSCH_METRIC=$METRIC ROSCH_MODEL=$MODEL \
                        python scripts/_build_rosch_table_v7.py 2>/dev/null \
                        > $LOG_BASE/rosch_${METRIC}_${MODEL}_${T_STAMP}.txt &
                    PERSONA_METRIC=$METRIC PERSONA_BASE=$MODEL \
                        python scripts/_build_persona_v1_table_v7.py 2>/dev/null \
                        > $LOG_BASE/persona_${METRIC}_${MODEL}_${T_STAMP}.txt &
                done
            done
            wait
            echo "tables regenerated -> $LOG_BASE/{rosch,persona}_*_${T_STAMP}.txt"
        fi
        echo ""
        echo "=== done @ $(date -u +%FT%TZ) ==="
    } > "$LOG" 2>&1

    echo "$(date -u +%FT%TZ)  iter=$ITER  log=$LOG" >> "$SUMMARY"
    sleep 1200    # 20 min
done
