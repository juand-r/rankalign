#!/bin/bash
# Watches recovery_v2.log for completion markers and downloads scores locally
LOG=/workspace/logs/recovery_v2.log
OUTDIR=/workspace/outputs
DONE_FILE=/workspace/logs/watch_done.txt
> 

while true; do
    for MARKER in VARIANT_no_tc_COMPLETE VARIANT_self_tc_COMPLETE VARIANT_neg_tc_COMPLETE PIPELINE_COMPLETE; do
        if grep -q "" "$LOG" 2>/dev/null && ! grep -q "" "$DONE_FILE" 2>/dev/null; then
            echo "[$(date +%H:%M:%S)]  detected" | tee -a "$DONE_FILE"
        fi
    done
    sleep 60
done
