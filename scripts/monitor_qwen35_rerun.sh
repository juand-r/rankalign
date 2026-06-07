#!/bin/bash
# monitor_qwen35_rerun.sh — overnight supervisor for the qwen3.5-9b wandb-rerun matrix on mll.
# ---------------------------------------------------------------------------
# RUNS ON MLL (login node — lightweight: squeue/sacct + sbatch only, no GPU work).
# Invoked every 30 min by the LAPTOP crontab via:
#     raca ssh mll 'bash /datastor2/jdr/rankalign/scripts/monitor_qwen35_rerun.sh'
#
# Supervises 10 cells (TRAIN_ONLY=1 wandb-curve reruns): ifeval/membership x s1/s2/s3/s4/s7.
# Each tick, for every cell's current Slurm job:
#   COMPLETED                          -> mark DONE (curve written + merged model saved)
#   PENDING/RUNNING/REQUEUED/...       -> ACTIVE, leave it
#   FAILED/TIMEOUT/NODE_FAIL/OOM/...   -> resubmit (TRAIN_ONLY=1, same flags), capped at MAXRE
# Writes ALL_DONE.flag once every cell is DONE (or GAVEUP after the resubmit cap).
#
# State + log live on mll under $REPO/.monitor/ so this is restart-safe and the laptop
# wrapper is stateless. The launcher is idempotent (skips train if the merged dir exists),
# so a blind resubmit is always safe.
#
# TEARDOWN: when ALL_DONE.flag appears (or pods/jobs are stopped), REMOVE the laptop
# crontab entry immediately:  crontab -l | grep -v monitor_qwen35_rerun | crontab -
set -uo pipefail

REPO=/datastor2/jdr/rankalign
LAUNCHER=scripts/run_qwen35_cell_mll.sbatch
SDIR=$REPO/.monitor
STATE=$SDIR/qwen35_rerun_state.tsv      # cell <TAB> jobid <TAB> resubmit_count <TAB> status
LOG=$SDIR/qwen35_rerun.log
DONE=$SDIR/ALL_DONE.flag
MAXRE=3
# sbatch --export string used for resubmits; set per-run in $SDIR/resubmit_export.
# Defaults to ALL. The disc-shots-few membership run uses "ALL,WANDB_SUFFIX=-discfew"
# (NO TRAIN_ONLY — we want train+eval). The earlier curve-only run used "ALL,TRAIN_ONLY=1".
REXPORT=$(cat "$SDIR/resubmit_export" 2>/dev/null || echo "ALL")

mkdir -p "$SDIR"
cd "$REPO" || { echo "cannot cd $REPO"; exit 1; }
ts(){ date '+%F %T %Z'; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }

[ -f "$DONE" ] && { log "ALL_DONE flag present — nothing to do."; exit 0; }
[ -f "$STATE" ] || { log "ERROR: no state file $STATE (init it first)"; exit 1; }

tmp=$(mktemp)
while IFS=$'\t' read -r cell jid rc st; do
    [ -z "${cell:-}" ] && continue
    ds=${cell%%:*}; setting=${cell##*:}
    # main-step state (first sacct line = the job allocation, not .batch/.extern)
    state=$(sacct -j "$jid" -n -P -o State 2>/dev/null | head -1 | awk '{print $1}')
    [ -z "$state" ] && state="UNKNOWN"
    case "$state" in
        COMPLETED)
            st=DONE ;;
        RUNNING|PENDING|REQUEUED|RESIZING|SUSPENDED|CONFIGURING|COMPLETING)
            st=ACTIVE ;;
        FAILED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|BOOT_FAIL|DEADLINE|CANCELLED*)
            if [ "${rc:-0}" -lt "$MAXRE" ]; then
                elapsed=$(sacct -j "$jid" -n -P -o Elapsed 2>/dev/null | head -1)
                out=$(sbatch --export="$REXPORT" --time=24:00:00 \
                        --job-name="qwrr-$ds-$setting" "$LAUNCHER" "$ds" "$setting" 2>&1)
                njid=$(echo "$out" | grep -oE '[0-9]+' | tail -1)
                if [ -n "$njid" ]; then
                    rc=$((rc+1)); jid="$njid"; st=ACTIVE
                    log "RESUBMIT $cell (was $state after ${elapsed:-?}) -> job $njid (resubmit #$rc/$MAXRE)"
                else
                    st=ACTIVE
                    log "RESUBMIT FAILED for $cell: $out (will retry next tick)"
                fi
            else
                st=GAVEUP
                log "GAVEUP $cell after $MAXRE resubmits (last state=$state) — needs manual debug"
            fi ;;
        *)
            st=ACTIVE
            log "UNKNOWN sacct state '$state' for $cell job $jid — leaving ACTIVE" ;;
    esac
    printf '%s\t%s\t%s\t%s\n' "$cell" "$jid" "${rc:-0}" "$st" >> "$tmp"
done < "$STATE"
mv "$tmp" "$STATE"

log "STATE: $(awk -F'\t' '{printf "%s=%s(j%s,re%s) ",$1,$4,$2,$3}' "$STATE")"

# ALL_DONE when no cell is still ACTIVE (everything DONE or GAVEUP)
if ! awk -F'\t' '{print $4}' "$STATE" | grep -qvE '^(DONE|GAVEUP)$'; then
    touch "$DONE"
    ndone=$(awk -F'\t' '$4=="DONE"' "$STATE" | wc -l)
    ngave=$(awk -F'\t' '$4=="GAVEUP"' "$STATE" | wc -l)
    log "ALL CELLS SETTLED -> wrote ALL_DONE.flag ($ndone DONE, $ngave GAVEUP). REMOVE THE CRONTAB NOW."
fi
