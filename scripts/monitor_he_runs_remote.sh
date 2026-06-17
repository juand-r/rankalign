#!/bin/bash
# monitor_he_runs_remote.sh — runs ON mll. Supervises the 16 NEW humaneval train+eval
# jobs: {gemma genctx, qwen} x s1/s3/s7/s13 x {upper, multi}. For each cell:
#   - if a job with that name is in squeue (RUNNING/PENDING) -> alive, leave it.
#   - else if its done-markers are complete -> done, leave it.
#   - else (left the queue before finishing) -> resubmit (launchers skip-if-done, so a
#     resubmit of a complete cell is a cheap no-op, and of a crashed cell resumes it).
# Writes .he_monitor/ALL_DONE when all 16 cells are complete. Idempotent; safe every 30 min.
#
# Scope: ONLY the 16 new s1/s3/s7/s13 cells. s2/s4 are done and NOT touched here.
# Expected done-markers per cell: s1/s3/s13 -> 328 (self+neg x 2bt x 82), s7 -> 164 (neg x 2bt x 82).
set -uo pipefail
REPO=/datastor2/jdr/rankalign
cd "$REPO" || exit 1
LOG=/datastor2/jdr/logs/he_monitor_remote.log
FLAGDIR="$REPO/.he_monitor"; mkdir -p "$FLAGDIR"
ts(){ date -u +%FT%TZ; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }

GG=pod-setup-train-scripts-gemma-4/mll_g4_train_eval_genctx.sbatch   # gemma genctx launcher (s1/s3/s7/s13)
GQ=scripts/mll_qwen35_he_train_eval.sbatch                          # qwen launcher
GMULTI_M="$REPO/gemma-4-models-mll-tmp-multi"
GMULTI_O="$REPO/outputs_gemma4_mll_tmp-multi"

INQ="$(squeue -u jdr -h -o '%j' 2>/dev/null)"   # current job names, one per line
expected(){ case "$1" in 1|3|13) echo 328;; 7) echo 164;; esac; }
cnt_markers(){ ls "$1"/$2 2>/dev/null | wc -l; }

complete_all=1
# args: name donedir markerglob setting resubmit-cmd
check_cell(){
  local name="$1" donedir="$2" glob="$3" S="$4" resub="$5"
  local exp cnt; exp=$(expected "$S"); cnt=$(cnt_markers "$donedir" "$glob")
  if printf '%s\n' "$INQ" | grep -qx "$name"; then
    log "  $name: ALIVE in queue (markers $cnt/$exp)"; [ "$cnt" -lt "$exp" ] && complete_all=0; return
  fi
  if [ "$cnt" -ge "$exp" ]; then log "  $name: COMPLETE ($cnt/$exp)"; return; fi
  complete_all=0
  log "  $name: DEAD+INCOMPLETE ($cnt/$exp) -> RESUBMIT"
  ( cd "$REPO" && eval "$resub" ) 2>&1 | sed 's/^/      /' | tee -a "$LOG"
}

log "===== monitor pass (jobs in queue: $(printf '%s' "$INQ" | grep -c . )) ====="
for S in 1 3 7 13; do
  check_cell "g4cu-s$S"  "$REPO/outputs_gemma4_mll_tmp/.done" "s${S}_*correct-upper*" "$S" \
    "HE_TASK=humaneval-v2.1correct-upper sbatch --job-name=g4cu-s$S $GG $S"
  check_cell "g4cm-s$S"  "$GMULTI_O/.done"                    "s${S}_*correct-multi*" "$S" \
    "HE_TASK=humaneval-v2.1correct-multi MODELS_DIR=$GMULTI_M OUTDIR=$GMULTI_O sbatch --job-name=g4cm-s$S $GG $S"
  check_cell "q35cu-s$S" "$REPO/outputs-rerun-wandb/.done"    "q35_cu_s${S}_*"        "$S" \
    "HE_TASK=humaneval-v2.1correct-upper sbatch --job-name=q35cu-s$S $GQ $S"
  check_cell "q35cm-s$S" "$REPO/outputs-rerun-wandb/.done"    "q35_cm_s${S}_*"        "$S" \
    "HE_TASK=humaneval-v2.1correct-multi sbatch --job-name=q35cm-s$S $GQ $S"
done

if [ "$complete_all" -eq 1 ]; then
  [ -f "$FLAGDIR/ALL_DONE" ] || { touch "$FLAGDIR/ALL_DONE"; log "*** ALL 16 CELLS COMPLETE -> ALL_DONE written ***"; }
else
  rm -f "$FLAGDIR/ALL_DONE"
fi
log "pass done (complete_all=$complete_all)"
