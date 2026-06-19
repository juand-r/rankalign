#!/bin/bash
# monitor_trs2_remote.sh — runs ON mll (via laptop cron). Supervises the 38 EXTRA
# train-set eval cells (gemma s3/s7 + qwen base/s2/s3/s4/s7, both datasets) launched by
# launch_he_trainset_extra.sh. For each cell: if its job is in queue -> alive; else if its
# done-markers are complete -> done; else resubmit (resumable, skips finished tasks).
# Writes .he_monitor/TRS2_ALL_DONE when every cell is complete. Logs to stdout (cron->log).
set -uo pipefail
REPO=/datastor2/jdr/rankalign
DONE_DIR="$REPO/outputs-he-trainset/.done"
FLAG="$REPO/.he_monitor/TRS2_ALL_DONE"
mkdir -p "$REPO/.he_monitor"
NT=82                               # problems per dataset
ts(){ date -u +%FT%TZ; }
inqueue(){ squeue -u jdr -h -o '%j' | grep -qx "$1"; }

# expected markers per cell: modes(s2/s3=2, s4/s7=1) x bt(2) x NT
exp_markers(){ case "$1" in 2|3) echo $((2*2*NT));; 4|7) echo $((1*2*NT));; esac; }
markers(){ ls "$DONE_DIR/trs2_${1}_${2}_s${3}_${4}_"*.done 2>/dev/null | wc -l; }  # model wtag S cname

submit(){ # model gpu S ckpts wtag hetask jobname
  MODEL="$1" CKPTS="$4" HE_TASK="$6" sbatch --gres="gpu:$2" --job-name="$7" \
    "$REPO/scripts/mll_he_trainset_eval_v2.sbatch" "$3" >/dev/null && echo "[$(ts)] RESUBMIT $7"
}

all_done=1
for pair in cu:humaneval-v2.1correct-upper cm:humaneval-v2.1correct-multi; do
  wtag=${pair%%:*}; het=${pair##*:}
  # gemma s3,s7 ep0/1/2
  for S in 3 7; do for C in 0 1 2; do
    jn="trs2-$wtag-g4-s$S-ep$C"; exp=$(exp_markers $S); got=$(markers gemma4 $wtag $S ep$C)
    if [ "$got" -ge "$exp" ]; then :; elif inqueue "$jn"; then all_done=0; else all_done=0; submit gemma4 4 "$S" "$C" "$wtag" "$het" "$jn"; fi
  done; done
  # qwen base (S=2 modes)
  jn="trs2-$wtag-qw-base"; exp=$(exp_markers 2); got=$(markers qwen $wtag 2 base)
  if [ "$got" -ge "$exp" ]; then :; elif inqueue "$jn"; then all_done=0; else all_done=0; submit qwen 2 2 base "$wtag" "$het" "$jn"; fi
  # qwen s2,s3,s4,s7 ep0/1/2
  for S in 2 3 4 7; do for C in 0 1 2; do
    jn="trs2-$wtag-qw-s$S-ep$C"; exp=$(exp_markers $S); got=$(markers qwen $wtag $S ep$C)
    if [ "$got" -ge "$exp" ]; then :; elif inqueue "$jn"; then all_done=0; else all_done=0; submit qwen 2 "$S" "$C" "$wtag" "$het" "$jn"; fi
  done; done
done
if [ "$all_done" = 1 ]; then touch "$FLAG"; echo "[$(ts)] TRS2_ALL_DONE — all 38 cells complete"; else echo "[$(ts)] trs2 in progress (running: $(squeue -u jdr -h -o '%j'|grep -c trs2))"; fi
