#!/bin/bash
# monitor_trpp_remote.sh — runs ON mll (via laptop cron). Supervises the per-problem TRAIN-set
# eval matrix (trpp): {cu,cm} x {g4,qw} x [base + s1/2/3/4/7/13 x ep0/1/2]. For each cell: in
# queue -> alive; markers complete -> done; else resubmit (resumable via trpp_<model> markers).
# Writes .he_monitor/TRPP_ALL_DONE when all cells complete.
set -uo pipefail
REPO=/datastor2/jdr/rankalign
DONE_DIR="$REPO/outputs-he-trainset-perproblem/.done"
FLAG="$REPO/.he_monitor/TRPP_ALL_DONE"
mkdir -p "$REPO/.he_monitor"
NT=80   # train problems/dataset
ts(){ date -u +%FT%TZ; }
inq(){ squeue -u jdr -h -o '%j' | grep -qx "$1"; }
# expected markers: modes(self+neg=2 for base/s1/s2/s3/s13; 1 for s4/s7) x bt(2) x NT
exp(){ case "$1" in 4|7) echo $((1*2*NT));; *) echo $((2*2*NT));; esac; }
mk(){ ls "$DONE_DIR/trpp_${1}_${2}_s${3}_${4}_"*.done 2>/dev/null | wc -l; }  # model wtag S cname
submit(){ MODEL="$1" CKPTS="$4" HE_TASK="$6" OUTDIR="$REPO/outputs-he-trainset-perproblem" \
  sbatch --gres="gpu:$2" --job-name="$7" "$REPO/scripts/mll_he_trainset_perproblem.sbatch" "$3" >/dev/null && echo "[$(ts)] RESUBMIT $7"; }
all=1
for pair in cu:humaneval-v2.1correct-upper cm:humaneval-v2.1correct-multi; do
  wtag=${pair%%:*}; het=${pair##*:}
  for mt in gemma4:4:g4 qwen:2:qw; do
    model=$(echo $mt|cut -d: -f1); gpu=$(echo $mt|cut -d: -f2); tag=$(echo $mt|cut -d: -f3)
    # base (S=2 modes)
    jn="trpp-$wtag-$tag-base"
    if [ "$(mk $model $wtag 2 base)" -ge "$(exp 2)" ]; then :; elif inq "$jn"; then all=0; else all=0; submit $model $gpu 2 base $wtag $het "$jn"; fi
    for S in 1 2 3 4 7 13; do for C in 0 1 2; do
      jn="trpp-$wtag-$tag-s$S-ep$C"
      if [ "$(mk $model $wtag $S ep$C)" -ge "$(exp $S)" ]; then :; elif inq "$jn"; then all=0; else all=0; submit $model $gpu $S $C $wtag $het "$jn"; fi
    done; done
  done
done
[ "$all" = 1 ] && { touch "$FLAG"; echo "[$(ts)] TRPP_ALL_DONE"; } || echo "[$(ts)] trpp in progress (running: $(squeue -u jdr -h -o '%j'|grep -c trpp))"
