#!/bin/bash
# Launch the 16 INDIVIDUAL gemma s2/s4 train-set eval jobs (one checkpoint each):
#   {cu,cm} x {s2,s4} x {base,ep0,ep1,ep2}  = 16 jobs, each gpu:4, 5h, --train.
# Each job runs ONE checkpoint for max backfill parallelism. Calls the committed
# launcher scripts/mll_he_trainset_eval.sbatch with CKPTS=<single>. Idempotent-ish:
# re-running submits fresh jobs, so only run once (the monitor handles restarts after).
set -uo pipefail
cd /datastor2/jdr/rankalign
echo "[launch_he_trainset_16] submitting 16 jobs..."
for pair in cu:humaneval-v2.1correct-upper cm:humaneval-v2.1correct-multi; do
  wtag=${pair%%:*}; hetask=${pair##*:}
  for S in 2 4; do
    for C in base 0 1 2; do
      if [ "$C" = base ]; then cname=base; else cname="ep$C"; fi
      CKPTS=$C HE_TASK=$hetask sbatch --job-name="g4trset-$wtag-s$S-$cname" \
        scripts/mll_he_trainset_eval.sbatch "$S" | sed "s/^/  $wtag-s$S-$cname: /"
    done
  done
done
echo "[launch_he_trainset_16] done. trset jobs in queue: $(squeue -u jdr -h -o '%j' | grep -c trset)"
