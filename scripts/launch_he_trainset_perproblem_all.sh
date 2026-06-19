#!/bin/bash
# Launch the FULL per-problem TRAIN-set eval matrix (the correct, per-problem version).
#   {upper,multi} x {gemma4,qwen} x [ base(once) + settings {1,2,3,4,7,13} x epochs {0,1,2} ]
# One checkpoint per job (backfill + resumable via trpp_<model> markers). gemma gpu:4, qwen gpu:2.
# base is scored once per (ds,model) under S=2 (gives self+neg modes covering all settings).
# Calls committed scripts/mll_he_trainset_perproblem.sbatch (no --train; per-problem tasks).
set -uo pipefail
cd /datastor2/jdr/rankalign
OUT=/datastor2/jdr/rankalign/outputs-he-trainset-perproblem
sub() { # model gpu S ckpts wtag hetask jobname
  MODEL="$1" CKPTS="$4" HE_TASK="$6" OUTDIR="$OUT" sbatch --gres="gpu:$2" --job-name="$7" \
    scripts/mll_he_trainset_perproblem.sbatch "$3" | sed "s/^/  $7: /"
}
echo "[launch_perproblem_all] submitting..."
for pair in cu:humaneval-v2.1correct-upper cm:humaneval-v2.1correct-multi; do
  wtag=${pair%%:*}; het=${pair##*:}
  for mt in gemma4:4 qwen:2; do
    model=${mt%%:*}; gpu=${mt##*:}; tag=$([ "$model" = gemma4 ] && echo g4 || echo qw)
    sub "$model" "$gpu" 2 base "$wtag" "$het" "trpp-$wtag-$tag-base"          # base (self+neg)
    for S in 1 2 3 4 7 13; do for C in 0 1 2; do
      sub "$model" "$gpu" "$S" "$C" "$wtag" "$het" "trpp-$wtag-$tag-s$S-ep$C"
    done; done
  done
done
echo "[launch_perproblem_all] done. trpp jobs queued: $(squeue -u jdr -h -o '%j'|grep -c trpp)"
