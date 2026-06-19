#!/bin/bash
# Launch the EXTRA train-set eval jobs needed for the training-dynamics report
# (beyond GROUP B's gemma s2/s4 which are already done):
#   gemma4 {s3,s7} : ep0/ep1/ep2  (base already done in GROUP B)
#   qwen   base    : one job (S=2 -> self+neg modes, covers all settings' base)
#   qwen   {s2,s3,s4,s7} : ep0/ep1/ep2
# Each job = ONE checkpoint for backfill. Calls committed mll_he_trainset_eval_v2.sbatch.
# Resumable via per-task done-markers (TAG trs2_<model>); safe to re-run (skips finished).
set -uo pipefail
cd /datastor2/jdr/rankalign
sub() {  # sub <model> <gpu> <S> <ckpts> <wtag> <hetask> <jobname>
  local model="$1" gpu="$2" S="$3" ck="$4" wtag="$5" het="$6" jn="$7"
  MODEL="$model" CKPTS="$ck" HE_TASK="$het" sbatch --gres="gpu:$gpu" --job-name="$jn" \
    scripts/mll_he_trainset_eval_v2.sbatch "$S" | sed "s/^/  $jn: /"
}
echo "[launch_he_trainset_extra] submitting..."
for pair in cu:humaneval-v2.1correct-upper cm:humaneval-v2.1correct-multi; do
  wtag=${pair%%:*}; het=${pair##*:}
  # gemma s3,s7 (ep0/1/2; base already have)
  for S in 3 7; do for C in 0 1 2; do
    sub gemma4 4 "$S" "$C" "$wtag" "$het" "trs2-$wtag-g4-s$S-ep$C"
  done; done
  # qwen base (self+neg via S=2)
  sub qwen 2 2 base "$wtag" "$het" "trs2-$wtag-qw-base"
  # qwen s2,s3,s4,s7 (ep0/1/2)
  for S in 2 3 4 7; do for C in 0 1 2; do
    sub qwen 2 "$S" "$C" "$wtag" "$het" "trs2-$wtag-qw-s$S-ep$C"
  done; done
done
echo "[launch_he_trainset_extra] done. trs2 jobs in queue: $(squeue -u jdr -h -o '%j' | grep -c trs2)"
