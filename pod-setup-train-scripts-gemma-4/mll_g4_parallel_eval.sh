#!/bin/bash
# Launch PARALLEL per-combo eval jobs for a trained gemma-4 setting (run after training, or to
# speed up eval vs the combined job's sequential pass). eval_by_claude reloads the 31B per task,
# so splitting by (mode,bt) combo parallelizes wall-clock across the idle A40 nodes.
#   s2: {self,neg} x {bt1 base, bt0 no-base} = 4 jobs ;  s4: {self} x {bt1,bt0} = 2 jobs
# Each job evals ALL 82 humaneval-v2.1correct-upper tasks for ONE combo. Shares OUTDIR +
# per-(epoch,mode,bt,task) done-markers with the train job -> idempotent, no double work.
set -euo pipefail
S="${1:?usage: mll_g4_parallel_eval.sh <2|4>}"
REPO=/datastor1/jdr/gv-gap/rankalign
cd "$REPO"
case "$S" in
  2) MODES="--self-typicality --neg-typicality" ;;
  4) MODES="--self-typicality" ;;
  *) echo "FATAL: setting must be 2 or 4"; exit 2 ;;
esac
for MODE in $MODES; do
  for BT in 1 0; do
    mn="${MODE#--}"
    # DEP=<train_jobid> -> eval waits for training to finish OK (afterok dependency).
    jid=$(sbatch --parsable --job-name="g4ev-s${S}-${mn}-bt${BT}" \
        ${DEP:+--dependency=afterok:$DEP} \
        --export=ALL,EVAL_ONLY=1,ONLY_MODE="$MODE",ONLY_BT="$BT" \
        pod-setup-train-scripts-gemma-4/mll_g4_train_eval.sbatch "$S")
    echo "s$S eval $MODE bt$BT -> job $jid"
  done
done
