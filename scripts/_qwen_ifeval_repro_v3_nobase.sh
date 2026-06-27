#!/bin/bash
# Reproducibility re-run #3 (v3) — SECOND eval pass: NO_BASE=1 (no-base / self- + neg- CSVs).
#
# WHY THIS EXISTS: a COMPLETE eval for the rerun matrix needs TWO passes, because the sbatch's
# single eval invocation writes only one CSV family per run:
#   pass 1 (default)  -> --base-typicality ON  -> scores_basetyp-/scores_basetypneg- CSVs
#   pass 2 (NO_BASE=1)-> --base-typicality OFF -> scores_self-/scores_neg- CSVs   <-- THIS SCRIPT
# _qwen_ifeval_repro_v3.sh ran pass 1 only. In the v2 run, pass 2 was done as a SEPARATE job two
# days later (basetyp files dated 2026-06-10, self/neg dated 2026-06-12) -- i.e. it was easy to
# forget the second pass. This script makes pass 2 explicit so v3 matches v2's full mode set.
#
# EVAL-ONLY: the epoch2 model dirs already exist in models2-rerun-wandb-v3, so run_qwen35_cell_mll
# .sbatch SKIPS training and only evaluates. EXACT replication of v2's second pass: same sbatch,
# same per-setting EVAL_MODES (s1/s2/s3 self+neg, s4 self, s7 neg), IFEval OOD only. Adds NO_BASE=1.
set -euo pipefail
REPO=/datastor2/jdr/rankalign
MODELS_DIR=$REPO/models2-rerun-wandb-v3
OUTPUTS_DIR=$REPO/outputs-rerun-wandb-v3
WANDB_SUFFIX=-v3
cd "$REPO"

# Safety: this MUST be eval-only. Every setting's epoch2 checkpoint has to exist already, or the
# sbatch would silently kick off a fresh 3-epoch retrain. Refuse unless all 5 are present.
N=$(ls -d "$MODELS_DIR"/*epoch2*_merged 2>/dev/null | wc -l)
if [ "$N" -ne 5 ]; then
    echo "REFUSING: expected 5 epoch2_merged dirs in $MODELS_DIR, found $N -> would retrain. Aborting." >&2
    exit 1
fi

for S in s1 s2 s3 s4 s7; do
    jid=$(sbatch --parsable --job-name="qwre3nb-$S" \
        --export=ALL,NO_BASE=1,MODELS_DIR=$MODELS_DIR,OUTPUTS_DIR=$OUTPUTS_DIR,WANDB_SUFFIX=$WANDB_SUFFIX \
        scripts/run_qwen35_cell_mll.sbatch ifeval "$S")
    echo "ifeval $S (NO_BASE eval-only) -> job $jid"
done
echo "models -> $MODELS_DIR ; scores -> $OUTPUTS_DIR ; NO_BASE=1 -> writes scores_self-/scores_neg- CSVs"
