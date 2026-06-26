#!/bin/bash
# Reproducibility re-run #3 (v3): re-TRAIN + EVAL qwen Qwen3.5-9B on IFEval for the rerun
# matrix (s1,s2,s3,s4,s7), writing to SEPARATE -v3 dirs so the existing wandb reruns
# (outputs-rerun-wandb, outputs-rerun-wandb-v2) are NEVER overwritten. Goal: a THIRD
# independent train-to-train datapoint to gauge per-method variance.
#
# EXACT replication of _qwen_ifeval_repro.sh (the v2 launcher): same run_qwen35_cell_mll.sbatch,
# same SETTING args, same flags. ONLY the three env vars below differ (dir + wandb suffix):
#   - models  -> models2-rerun-wandb-v3/   (new, must be EMPTY so it retrains)
#   - scores  -> outputs-rerun-wandb-v3/   (new)
#   - wandb run names suffixed -v3
#   - venv    -> qwen35 (sbatch default; NOT overridden)
#   - eval    -> IFEval OOD only (sbatch default IFEVAL_SPLIT=ood)
#   - epochs  -> 3 (sbatch default; saves epoch0/1/2)
set -euo pipefail
REPO=/datastor2/jdr/rankalign
MODELS_DIR=$REPO/models2-rerun-wandb-v3
OUTPUTS_DIR=$REPO/outputs-rerun-wandb-v3
WANDB_SUFFIX=-v3
cd "$REPO"

# Safety: a non-empty MODELS_DIR would make the sbatch SKIP training (eval-only) -> defeats the
# purpose. Refuse unless it is absent or empty.
if [ -d "$MODELS_DIR" ] && [ -n "$(ls -A "$MODELS_DIR" 2>/dev/null)" ]; then
    echo "REFUSING: $MODELS_DIR exists and is non-empty -> training would be skipped. Aborting." >&2
    exit 1
fi

for S in s1 s2 s3 s4 s7; do
    jid=$(sbatch --parsable --job-name="qwre3-$S" \
        --export=ALL,MODELS_DIR=$MODELS_DIR,OUTPUTS_DIR=$OUTPUTS_DIR,WANDB_SUFFIX=$WANDB_SUFFIX \
        scripts/run_qwen35_cell_mll.sbatch ifeval "$S")
    echo "ifeval $S -> job $jid"
done
echo "models -> $MODELS_DIR ; scores -> $OUTPUTS_DIR ; wandb suffix $WANDB_SUFFIX"
