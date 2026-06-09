#!/bin/bash
# Reproducibility re-run: re-TRAIN + EVAL qwen Qwen3.5-9B on IFEval for the full rerun
# matrix (s1,s2,s3,s4,s7), writing to SEPARATE dirs so the existing wandb reruns are
# NEVER overwritten. Goal: gauge train-to-train variance — compare these fresh numbers
# to the current rerun-only table (esp. the SFT validator).
#
# Identical flags to the original rerun (same run_qwen35_cell_mll.sbatch, same SETTING args;
# verified: no flag-touching commits since, and existing model-dir suffixes match). Only the
# output dirs + wandb run names differ.
#   - models  -> models2-rerun-wandb-v2/   (new)
#   - scores  -> outputs-rerun-wandb-v2/   (new)
#   - wandb run names suffixed -v2
#   - venv    -> qwen35 (sbatch default; NOT overridden)
#   - eval    -> IFEval OOD only (sbatch default IFEVAL_SPLIT=ood)
#   - epochs  -> 3 (sbatch default; saves epoch0/1/2)
set -euo pipefail
REPO=/datastor2/jdr/rankalign
MODELS_DIR=$REPO/models2-rerun-wandb-v2
OUTPUTS_DIR=$REPO/outputs-rerun-wandb-v2
WANDB_SUFFIX=-v2
cd "$REPO"
for S in s1 s2 s3 s4 s7; do
    jid=$(sbatch --parsable --job-name="qwre-$S" \
        --export=ALL,MODELS_DIR=$MODELS_DIR,OUTPUTS_DIR=$OUTPUTS_DIR,WANDB_SUFFIX=$WANDB_SUFFIX \
        scripts/run_qwen35_cell_mll.sbatch ifeval "$S")
    echo "ifeval $S -> job $jid"
done
echo "models -> $MODELS_DIR ; scores -> $OUTPUTS_DIR ; wandb suffix $WANDB_SUFFIX"
