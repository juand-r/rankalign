#!/bin/bash
# Add the MISSING own-typicality (self-/neg-) evals to the v2 reproducibility run.
#
# The v2 train+eval (_qwen_ifeval_repro.sh) produced BASE-typicality only (basetyp-/basetypneg-).
# The original rerun also has own-typ (self-/neg-), so the side-by-side needs them too. This runs
# EVAL-ONLY with NO_BASE=1 on the v2 EPOCH-2 models (the sbatch globs MODELS_DIR for the newest
# epoch-2 merged dir), which makes eval_by_claude emit self-/neg- (own-typ) score CSVs into the
# same v2 output dir. Scores land in outputs-rerun-wandb-v2/; qwen35 venv (sbatch default).
set -euo pipefail
REPO=/datastor2/jdr/rankalign
cd "$REPO"
for S in s1 s2 s3 s4 s7; do
    jid=$(sbatch --parsable --job-name="qwre-own-$S" \
        --export=ALL,MODELS_DIR=$REPO/models2-rerun-wandb-v2,OUTPUTS_DIR=$REPO/outputs-rerun-wandb-v2,EVAL_ONLY=1,NO_BASE=1,IFEVAL_SPLIT=ood \
        scripts/run_qwen35_cell_mll.sbatch ifeval "$S")
    echo "ifeval $S own-typ eval -> job $jid"
done
