#!/bin/bash
# Evaluate the ORIGINAL pod qwen models in the gemma4 venv (transformers 5.x, NO fla ->
# torch fallback, ~= the pod environment), IFEval OOD, eval-only. Pairs with the wandb
# reruns evaluated in qwen35 (fla) so each model is run in the env closest to how it was
# produced, and the fla-kernel difference becomes a controlled factor rather than a confound.
#
# Read-out: validator val_roc/val_acc per setting (aggregate later). Compare:
#   - original pod eval (on the pod): SFT val_roc = 57.5
#   - rerun model in qwen35:          SFT val_roc = 82.8
#   - THIS (original model in gemma4): tells us whether the env/kernel (not the model)
#     drives the gap. If ~57 -> env(fla) is the driver; if ~83 -> the original pod eval
#     itself was the anomaly.
#
# Scores -> outputs-orig-in-gemma4/ (isolated). Uses the canonical-named (delta0.001) dirs
# so checkpoint_name_parser derives a valid eval name.
set -euo pipefail
REPO=/datastor2/jdr/rankalign
GEMMA4=/datastor2/jdr/venvs/gemma4
cd "$REPO"

# setting -> original-HF canonical model dir (delta0.001 = placeholder; delta is
# generator-side and does NOT affect the validator score).
declare -A MODELS=(
  [s1]="models2-original-hf/v7-Qwen--Qwen3.5-9B-delta0.001-epoch2--ifeval-concat-all--d2g--random--alpha1.0--full-completion--pref0.0--nllv1.0--nllg1.0--vallogodds--labelonly0.1--fix1_merged"
)
for S in "${!MODELS[@]}"; do
    DIR="$REPO/${MODELS[$S]}"
    if [ ! -d "$DIR" ]; then echo "MISSING: $DIR"; exit 1; fi
    jid=$(sbatch --parsable \
        --job-name="orig-g4-$S" \
        --export=ALL,VENV=$GEMMA4,EVAL_ONLY=1,NO_BASE=1,IFEVAL_SPLIT=ood,OUTPUTS_DIR=$REPO/outputs-orig-in-gemma4,MODEL_DIR_OVERRIDE="$DIR" \
        scripts/run_qwen35_cell_mll.sbatch ifeval "$S")
    echo "$S (orig pod model, gemma4 venv) -> job $jid"
done
