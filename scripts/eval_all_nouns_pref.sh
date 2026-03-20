#!/bin/bash
#
# Evaluate all 8 Pref union models on a single noun.
# One job per noun, processing all model variants sequentially.
#
# Usage:
#   bash scripts/eval_all_nouns_pref.sh <noun>
#
# Example:
#   run 1 5 "bash -c 'bash /datastor1/jdr/gv-gap/rankalign/scripts/eval_all_nouns_pref.sh bananas'"
#
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/scripts"

NOUN="$1"
[ -z "$NOUN" ] && { echo "Usage: eval_all_nouns_pref.sh <noun>"; exit 1; }
TASK="hypernym-${NOUN}"

PREFIX="v6-google--gemma-2-2b-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-double-all--d2g--random--alpha1.0"

MODELS=(
    "${PREFIX}--full-completion--force-same-x"
    "${PREFIX}--full-completion--force-same-x--vallogodds"
    "${PREFIX}--tc-online--full-completion--force-same-x"
    "${PREFIX}--tc-online--full-completion--force-same-x--vallogodds"
    "${PREFIX}--lenorm--full-completion--force-same-x"
    "${PREFIX}--lenorm--full-completion--force-same-x--vallogodds"
    "${PREFIX}--tc-online--lenorm--full-completion--force-same-x"
    "${PREFIX}--tc-online--lenorm--full-completion--force-same-x--vallogodds"
)

TOTAL=${#MODELS[@]}
echo "=== Evaluating $TOTAL Pref models on $TASK ==="

for i in "${!MODELS[@]}"; do
    MODEL_DIR="${MODELS[$i]}"
    MODEL_PATH="../models/${MODEL_DIR}"
    echo "[$(($i+1))/$TOTAL] ${MODEL_DIR}"
    python3 eval.py \
        --model "$MODEL_PATH" \
        --task "$TASK" \
        --split_type random \
        --validator-log-odds \
        --typicality-correction \
        --save-scores-csv \
        --viz
    echo ""
done

echo "=== Done: $TASK (Pref) ==="
