#!/bin/bash
# Run compute_val_loss.py for all settings on a given model/task.
#
# Usage:
#   bash scripts/run_val_loss.sh              # defaults: gemma-2-9b-it, membership
#   MODEL=Qwen/Qwen3.5-9B TASK=ifeval-concat bash scripts/run_val_loss.sh
#
# Submits one GPU job per setting (each loads base + 3 checkpoints sequentially).

set -euo pipefail

MODEL="${MODEL:-google/gemma-2-9b-it}"
TASK="${TASK:-membership-sans-rosch-v0}"
MODELS_DIR="${MODELS_DIR:-/datastor2/jdr/rankalign/models2-rerun-wandb}"
SETTINGS="${SETTINGS:-s1 s2 s3 s4 s7}"
OUTPUT="${OUTPUT:-analysis/tables/val_loss.csv}"
HOURS="${HOURS:-3}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================"
echo "Validation Loss Computation"
echo "  MODEL:      $MODEL"
echo "  TASK:       $TASK"
echo "  MODELS_DIR: $MODELS_DIR"
echo "  SETTINGS:   $SETTINGS"
echo "  OUTPUT:     $OUTPUT"
echo "  HOURS:      $HOURS"
echo "============================================"

for s in $SETTINGS; do
    echo
    echo ">>> Setting $s"

    if [ -n "${DRYRUN:-}" ]; then
        echo "  DRYRUN: run 1 $HOURS --mem 64G /u/jdr/venvs/venv_lexcons/bin/python $REPO/scripts/compute_val_loss.py --model $MODEL --task $TASK --setting $s --models-dir $MODELS_DIR --output $OUTPUT"
    else
        run 1 "$HOURS" --mem 64G /u/jdr/venvs/venv_lexcons/bin/python "$REPO/scripts/compute_val_loss.py" --model "$MODEL" --task "$TASK" --setting "$s" --models-dir "$MODELS_DIR" --output "$OUTPUT"
    fi
done

echo
echo "=== All settings dispatched ==="
