#!/bin/bash
# Batch launcher: train-set dynamics evals for ALL Qwen3.5-9B ifeval-concat settings.
#
# Settings available (same structure as gemma 9b-it):
#   s1 (no TC) -> eval self + neg
#   s2 (no TC) -> eval self + neg
#   s3 (no TC) -> eval self + neg
#   s4 (tc=self) -> eval self only
#   s7 (tc=neg) -> eval neg only
#
# Total: 2 base + (s1×6 + s2×6 + s3×6 + s4×3 + s7×3) = 26 jobs
#
# Usage:
#   bash scripts/run_trainset_dynamics_ifeval_all_qwen.sh
#   DRYRUN=1 bash scripts/run_trainset_dynamics_ifeval_all_qwen.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "Launching train-set dynamics: ifeval-concat (Qwen3.5-9B)"
echo "=========================================="

export NO_BASE=1  # base submitted separately below

# Base model: self-TC and neg-TC
echo ""
echo "===== Base model (self-TC) ====="
NO_BASE="" EPOCHS="" TC_OVERRIDE=self bash "$SCRIPT_DIR/run_trainset_dynamics_ifeval_qwen.sh" s3

echo ""
echo "===== Base model (neg-TC) ====="
NO_BASE="" EPOCHS="" TC_OVERRIDE=neg bash "$SCRIPT_DIR/run_trainset_dynamics_ifeval_qwen.sh" s7

# Settings needing self-TC eval: s1, s2, s3, s4
for S in s1 s2 s3 s4; do
    echo ""; echo "===== $S (self) ====="
    TC_OVERRIDE=self bash "$SCRIPT_DIR/run_trainset_dynamics_ifeval_qwen.sh" "$S"
done

# Settings needing neg-TC eval: s1, s2, s3, s7
for S in s1 s2 s3 s7; do
    echo ""; echo "===== $S (neg) ====="
    TC_OVERRIDE=neg bash "$SCRIPT_DIR/run_trainset_dynamics_ifeval_qwen.sh" "$S"
done

echo ""
echo "All submitted."
