#!/bin/bash
# Batch launcher: train-set dynamics evals for ALL gemma-2-9b-it ifeval-concat settings.
#
# Settings available:
#   s1 (no TC) -> eval self + neg
#   s2 (no TC) -> eval self + neg
#   s3 (no TC) -> eval self + neg
#   s4 (tc=self) -> eval self only
#   s7 (tc=neg) -> eval neg only (self too? user may decide)
#   s13 (cft, no TC) -> eval self + neg [excluded by default per user request]
#
# Total (excl s13): s1(6) + s2(6) + s3(6) + s4(3) + s7(3) = 24 epoch evals
#   + 1 base (self) + 1 base (neg) = 26 jobs
#
# Usage:
#   bash scripts/run_trainset_dynamics_ifeval_all_9bit.sh
#
# Set DRYRUN=1 to preview without submitting.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "Launching train-set dynamics: ifeval-concat"
echo "Model: gemma-2-9b-it"
echo "=========================================="

export NO_BASE=1  # base already evaluated (submitted separately below)

# Base model: self-TC and neg-TC (epoch evals skipped for base-only)
# We submit base via a dummy setting but with EPOCHS="" so only base fires.
echo ""
echo "===== Base model (self-TC) ====="
NO_BASE="" EPOCHS="" TC_OVERRIDE=self bash "$SCRIPT_DIR/run_trainset_dynamics_ifeval.sh" s3

echo ""
echo "===== Base model (neg-TC) ====="
NO_BASE="" EPOCHS="" TC_OVERRIDE=neg bash "$SCRIPT_DIR/run_trainset_dynamics_ifeval.sh" s7

# Settings needing self-TC eval: s1, s2, s3, s4
for S in s1 s2 s3 s4; do
    echo ""; echo "===== $S (self) ====="
    TC_OVERRIDE=self bash "$SCRIPT_DIR/run_trainset_dynamics_ifeval.sh" "$S"
done

# Settings needing neg-TC eval: s1, s2, s3, s7
for S in s1 s2 s3 s7; do
    echo ""; echo "===== $S (neg) ====="
    TC_OVERRIDE=neg bash "$SCRIPT_DIR/run_trainset_dynamics_ifeval.sh" "$S"
done
