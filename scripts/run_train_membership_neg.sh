#!/bin/bash
# Training runs for membership-sans-rosch-v0 with NEG-typicality correction
# (LoRA on non-2b models per run_train_semi.sh).
# Eval out-of-distribution on rosch-X after training.
#
# Sibling of run_train_membership.sh: launches only the two TC-using variants
# with --neg-typcorr. The non-TC variants (sft labelonly, pref-only semi
# vanilla, comb semi log-odds) live in run_train_membership.sh and don't need
# to be re-run.
#
# Variants:
#   1. Comb, semi, val log-odds, neg-TC, force-same-x
#   2. Pref-only, semi, val log-odds, neg-TC, force-same-x
#
# Usage:
#   bash scripts/run_train_membership_neg.sh [MODEL]
#
#   MODEL: HuggingFace id (default: google/gemma-2-9b-it).
#   First argument overrides the MODEL environment variable if both are set.
#
# Examples:
#   bash scripts/run_train_membership_neg.sh
#   bash scripts/run_train_membership_neg.sh google/gemma-2-2b
#   bash scripts/run_train_membership_neg.sh google/gemma-2-2b-it
#   MODEL=google/gemma-2-2b HOURS=8 bash scripts/run_train_membership_neg.sh
#
# Env:
#   MODEL   - default HF model when no positional argument is given
#   HOURS   - Slurm walltime hours per job (default: 6)

MODEL="${1:-${MODEL:-google/gemma-2-9b-it}}"
HOURS="${HOURS:-6}"
TASK=membership-sans-rosch-v0

echo "========================================"
echo "Membership training launcher"
echo "  MODEL:  $MODEL"
echo "  TASK:   $TASK"
echo "  HOURS:  $HOURS (each of 2 jobs)"
echo "========================================"

run 1 "$HOURS" scripts/run_train_semi.sh "$MODEL" "$TASK" comb semi 0.1 --neg-typcorr --log-odds

run 1 "$HOURS" scripts/run_train_semi.sh "$MODEL" "$TASK" pref-only semi 0.1 --neg-typcorr --log-odds
