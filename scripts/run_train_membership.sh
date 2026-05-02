#!/bin/bash
# Training runs for membership-sans-rosch-v0 (LoRA on non-2b models per run_train_semi.sh).
# Eval out-of-distribution on rosch-X after training (see run_eval_membership_trained_models.sh).
#
# 5 variants (same pattern as codecontests):
#   1. SFT, labelonly, no force-same-x
#   2. RankAlign: pref-only, semi, no TC, no val log-odds, no force-same-x
#   3. Comb, semi, val log-odds, force-same-x
#   4. Comb, semi, val log-odds, self-TC, force-same-x
#   5. Pref-only, semi, val log-odds, self-TC, force-same-x
#
# Usage:
#   bash scripts/run_train_membership.sh [MODEL]
#
#   MODEL: HuggingFace id (default: google/gemma-2-9b-it).
#   First argument overrides the MODEL environment variable if both are set.
#
# Examples:
#   bash scripts/run_train_membership.sh
#   bash scripts/run_train_membership.sh google/gemma-2-2b
#   bash scripts/run_train_membership.sh google/gemma-2-2b-it
#   MODEL=google/gemma-2-2b HOURS=8 bash scripts/run_train_membership.sh
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
echo "  HOURS:  $HOURS (each of 5 jobs)"
echo "========================================"

run 1 "$HOURS" scripts/run_train_semi.sh "$MODEL" "$TASK" sft labelonly 0.1 --no-force-same-x

run 1 "$HOURS" scripts/run_train_semi.sh "$MODEL" "$TASK" pref-only semi 0.1 --no-force-same-x

run 1 "$HOURS" scripts/run_train_semi.sh "$MODEL" "$TASK" comb semi 0.1 --log-odds

run 1 "$HOURS" scripts/run_train_semi.sh "$MODEL" "$TASK" comb semi 0.1 --self-typcorr --log-odds

run 1 "$HOURS" scripts/run_train_semi.sh "$MODEL" "$TASK" pref-only semi 0.1 --self-typcorr --log-odds
