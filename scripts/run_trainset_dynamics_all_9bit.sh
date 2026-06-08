#!/bin/bash
# Launch train-set dynamics evals for ALL gemma-2-9b-it membership settings.
# Calls run_trainset_dynamics.sh for each (setting, TC) combo.
# Base model already done — skip with NO_BASE=1.
#
# Usage: bash scripts/run_trainset_dynamics_all_9bit.sh
#   Set DRYRUN=1 to preview without submitting.

set -euo pipefail

export NO_BASE=1  # base already evaluated

# Settings needing self-TC eval
for S in s2 s3 s5 s6 s11; do
    echo ""; echo "===== $S (self) ====="
    TC_OVERRIDE=self bash scripts/run_trainset_dynamics.sh gemma-2-9b-it "$S"
done

# Settings needing neg-TC eval
for S in s2 s3 s7 s12; do
    echo ""; echo "===== $S (neg) ====="
    TC_OVERRIDE=neg bash scripts/run_trainset_dynamics.sh gemma-2-9b-it "$S"
done

# s1 only has ep0 and ep1
for TC in self neg; do
    echo ""; echo "===== s1 ($TC) ====="
    EPOCHS="0 1" TC_OVERRIDE=$TC bash scripts/run_trainset_dynamics.sh gemma-2-9b-it s1
done

echo ""
echo "All submitted."
