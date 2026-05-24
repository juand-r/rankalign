#!/bin/bash
# Phase 1B launcher: gemma-2-9b-it × 5 priority settings × 3 datasets.
# Fires the unified dispatcher in a loop. Only run when queue depth allows
# (see overnight_plan.md; aim for total queue <= 32).
#
# Usage: bash scripts/_overnight_phase1b.sh
#   Submits 15 trains + ~16 evals (s7 has tc=neg eval, all others tc=self
#   except s1/s2/s3 which we capped at "self" only). Total ~30 jobs.

set -e
cd "$(dirname "$0")/.."

for setting in s4 s7 s2 s3 s1; do
    for dataset in persona membership ifeval; do
        bash scripts/_overnight_launch.sh "$dataset" gemma-2-9b-it "$setting" 2>&1 | tail -3
    done
done

echo "phase 1B submission complete"
