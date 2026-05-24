#!/bin/bash
# Phase 2 launcher: less-important settings (s5, s6, s11, s12) x 2 models x 3 datasets.
# 24 cells, ~48 jobs (1 eval per train). Run when there's queue capacity
# AND phase 1 has been validated (i.e. some 1A jobs have started without
# argparse / runtime crashes).
#
# Usage: bash scripts/_overnight_phase2.sh

set -e
cd "$(dirname "$0")/.."

for setting in s5 s6 s11 s12; do
    for model in gemma-2-2b-it gemma-2-9b-it; do
        for dataset in persona membership ifeval; do
            bash scripts/_overnight_launch.sh "$dataset" "$model" "$setting" 2>&1 | tail -3
        done
    done
done

echo "phase 2 submission complete"
