#!/usr/bin/env python3
"""
Generate one script PER NOUN (task), grouping all missing vallogodds evals
for that noun into a single script. Each script runs all the model configs
for that noun sequentially -- intended to be launched as one job on 1 GPU.

Usage:
  python scripts/generate_slurm_eval_jobs.py
  # Then with your launcher:
  for f in scripts/slurm_jobs/eval_*.sh; do run 1 16 "bash $f"; done
"""
import os
import sys
from collections import OrderedDict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)
os.chdir(REPO_ROOT)

import list_missing_v6_runs

def main():
    _, missing_eval = list_missing_v6_runs.main()
    out_dir = os.path.join(REPO_ROOT, "scripts", "slurm_jobs")
    os.makedirs(out_dir, exist_ok=True)

    # Clean old scripts
    for f in os.listdir(out_dir):
        if f.startswith("eval_") and f.endswith(".sh"):
            os.remove(os.path.join(out_dir, f))

    # Filter to vallogodds only, group by task (noun)
    by_task = OrderedDict()
    for model_dir, task in missing_eval:
        if "vallogodds" not in model_dir:
            continue
        by_task.setdefault(task, []).append(model_dir)

    for task, model_dirs in by_task.items():
        lines = ["#!/bin/bash"]
        lines.append("set -e")
        lines.append('REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"')
        lines.append('cd "$REPO_ROOT"')
        lines.append('echo "=== Evaluating %d models for %s ==="' % (len(model_dirs), task))
        lines.append("")
        for j, model_dir in enumerate(model_dirs):
            model_path = "models/" + model_dir
            lines.append("echo '[%d/%d] %s'" % (j + 1, len(model_dirs), model_dir))
            lines.append("bash scripts/run_one_cmd.sh eval %s %s" % (task, model_path))
            lines.append("")
        lines.append('echo "=== Done: %s ==="' % task)
        lines.append("")

        noun = task.replace("hypernym-", "")
        out_path = os.path.join(out_dir, "eval_%s.sh" % noun)
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
        print("Wrote %s  (%d evals)" % (out_path, len(model_dirs)))

    total = sum(len(v) for v in by_task.values())
    print("\n%d scripts, %d total evals (vallogodds only)." % (len(by_task), total))
    print("Submit with: for f in scripts/slurm_jobs/eval_*.sh; do run 1 16 \"bash $f\"; done")


if __name__ == "__main__":
    main()
