#!/usr/bin/env python3
"""Salvage rosch-task metrics from Slurm .out logs of jobs that crashed
with OSError 36 (filename too long) BEFORE writing score CSVs.

For each job, the eval iterates 10 rosch tasks. Every per-task metric block
is printed to stdout BEFORE the failed CSV write, so we can scrape:

    [N/10] rosch-<task>
    correlation: all = X, pos = X, neg = X
    spearman: all = X, pos = X, neg = X
    disc_acc: X, disc_roc: X (threshold=...)
    gen_roc: X

This script does NOT require any score CSVs to exist.
"""
import re
import sys
import math
from pathlib import Path

JOBS = {
    41982: "membership-gemma-2-9b-it-s4-evaltc-self",
    41992: "membership-gemma-2-9b-it-s7-evaltc-neg",
    42052: "membership-gemma-2-9b-it-s11-evaltc-self",
    42062: "membership-gemma-2-9b-it-s12-evaltc-neg",
    42065: "membership-gemma-2-2b-it-s4-evaltc-self",
    42069: "membership-gemma-2-2b-it-s7-evaltc-neg",
    42077: "membership-gemma-2-2b-it-s3-evaltc-self",
    42085: "membership-gemma-2-2b-it-s5-evaltc-self",
}

LOG_DIR = Path("/datastor2/jdr/logs")

TASK_RE = re.compile(r"^\[(\d+)/10\]\s+(rosch-[\w-]+)")
CORR_RE = re.compile(r"correlation:\s*all\s*=\s*([-\d.eE+]+)")
SPEAR_RE = re.compile(r"spearman:\s*all\s*=\s*([-\d.eE+]+)")
DISC_RE = re.compile(r"disc_acc:\s*([-\d.eE+]+),\s*disc_roc:\s*([-\d.eE+]+)")
GEN_RE = re.compile(r"^gen_roc:\s*([-\d.eE+]+)")


def parse_log(jobid: int) -> dict[str, dict[str, float]]:
    out = LOG_DIR / f"{jobid}.out"
    if not out.is_file():
        return {}
    rows: dict[str, dict[str, float]] = {}
    cur_task = None
    with out.open() as f:
        for line in f:
            m = TASK_RE.match(line)
            if m:
                cur_task = m.group(2)
                rows.setdefault(cur_task, {})
                continue
            if cur_task is None:
                continue
            d = rows[cur_task]
            if (m := CORR_RE.search(line)):
                d["corr_all"] = float(m.group(1))
            elif (m := SPEAR_RE.search(line)):
                d["spear_all"] = float(m.group(1))
            elif (m := DISC_RE.search(line)):
                d["disc_acc"] = float(m.group(1))
                d["disc_roc"] = float(m.group(2))
            elif (m := GEN_RE.match(line)):
                d["gen_roc"] = float(m.group(1))
    return rows


def stats(vals: list[float]) -> tuple[float, float]:
    n = len(vals)
    if n == 0:
        return (float("nan"), float("nan"))
    mean = sum(vals) / n
    if n == 1:
        return (mean, 0.0)
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    se = math.sqrt(var / n)
    return (mean, se)


def main():
    METRICS = ["gen_roc", "disc_roc", "disc_acc", "corr_all", "spear_all"]
    print("# Rosch metrics salvaged from crashed eval logs (OSError 36)\n")
    print("Each cell ran 10 rosch test tasks; metrics printed before the")
    print("CSV write that crashed. mean ± stderr across the 10 tasks.\n")

    summary_rows = []
    for jobid, label in JOBS.items():
        rows = parse_log(jobid)
        if not rows:
            print(f"## job {jobid}  ({label})\n   NO LOG FOUND\n")
            continue
        print(f"## job {jobid}  ({label})")
        print(f"   tasks parsed: {len(rows)}/10")
        # per-task mini-table
        print("\n| task | gen_roc | disc_roc | disc_acc | corr_all | spear_all |")
        print("|---|---|---|---|---|---|")
        for t in sorted(rows):
            d = rows[t]
            cells = [f"{d.get(m, float('nan')):.4f}" for m in METRICS]
            print(f"| {t} | {' | '.join(cells)} |")
        # aggregate
        agg = {m: [d[m] for d in rows.values() if m in d] for m in METRICS}
        print("\n**Aggregate (mean ± stderr across", len(rows), "tasks):**\n")
        agg_cells = []
        for m in METRICS:
            mn, se = stats(agg[m])
            agg_cells.append(f"{mn:.4f} ± {se:.4f}")
        print("| " + " | ".join(METRICS) + " |")
        print("|---" * len(METRICS) + "|")
        print("| " + " | ".join(agg_cells) + " |\n")
        summary_rows.append((jobid, label, agg_cells))

    # cross-cell summary
    print("\n# Cross-cell summary (mean ± stderr per metric, per cell)\n")
    print("| jobid | label | " + " | ".join(METRICS) + " |")
    print("|---" * (2 + len(METRICS)) + "|")
    for jobid, label, cells in summary_rows:
        print(f"| {jobid} | {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
