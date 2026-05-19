"""Compute AUROC for exit-layer scores from run_v21_exit_eval.py output.

Reads the v21_exit_scores.jsonl produced by run_v21_exit_eval.py and computes
per-task and macro AUROC for:
  - logp_full (baseline)
  - logp_full / num_tokens (lenorm baseline)
  - logp_exit025/050/075 raw
  - ACD corrected: logp_full - beta * logp_exit{frac}  (for beta=1.0)
  - ACD lenorm: corrected / num_tokens

Outputs a markdown summary to stdout and saves a CSV.

Usage:
  python analyze_exit_layers.py \
      --input /path/to/v21_exit_scores.jsonl \
      [--beta 1.0] \
      [--output-csv /path/to/results.csv]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def load_scores(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def auroc_safe(y_true, y_score) -> float | None:
    """Return AUROC or None if only one class present."""
    labels = set(y_true)
    if len(labels) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def compute_macro_auroc(rows: list[dict], score_key: str, invert: bool = False) -> float:
    """Compute macro-averaged AUROC across tasks for a given score column.

    Higher score = more likely correct. If invert=True, negate scores first.
    """
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r)

    aurocs = []
    for task_id, task_rows in by_task.items():
        y_true = [1 if r["correct"].strip().capitalize() == "Yes" else 0 for r in task_rows]
        y_score = [r[score_key] for r in task_rows]
        if invert:
            y_score = [-s for s in y_score]
        a = auroc_safe(y_true, y_score)
        if a is not None:
            aurocs.append(a)

    return float(np.mean(aurocs)) if aurocs else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    rows = load_scores(args.input)
    print(f"Loaded {len(rows)} scored items")

    # Detect available exit fractions from column names
    sample = rows[0]
    exit_cols = sorted(k for k in sample if k.startswith("logp_exit"))
    exit_fracs = []
    for col in exit_cols:
        tag = col.replace("logp_exit", "")  # e.g. "025"
        frac = int(tag) / 100.0
        exit_fracs.append((frac, col))
    print(f"Exit fractions found: {[f for f, _ in exit_fracs]}")

    # Add derived columns
    beta = args.beta
    for r in rows:
        n = r["num_tokens"]
        r["logp_full_lenorm"] = r["logp_full"] / n if n > 0 else float("nan")
        for frac, col in exit_fracs:
            tag = f"{int(round(frac*100)):03d}"
            acd = r["logp_full"] - beta * r[col]
            r[f"acd{tag}"] = acd
            r[f"acd{tag}_lenorm"] = acd / n if n > 0 else float("nan")

    # Compute macro AUROC for all variants
    variants = [
        ("raw (full logp)", "logp_full"),
        ("lenorm (full/n)", "logp_full_lenorm"),
    ]
    for frac, col in exit_fracs:
        tag = f"{int(round(frac*100)):03d}"
        variants += [
            (f"exit{tag} raw", col),
            (f"ACD exit{tag} (β={beta})", f"acd{tag}"),
            (f"ACD exit{tag} lenorm", f"acd{tag}_lenorm"),
        ]

    print("\n" + "=" * 60)
    print(f"Macro-AUROC across {len(set(r['task_id'] for r in rows))} tasks")
    print("=" * 60)
    results = []
    for label, key in variants:
        auroc = compute_macro_auroc(rows, key)
        results.append((label, key, auroc))
        print(f"  {label:<35s}  {auroc:.4f}")

    # Per-task breakdown for full vs best ACD
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r)

    per_task = []
    for task_id in sorted(by_task.keys()):
        task_rows = by_task[task_id]
        n_correct = sum(1 for r in task_rows if r["correct"].strip().capitalize() == "Yes")
        n_wrong = len(task_rows) - n_correct
        y_true = [1 if r["correct"].strip().capitalize() == "Yes" else 0 for r in task_rows]

        task_result = {"task_id": task_id, "n_correct": n_correct, "n_wrong": n_wrong}
        for label, key, _ in results:
            a = auroc_safe(y_true, [r[key] for r in task_rows])
            task_result[key] = a if a is not None else float("nan")
        per_task.append(task_result)

    if args.output_csv:
        import csv
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        if per_task:
            fieldnames = list(per_task[0].keys())
            with open(out, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(per_task)
            print(f"\nPer-task AUROC saved to: {out}")

    # Summary table (LaTeX-friendly)
    print("\n" + "=" * 60)
    print("Summary (macro AUROC):")
    print("=" * 60)
    print(f"{'Variant':<35s}  {'AUROC':>6s}")
    print("-" * 44)
    for label, key, auroc in results:
        print(f"{label:<35s}  {auroc:.4f}")


if __name__ == "__main__":
    main()
