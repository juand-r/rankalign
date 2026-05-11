#!/usr/bin/env python3
"""Verify that the markdown summary tables match what summarize_scores_file.py
produces directly from the score CSVs. Catches:

  1. Stale markdown (md generated before some score CSVs landed).
  2. Bugs in build_quickiter_summary_tables.py (wrong rows / wrong cells).
  3. Bugs in the markdown extraction regex (cells missed or mis-indexed).
  4. Hand-edits to the markdown that drifted from the underlying data.

Pipeline (everything in one script, no implicit shared state):

    score CSVs in outputs-quickiter/
        ──[ summarize_scores_file.py, fresh run ]──> /tmp/check_*_long.csv
                                                          │
                                                          ▼
                                                   (ground truth)
                                                          │
                                                          ▼
    *_quickiter_summary*.md ──[ this script's md parser ]──> (md values)
                                                          │
                                                          ▼
                                                  cell-by-cell compare

Failures are printed with the exact (model, task, metric, variant_id,
eval_ref) cell, the CSV value, and the md value. Exits non-zero if any
mismatch (beyond float-rounding tolerance) is found, so this is safe to
wire into CI / pre-commit.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs-quickiter"
SUMMARIZE = ROOT / "scripts" / "summarize_scores_file.py"

# Markdown values are written as (raw × SCALE) with .2f, so round-trip error is
# at most 5e-3. Keep these in sync with the build scripts.
SCALE = 100
TOL = 1e-2

# (label, model, task, md_path, score_glob)
COMBOS = [
    ("rosch 2b",      "gemma-2-2b",     "rosch-furniture-and-bird",
        OUTPUTS / "rosch_quickiter_summary.md",
        str(OUTPUTS / "scores_*rosch-furniture-and-bird*.csv")),
    ("rosch 2b-it",   "gemma-2-2b-it",  "rosch-furniture-and-bird",
        OUTPUTS / "rosch_quickiter_summary_gemma-2-2b-it.md",
        str(OUTPUTS / "scores_*rosch-furniture-and-bird*.csv")),
    ("ambigqa 2b",    "gemma-2-2b",     "ambigqa-train-as-test",
        OUTPUTS / "ambigqa_quickiter_summary_gemma-2-2b.md",
        str(OUTPUTS / "scores_*ambigqa-train-as-test*.csv")),
    ("ambigqa 2b-it", "gemma-2-2b-it",  "ambigqa-train-as-test",
        OUTPUTS / "ambigqa_quickiter_summary_gemma-2-2b-it.md",
        str(OUTPUTS / "scores_*ambigqa-train-as-test*.csv")),
]

# Same signature → (id, label) map as build_quickiter_summary_tables.py.
SIG_MAP = {
    "full-completion_force-same-x":
        ("1", "RankAlign baseline"),
    "tc-self_full-completion_force-same-x":
        ("2", "+ offline self-TC"),
    "tc-self_full-completion_force-same-x_online-tc":
        ("3", "+ online self-TC"),
    "full-completion_force-same-x_online-pairs":
        ("4", "+ online pairs"),
    "tc-self_full-completion_force-same-x_online-pairs_online-tc":
        ("5", "+ both online (self)"),
    "full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x":
        ("6", "SFT (NLL all)"),
    "tc-neg_full-completion_force-same-x":
        ("7", "+ offline neg-TC"),
    "tc-neg_full-completion_force-same-x_online-tc":
        ("8", "+ online neg-TC"),
    "tc-neg_full-completion_force-same-x_online-pairs_online-tc":
        ("9", "+ both online (neg)"),
}
EVAL_REFS = ["self", "neg", "basetyp", "basetypneg"]
METRICS = ["gen_roc", "val_roc", "val_acc", "pearson"]
METRIC_TITLES = {
    "gen_roc": "Generator ROC-AUC",
    "val_roc": "Validator ROC-AUC",
    "val_acc": "Validator accuracy",
    "pearson": "Pearson(gen, validator)",
}


# ─── ground truth: (re-)summarize → CSV → load ──────────────────────────────

def regenerate_long_csv(score_glob: str, out_csv: Path) -> pd.DataFrame:
    """Run summarize_scores_file.py with the given glob, returning the dataframe."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(SUMMARIZE), "--glob", score_glob, "--csv", str(out_csv)]
    print(f"  [run] {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        raise SystemExit(f"summarize_scores_file.py failed for {score_glob}")
    return pd.read_csv(out_csv)


def parse_csv_filename(fname: str, model: str, task: str):
    """Replicates the parser in build_quickiter_summary_tables.py.

    Returns (variant_id, train_label, eval_ref) or None if the filename
    doesn't belong to (model, task) or doesn't match a known signature.
    """
    m = re.escape(model)
    t = re.escape(task)
    # Base model:
    base_self_re = re.compile(
        rf"^scores_self-v6-google_{m}_{t}_test_log-odds_tc_\d+\.csv$"
    )
    base_neg_re = re.compile(
        rf"^scores_neg-v6-google_{m}_{t}_test_log-odds_tc_\d+\.csv$"
    )
    if base_self_re.match(fname):
        return "0", f"Base HF ({model})", "self"
    if base_neg_re.match(fname):
        return "0", f"Base HF ({model})", "neg"
    fine_re = re.compile(
        rf"^scores_(basetypneg|basetyp|neg|self)-v6-google_{m}-delta0\.15-epoch2_"
        rf"{t}-all_d2g_random_alpha1\.0_(.+)_{t}_test_log-odds_tc_\d+\.csv$"
    )
    g = fine_re.match(fname)
    if not g:
        return None
    eval_ref, sig = g.group(1), g.group(2)
    if sig not in SIG_MAP:
        return None
    vid, label = SIG_MAP[sig]
    return vid, label, eval_ref


def csv_truth(df_long: pd.DataFrame, model: str, task: str) -> dict:
    """{(variant_id, eval_ref): {metric: value}} from the variant=='tc' rows."""
    sub = df_long[df_long["variant"] == "tc"]
    truth: dict[tuple, dict] = {}
    for _, r in sub.iterrows():
        parsed = parse_csv_filename(r["file"], model, task)
        if parsed is None:
            continue
        vid, _label, eval_ref = parsed
        truth[(vid, eval_ref)] = {
            "gen_roc":  float(r["gen_roc"]) * SCALE,
            "val_roc":  float(r["val_roc"]) * SCALE,
            "val_acc":  float(r["val_acc"]) * SCALE,
            "pearson":  float(r["pearson"]) * SCALE,
        }
    return truth


# ─── markdown extraction (same as scripts/_xtab.py, generalized) ────────────

def parse_md_table(md_path: Path) -> dict:
    """{(variant_id, eval_ref): {metric: value}} extracted from markdown.

    For each metric heading, walks consecutive lines starting with '|' and
    parses them as | # | trained model | self | neg | basetyp | basetypneg |.
    """
    text = md_path.read_text()
    lines = text.splitlines()

    out: dict[tuple, dict] = {}
    metric_for_title = {
        "Generator ROC-AUC": "gen_roc",
        "Validator ROC-AUC": "val_roc",
        "Validator accuracy": "val_acc",
        "Pearson": "pearson",
    }

    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("### "):
            heading = ln[4:].strip()
            metric_key = None
            for prefix, key in metric_for_title.items():
                if heading.startswith(prefix):
                    metric_key = key
                    break
            if metric_key is None:
                i += 1
                continue
            j = i + 1
            block = []
            while j < len(lines):
                if lines[j].startswith("|"):
                    block.append(lines[j])
                    j += 1
                elif block:
                    break
                else:
                    j += 1
            if len(block) < 3:
                i = j
                continue
            for row in block[2:]:  # skip header & separator
                cells = [c.strip() for c in row.strip("|").split("|")]
                if len(cells) < 6:
                    continue
                vid = cells[0]
                for ref, raw in zip(EVAL_REFS, cells[2:6]):
                    val = None if raw == "—" else float(raw)
                    out.setdefault((vid, ref), {})[metric_key] = val
            i = j
        else:
            i += 1
    return out


# ─── compare ────────────────────────────────────────────────────────────────

def compare(label: str, truth: dict, md: dict) -> list[str]:
    """Return a list of human-readable mismatch lines."""
    issues = []
    keys = set(truth) | set(md)
    n_compared = 0
    n_md_missing = 0
    n_csv_missing = 0
    n_mismatch = 0
    for key in sorted(keys):
        vid, eval_ref = key
        in_truth = key in truth
        in_md = key in md
        if in_truth and not in_md:
            n_csv_missing += 1
            issues.append(
                f"  [{label}] cell ({vid}, {eval_ref}) in CSV but missing from markdown"
            )
            continue
        if in_md and not in_truth:
            md_vals_present = {m: v for m, v in md[key].items() if v is not None}
            if md_vals_present:
                n_md_missing += 1
                issues.append(
                    f"  [{label}] cell ({vid}, {eval_ref}) in markdown "
                    f"with values {md_vals_present} but missing from CSV"
                )
            continue
        for metric in METRICS:
            t_val = truth[key].get(metric)
            m_val = md[key].get(metric)
            if t_val is None and m_val is None:
                continue
            if t_val is None or m_val is None:
                n_mismatch += 1
                issues.append(
                    f"  [{label}] {metric} ({vid}, {eval_ref}): "
                    f"CSV={t_val}  md={m_val}  (one is missing)"
                )
                continue
            n_compared += 1
            if abs(t_val - m_val) > TOL:
                n_mismatch += 1
                issues.append(
                    f"  [{label}] {metric} ({vid}, {eval_ref}): "
                    f"CSV={t_val:.6f}  md={m_val:.6f}  Δ={abs(t_val-m_val):.6f}"
                )
    summary = (
        f"  [{label}] compared {n_compared} numeric cells; "
        f"{n_mismatch} mismatch, {n_md_missing} extra-in-md, {n_csv_missing} extra-in-csv."
    )
    issues.insert(0, summary)
    return issues


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default=None,
                    help="Where to write the fresh long CSVs (default: tmpdir).")
    ap.add_argument("--keep-workdir", action="store_true",
                    help="Don't delete the workdir on exit (useful for inspection).")
    args = ap.parse_args()

    if args.workdir:
        wd = Path(args.workdir).resolve()
        wd.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        wd = Path(tempfile.mkdtemp(prefix="check_summary_"))
        cleanup = not args.keep_workdir

    print(f"workdir: {wd}")

    fresh_long_by_task: dict[str, pd.DataFrame] = {}
    for label, model, task, md_path, glob in COMBOS:
        if task not in fresh_long_by_task:
            csv_out = wd / f"long_{task}.csv"
            print(f"\n[refresh] {task}")
            fresh_long_by_task[task] = regenerate_long_csv(glob, csv_out)

    n_total_issues = 0
    print("\n=== consistency check (CSV from summarize_scores_file.py vs markdown) ===")
    for label, model, task, md_path, _glob in COMBOS:
        print(f"\n[{label}] model={model} task={task}")
        print(f"  md: {md_path.relative_to(ROOT)}")
        if not md_path.exists():
            print(f"  ERROR: markdown not found: {md_path}")
            n_total_issues += 1
            continue
        truth = csv_truth(fresh_long_by_task[task], model, task)
        md = parse_md_table(md_path)
        issues = compare(label, truth, md)
        for line in issues:
            print(line)
        n_total_issues += sum(1 for ln in issues[1:] if ln.strip())

    if cleanup:
        shutil.rmtree(wd, ignore_errors=True)
        print(f"\n(cleaned up workdir {wd})")
    else:
        print(f"\n(workdir kept at {wd})")

    print()
    if n_total_issues == 0:
        print("OK — every markdown cell agrees with the regenerated CSV.")
        return 0
    print(f"FAIL — {n_total_issues} discrepancy lines.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
