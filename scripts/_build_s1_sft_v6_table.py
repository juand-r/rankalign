#!/usr/bin/env python3
"""Standalone metrics table for the s1 (SFT-lo) baseline, gemma-4-31B-it,
humaneval-v2.1correct-upper, from v6 scores CSVs.

s1 = SFT-lo (pref0.0 nllv1.0 nllg1.0 labelonly0.1; no consistency-ft, no fsx, no TC).
Only base-typicality eval variants were run: basetyp -> PMI base, basetypneg -> Neg base.
PMI self / Neg self were never computed. Scores are v6 (delta0.15-epoch1).

Aggregation matches scripts/_build_humaneval_cu_v7_table.py: per-problem metric,
then mean +/- SE (ddof=1) * 100 across problems.
"""
from __future__ import annotations
import sys
import math
import glob
from pathlib import Path
import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from summarize_scores_file import load_scores, compute_all_metrics  # noqa: E402

SCORES_DIR = SCRIPTS.parent / "outputs_gemma4_from_pod-v7" / "s1_sft_v6"
OUT_MD = SCRIPTS.parent / "docs" / "humaneval_cu_s1_sft_v6_tables_2026-05-25.md"

METRICS = ["gen_roc", "pearson", "spearman", "val_roc", "val_acc"]
METRIC_LABELS = {
    "gen_roc": "GenROC x 100",
    "pearson": "Pearson(gen, val) x 100",
    "spearman": "Spearman(gen, val) x 100",
    "val_roc": "ValROC x 100",
    "val_acc": "ValAcc x 100",
}
# (column label, file glob, variant key)
COLUMNS = [
    ("Raw", "scores_basetyp-*.csv", "raw"),
    ("PMI base", "scores_basetyp-*.csv", "tc"),
    ("Neg base", "scores_basetypneg-*.csv", "tc"),
]


def agg(files: list[str], variant: str, metric: str):
    vals = []
    for f in files:
        try:
            m = compute_all_metrics(load_scores(f))
        except Exception:
            continue
        if variant not in m:
            continue
        v = m[variant].get(metric)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        vals.append(float(v))
    n = len(vals)
    if n == 0:
        return None
    arr = np.array(vals) * 100.0
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return mean, se, n


def cell(res, expected: int = 82) -> str:
    if res is None:
        return "—"
    mean, se, n = res
    s = f"{mean:.2f} ± {se:.2f}" if not math.isnan(se) else f"{mean:.2f}"
    if n < expected:
        s += f" (n={n})"
    return s


def main() -> None:
    globs = {g: sorted(glob.glob(str(SCORES_DIR / g))) for _, g, _ in COLUMNS}
    L = []
    L.append("# humaneval-v2.1correct-upper x gemma-4-31B-it — s1 (SFT-lo) baseline [v6]")
    L.append("")
    L.append("Snapshot: **2026-05-25**")
    L.append("")
    L.append("**Experiment:** correct-upper | **Model:** gemma-4-31B-it | "
             "**Setting:** s1 = SFT-lo (`pref0.0 nllv1.0 nllg1.0 labelonly0.1`; "
             "no consistency-ft, no fsx, no TC) | **Dataset:** humaneval-v2.1correct-upper (82 problems)")
    L.append("")
    L.append("> **Caveats:** (1) These scores are **v6** (`delta0.15-epoch1`), not v7 "
             "(`delta2.14`). For a pure-SFT baseline the rankalign `delta` does not affect "
             "training, but it is a different run version — flag for the reader. "
             "(2) Only the **base-typicality** eval variants were run, so **PMI self / Neg self "
             "were never computed** for s1 (shown as `---`).")
    L.append("")
    L.append("Source: `outputs_gemma4_from_pod-v7/s1_sft_v6/` (164 CSVs = 82 basetyp + 82 basetypneg), "
             "from mll `/datastor2/jdr/rankalign/outputs_gemma4_from_pod`. Rebuild: "
             "`python scripts/_build_s1_sft_v6_table.py`.")
    L.append("")
    for metric in METRICS:
        c = {col: cell(agg(globs[g], variant, metric)) for col, g, variant in COLUMNS}
        L.append(f"## {METRIC_LABELS[metric]}")
        L.append("")
        L.append("| Method | Raw | PMI self | PMI base | Neg self | Neg base |")
        L.append("| --- | --- | --- | --- | --- | --- |")
        L.append(f"| 1 SFT labelonly 10% (s1, v6) | {c['Raw']} | --- | {c['PMI base']} | --- | {c['Neg base']} |")
        L.append("")
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"Wrote {OUT_MD}\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
