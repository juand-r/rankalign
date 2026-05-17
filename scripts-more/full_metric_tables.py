#!/usr/bin/env python3
"""Pivot per_task_metrics.csv (from analyze_gemma4_tc.py) into the full set of
tables: one per (metric, TC-side), rows = Base / RankAlign / RankAlign+TC,
columns = raw / tc / lenorm / tc+lenorm, cells = mean±std over the 82 tasks.

Writes one CSV per (metric, side) plus a combined long CSV, and prints all
tables. Run AFTER analyze_gemma4_tc.py (which produces per_task_metrics.csv).

Usage:
  python scripts-more/full_metric_tables.py \
      --analysis-dir /datastor1/jdr/gv-gap/rankalign/outputs_gemma4_3epoch_e2/_analysis
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

METRICS = ["gen_roc", "val_roc", "val_acc", "pearson", "spearman"]
VARIANTS = ["raw", "tc", "lenorm", "tc+lenorm"]
ROWS = ["Base", "RankAlign", "RankAlign+TC"]

# (side -> {display row -> group name in per_task_metrics.csv})
SIDE_GROUPS = {
    "self": {"Base": "Base_self", "RankAlign": "RankAlign_self",
             "RankAlign+TC": "RankAlign+tc_self"},
    "neg":  {"Base": "Base_neg", "RankAlign": "RankAlign_neg",
             "RankAlign+TC": "RankAlign+negtc_neg"},
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis-dir", required=True,
                    help="dir containing per_task_metrics.csv (from analyze_gemma4_tc.py)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    adir = Path(args.analysis_dir)
    pt = pd.read_csv(adir / "per_task_metrics.csv")

    out_dir = adir / "metric_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    long_rows = []  # combined long-format record

    for metric in METRICS:
        for side, gmap in SIDE_GROUPS.items():
            mean_tbl = pd.DataFrame(index=ROWS, columns=VARIANTS, dtype=object)
            std_tbl = pd.DataFrame(index=ROWS, columns=VARIANTS, dtype=object)
            for row_label, group in gmap.items():
                for v in VARIANTS:
                    sel = pt[(pt.group == group) & (pt.variant == v)]
                    vals = sel[metric].astype(float)
                    m = vals.mean()
                    s = vals.std()
                    n = len(vals)
                    mean_tbl.loc[row_label, v] = round(m, 4)
                    std_tbl.loc[row_label, v] = round(s, 4)
                    long_rows.append(dict(metric=metric, side=side,
                                          setting=row_label, variant=v,
                                          mean=round(m, 4), std=round(s, 4),
                                          n=n))
            # combined mean±std display table
            disp = pd.DataFrame(index=ROWS, columns=VARIANTS, dtype=object)
            for r in ROWS:
                for v in VARIANTS:
                    disp.loc[r, v] = f"{mean_tbl.loc[r, v]:.4f} ± {std_tbl.loc[r, v]:.4f}"
            csv_path = out_dir / f"{metric}_{side}.csv"
            disp.to_csv(csv_path)
            note = ""
            if metric in ("val_roc", "val_acc"):
                note = ("   [val_* is validator-only — identical across raw/tc/"
                        "lenorm/tc+lenorm by construction]")
            print(f"\n=== {metric}  ({side}-TC) — mean ± std, 82 tasks ==={note}")
            print(disp.to_string())

    long = pd.DataFrame(long_rows)
    long_csv = out_dir / "all_metrics_long.csv"
    long.to_csv(long_csv, index=False)
    print(f"\nWrote per-(metric,side) CSVs + {long_csv}")
    print(f"({len(METRICS)} metrics × 2 sides = {len(METRICS)*2} tables in {out_dir})")


if __name__ == "__main__":
    main()
