#!/usr/bin/env python3
"""Aggregate mean(std) gen-ROC across the 10 rosch tasks for the older
membership-sans-rosch-v0 -> rosch eval cohorts in `outputs/` (May 1 and
May 4), and side-by-side with the canonical May 11 / May 13 cohorts in
`outputs-quickiter/`.

Usage:
    python scripts/aggregate_old_membership_to_rosch.py

Output:
    outputs-quickiter/membership-old-recipes-to-rosch/MEAN_across_10_rosch_tasks.md
    outputs-quickiter/membership-old-recipes-to-rosch/per_signature_long.csv
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_scores_file import compute_all_metrics, load_scores  # noqa: E402

OUT_DIR = ROOT / "outputs-quickiter" / "membership-old-recipes-to-rosch"

ROSCH_TASKS = [
    "rosch-bird", "rosch-carpenters-tool", "rosch-clothing", "rosch-fruit",
    "rosch-furniture", "rosch-sport", "rosch-toy", "rosch-vegetable",
    "rosch-vehicle", "rosch-weapon",
]

# scores_<eval_ref>-v6-google_<model>-delta0.15-epoch<E>_membership-sans-rosch-v0-all_<recipe>_<eval_task>_test_log-odds_tc_<date>.csv
FNAME_RE = re.compile(
    r"^scores_(self|neg|basetyp|basetypneg)-v6-google_"
    r"(?P<model>.+?)-delta0\.15-epoch(?P<epoch>\d+)_"
    r"membership-sans-rosch-v0-all_"
    r"(?P<recipe>.+?)_"
    r"(?P<eval_task>rosch-[a-z-]+)_test_log-odds_tc_"
    r"(?P<date>\d{8})\.csv$"
)


def parse(path: Path):
    m = FNAME_RE.match(path.name)
    if not m:
        return None
    eval_ref = m.group(1)
    return dict(
        path=path,
        eval_ref=eval_ref,
        model=m.group("model"),
        epoch=int(m.group("epoch")),
        recipe=m.group("recipe"),
        eval_task=m.group("eval_task"),
        date=m.group("date"),
    )


def collect_files() -> list[dict]:
    rows = []
    for sub in ("outputs", "outputs-quickiter"):
        d = ROOT / sub
        if not d.is_dir():
            continue
        for p in d.glob("scores_*membership-sans-rosch-v0-all*_rosch-*_test_log-odds_tc_*.csv"):
            parsed = parse(p)
            if parsed is None:
                continue
            parsed["source_dir"] = sub
            rows.append(parsed)
    return rows


def compute_per_file(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        try:
            df = load_scores(r["path"])
            metrics = compute_all_metrics(df)
        except Exception as e:
            print(f"SKIP {r['path'].name}: {e}", file=sys.stderr)
            continue
        if "tc" not in metrics:
            continue
        m = metrics["tc"]
        out.append({
            "model":     r["model"],
            "epoch":     r["epoch"],
            "recipe":    r["recipe"],
            "eval_ref":  r["eval_ref"],
            "eval_task": r["eval_task"],
            "date":      r["date"],
            "source":    r["source_dir"],
            "gen_roc":   m["gen_roc"],
            "val_roc":   m["val_roc"],
            "val_acc":   m["val_acc"],
            "pearson":   m["pearson"],
            "n_pos":     metrics["n_positive"],
            "n_neg":     metrics["n_negative"],
        })
    return pd.DataFrame(out)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean(std) across the 10 rosch tasks per (model, epoch, recipe, eval_ref)."""
    grp_cols = ["model", "epoch", "recipe", "eval_ref"]
    agg = (
        df.groupby(grp_cols)
          .agg(
              n_tasks=("eval_task", "nunique"),
              gen_roc_mean=("gen_roc", "mean"),
              gen_roc_std=("gen_roc",  "std"),
              val_roc_mean=("val_roc", "mean"),
              val_roc_std=("val_roc",  "std"),
              val_acc_mean=("val_acc", "mean"),
              val_acc_std=("val_acc",  "std"),
              pearson_mean=("pearson", "mean"),
              pearson_std=("pearson",  "std"),
              date=("date", "max"),
              source=("source", "first"),
          )
          .reset_index()
    )
    return agg


def fmt(mean: float, std: float) -> str:
    if pd.isna(mean):
        return "—"
    if pd.isna(std):
        return f"{100 * mean:.2f}"
    return f"{100 * mean:.2f} ({100 * std:.2f})"


def emit_markdown(agg: pd.DataFrame, out_path: Path) -> None:
    parts = [
        "# membership-sans-rosch-v0 → rosch — mean(std) across 10 rosch tasks",
        "",
        "Aggregated from every `scores_*membership-sans-rosch-v0-all*_rosch-*` CSV "
        "we have on disk, grouped by (model, epoch, training recipe, eval_ref). "
        "Each cell shows **mean × 100 (std × 100)** of the corresponding metric "
        "across the 10 rosch eval tasks. Generator score variant = `tc` "
        "(typicality-corrected gen score where applicable).",
        "",
        "Older cohorts live in `outputs/`; the canonical (May 11 / May 13) "
        "cohorts live in `outputs-quickiter/`.",
        "",
    ]
    cols = ["gen_roc", "val_roc", "val_acc", "pearson"]
    titles = {
        "gen_roc": "Generator ROC-AUC",
        "val_roc": "Validator ROC-AUC",
        "val_acc": "Validator accuracy (thr 0)",
        "pearson": "Pearson(gen, validator)",
    }

    grouped = agg.groupby(["model", "epoch"], sort=True)
    for (model, epoch), g in grouped:
        parts.append(f"## {model}, epoch{epoch}  (n_tasks per row shown in `n`)")
        parts.append("")
        for metric in cols:
            parts.append(f"### {titles[metric]} — × 100")
            parts.append("")
            parts.append("| date | source | recipe | eval_ref | n | value |")
            parts.append("|---|---|---|---|---|---|")
            sub = g.sort_values(["recipe", "eval_ref", "date"]).reset_index(drop=True)
            for _, row in sub.iterrows():
                val = fmt(row[f"{metric}_mean"], row[f"{metric}_std"])
                parts.append(
                    f"| {row['date']} | `{row['source']}` | "
                    f"`{row['recipe']}` | `{row['eval_ref']}` | "
                    f"{int(row['n_tasks'])} | {val} |"
                )
            parts.append("")
        parts.append("")
    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {out_path.relative_to(ROOT)}")


def main():
    rows = collect_files()
    print(f"Found {len(rows)} score CSVs across cohorts")
    if not rows:
        return
    df = compute_per_file(rows)
    print(f"Computed metrics on {len(df)} (file, eval_task) cells")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "per_signature_long.csv", index=False)
    print(f"Wrote {(OUT_DIR / 'per_signature_long.csv').relative_to(ROOT)}")

    agg = aggregate(df)
    emit_markdown(agg, OUT_DIR / "MEAN_across_10_rosch_tasks.md")


if __name__ == "__main__":
    main()
