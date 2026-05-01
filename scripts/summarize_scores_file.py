#!/usr/bin/env python3
"""
Quick per-file score summarizer.

Usage:
    python scripts/summarize_scores_file.py outputs/scores_*.csv
    python scripts/summarize_scores_file.py outputs/scores_model_task_test_log-odds_20260501.csv
    python scripts/summarize_scores_file.py --glob "outputs/scores_*humaneval*20260501*.csv"

Importable API:
    from summarize_scores_file import load_scores, compute_all_metrics, format_report

    df = load_scores("outputs/scores_something.csv")
    metrics = compute_all_metrics(df)
    print(format_report(metrics, filename="scores_something.csv"))

    # Multi-file aggregation:
    from summarize_scores_file import summarize_files
    combined = summarize_files(list_of_paths)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, accuracy_score


LABEL_COLUMNS = ["correct", "label", "gpt4_ground_truth"]

GEN_VARIANTS = [
    ("raw",        "gen_score"),
    ("tc",         "gen_score_typcorr"),
    ("lenorm",     "gen_score_lenorm"),
    ("tc+lenorm",  "gen_score_typcorr_lenorm"),
]


def _find_label_col(df: pd.DataFrame) -> str | None:
    for col in LABEL_COLUMNS:
        if col in df.columns:
            return col
    return None


def _to_binary(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.map({"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0})


def load_scores(path: str | Path) -> pd.DataFrame:
    """Load a scores CSV, normalizing the label column to a binary int 'y'."""
    df = pd.read_csv(path)
    label_col = _find_label_col(df)
    if label_col is not None:
        df["y"] = _to_binary(df[label_col])
    return df


def compute_metrics_for_gen(
    gen: np.ndarray, val: np.ndarray, y: np.ndarray, val_threshold: float = 0.0
) -> dict:
    """Core metric computation for one generator variant.

    Returns dict with: gen_roc, val_roc, val_acc, pearson, spearman, n.
    val_roc and val_acc are always the same regardless of gen variant, but
    included for completeness.
    """
    mask = ~(np.isnan(gen) | np.isnan(val) | np.isnan(y))
    n = int(mask.sum())
    if n < 4:
        return dict(gen_roc=np.nan, val_roc=np.nan, val_acc=np.nan,
                    pearson=np.nan, spearman=np.nan, n=n)

    gen, val, y = gen[mask], val[mask], y[mask].astype(int)

    if len(set(y)) < 2:
        return dict(gen_roc=np.nan, val_roc=np.nan, val_acc=np.nan,
                    pearson=np.nan, spearman=np.nan, n=n)

    try:
        gen_roc = roc_auc_score(y, gen)
    except Exception:
        gen_roc = np.nan
    try:
        val_roc = roc_auc_score(y, val)
    except Exception:
        val_roc = np.nan

    val_acc = accuracy_score(y, (val > val_threshold).astype(int))

    try:
        pearson_r, _ = pearsonr(gen, val)
    except Exception:
        pearson_r = np.nan
    try:
        spearman_r, _ = spearmanr(gen, val)
    except Exception:
        spearman_r = np.nan

    return dict(gen_roc=gen_roc, val_roc=val_roc, val_acc=val_acc,
                pearson=pearson_r, spearman=spearman_r, n=n)


def compute_all_metrics(
    df: pd.DataFrame, val_threshold: float = 0.0
) -> dict:
    """Compute metrics for all generator-score variants present in df.

    Returns a dict with keys like:
        "raw": {gen_roc, val_roc, val_acc, pearson, spearman, n}
        "tc":  {gen_roc, ...}
        ...
    Plus top-level "n_total", "n_positive", "n_negative".
    """
    if "y" not in df.columns:
        raise ValueError("DataFrame has no recognized label column. "
                         f"Expected one of: {LABEL_COLUMNS}")
    if "val_score" not in df.columns:
        raise ValueError("DataFrame has no 'val_score' column.")

    val = df["val_score"].values.astype(float)
    y = df["y"].values.astype(float)

    result = {
        "n_total": len(df),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
    }

    for variant_name, col_name in GEN_VARIANTS:
        if col_name not in df.columns:
            continue
        gen = df[col_name].values.astype(float)
        if np.isnan(gen).all():
            continue
        result[variant_name] = compute_metrics_for_gen(gen, val, y, val_threshold)

    return result


def format_report(metrics: dict, filename: str = "") -> str:
    """Format metrics dict into a human-readable report string."""
    lines = []
    if filename:
        lines.append(f"File: {filename}")
    lines.append(f"  Rows: {metrics['n_total']}  "
                 f"(pos={metrics['n_positive']}, neg={metrics['n_negative']})")
    lines.append("")

    header = f"  {'variant':<14} {'gen_roc':>8} {'val_roc':>8} {'val_acc':>8} {'pearson':>8} {'spearman':>9}   {'n':>5}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for variant_name, _ in GEN_VARIANTS:
        if variant_name not in metrics:
            continue
        m = metrics[variant_name]
        lines.append(
            f"  {variant_name:<14} "
            f"{m['gen_roc']:8.4f} {m['val_roc']:8.4f} {m['val_acc']:8.4f} "
            f"{m['pearson']:8.4f} {m['spearman']:9.4f}   {m['n']:5d}"
        )

    return "\n".join(lines)


def summarize_files(paths: list[Path], quiet: bool = False) -> pd.DataFrame:
    """Summarize multiple score files. Returns a DataFrame of all metrics.

    Each row = one (file, gen_variant) combination.
    """
    rows = []
    for path in paths:
        try:
            df = load_scores(path)
        except Exception as e:
            if not quiet:
                print(f"SKIP {path.name}: {e}", file=sys.stderr)
            continue

        try:
            metrics = compute_all_metrics(df)
        except ValueError as e:
            if not quiet:
                print(f"SKIP {path.name}: {e}", file=sys.stderr)
            continue

        for variant_name, _ in GEN_VARIANTS:
            if variant_name not in metrics:
                continue
            m = metrics[variant_name]
            rows.append({
                "file": path.name,
                "variant": variant_name,
                "n_total": metrics["n_total"],
                "n_pos": metrics["n_positive"],
                "n_neg": metrics["n_negative"],
                **m,
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize one or more scores_*.csv files.",
        epilog="Examples:\n"
               "  python scripts/summarize_scores_file.py outputs/scores_*.csv\n"
               "  python scripts/summarize_scores_file.py --glob 'outputs/scores_*humaneval*2026*.csv'\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("files", nargs="*", type=Path, help="Score CSV files to summarize")
    parser.add_argument("--glob", type=str, default=None,
                        help="Glob pattern to find score files (alternative to positional args)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Save combined metrics to this CSV path")
    parser.add_argument("--compact", action="store_true",
                        help="Compact table output (one row per file) instead of per-file reports")
    args = parser.parse_args()

    paths = list(args.files) if args.files else []
    if args.glob:
        import glob as glob_mod
        paths.extend(Path(p) for p in sorted(glob_mod.glob(args.glob)))

    if not paths:
        parser.error("No files provided. Pass score CSVs as arguments or use --glob.")

    if args.compact or len(paths) > 5:
        combined = summarize_files(paths)
        if combined.empty:
            print("No metrics computed from any file.")
            return

        pd.set_option("display.float_format", "{:.4f}".format)
        pd.set_option("display.width", 240)
        pd.set_option("display.max_columns", 20)
        pd.set_option("display.max_rows", 500)
        pd.set_option("display.max_colwidth", 80)

        print(f"\n{len(paths)} file(s), {len(combined)} rows of metrics\n")
        print(combined.to_string(index=False))

        if args.csv:
            combined.to_csv(args.csv, index=False)
            print(f"\nSaved to [{args.csv}]({args.csv})")
    else:
        for path in paths:
            try:
                df = load_scores(path)
                metrics = compute_all_metrics(df)
                print(format_report(metrics, filename=path.name))
                print()
            except Exception as e:
                print(f"ERROR {path.name}: {e}\n", file=sys.stderr)

        if args.csv:
            combined = summarize_files(paths)
            if not combined.empty:
                combined.to_csv(args.csv, index=False)
                print(f"Saved to [{args.csv}]({args.csv})")


if __name__ == "__main__":
    main()
