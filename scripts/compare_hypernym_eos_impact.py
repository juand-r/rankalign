#!/usr/bin/env python3
"""
Compare Hypernym evaluation metrics with vs without EOS scoring.

This script reuses summarize_scores.py metric computation so the compared metrics
are exactly the same set:
  - gen_roc, val_roc, val_acc
  - corr, corr_pos, corr_neg

Outputs:
  1) Detailed per-task comparison table (EOS vs non-EOS + deltas)
  2) Grouped summary table (mean deltas per model/config/eval variant)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from summarize_scores import (  # noqa: E402
    DEFAULT_OUTPUTS_DIR,
    DEFAULT_SUMMARIES_DIR,
    discover_and_summarize,
)


TARGET_MODELS = {
    "v6-google_gemma-2-2b",
    "v6-google_gemma-2-2b-it",
    "v6-google_gemma-2-9b-it",
}

PAIR_KEY_COLS = [
    "model",
    "task",
    "split",
    "self_tc",
    "neg_tc",
    "gpt2_tc",
    "basetyp_tc",
    "finetuned",
    "training_config",
    "eval_variant",
]

METRIC_COLS = [
    "gen_roc",
    "val_roc",
    "val_acc",
    "corr",
    "corr_pos",
    "corr_neg",
]


def _tc_mode(row: pd.Series) -> str:
    if row.get("basetyp_tc", False) and row.get("neg_tc", False):
        return "basetyp-neg"
    if row.get("basetyp_tc", False) and row.get("self_tc", False):
        return "basetyp-self"
    if row.get("neg_tc", False):
        return "neg"
    if row.get("self_tc", False):
        return "self"
    if row.get("gpt2_tc", False):
        return "gpt2"
    return "none"


def _prepare_pairs(summary: pd.DataFrame) -> pd.DataFrame:
    # Keep just the launched Hypernym EOS job family scope.
    filt = summary[
        summary["task"].str.startswith("hypernym-")
        & (summary["split"] == "test")
        & summary["model"].isin(TARGET_MODELS)
    ].copy()

    eos_df = filt[filt["include_eos"]].copy()
    noeos_df = filt[~filt["include_eos"]].copy()

    if eos_df.empty:
        raise RuntimeError("No EOS rows found for hypernym in summary.")
    if noeos_df.empty:
        raise RuntimeError("No non-EOS rows found for hypernym in summary.")

    eos_idx = eos_df.set_index(PAIR_KEY_COLS)
    noeos_idx = noeos_df.set_index(PAIR_KEY_COLS)

    paired = eos_idx.join(noeos_idx, how="inner", lsuffix="_eos", rsuffix="_noeos")
    if paired.empty:
        raise RuntimeError("No matched EOS/non-EOS row pairs found.")

    out = paired.reset_index()
    out["training_config"] = out["training_config"].fillna("")
    out["training_label"] = out["training_config"].replace("", "<base>")
    out["tc_mode"] = out.apply(_tc_mode, axis=1)

    for col in METRIC_COLS + ["n_samples"]:
        out[f"delta_{col}"] = out[f"{col}_eos"] - out[f"{col}_noeos"]

    return out


def _build_group_summary(detailed: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "model",
        "finetuned",
        "training_label",
        "tc_mode",
        "eval_variant",
    ]

    agg_cols = {}
    for col in METRIC_COLS + ["n_samples"]:
        agg_cols[f"{col}_eos"] = "mean"
        agg_cols[f"{col}_noeos"] = "mean"
        agg_cols[f"delta_{col}"] = "mean"

    grouped = (
        detailed.groupby(group_cols, dropna=False)
        .agg(agg_cols)
        .reset_index()
        .sort_values(group_cols)
    )

    # Number of task pairs represented in each grouped row.
    counts = (
        detailed.groupby(group_cols, dropna=False)
        .size()
        .rename("n_task_pairs")
        .reset_index()
    )
    grouped = grouped.merge(counts, on=group_cols, how="left")
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Hypernym EOS vs non-EOS summary metrics."
    )
    parser.add_argument(
        "--outputs-dir",
        default=str(DEFAULT_OUTPUTS_DIR),
        help="Directory containing scores_*.csv files.",
    )
    parser.add_argument(
        "--detailed-output",
        default=str(DEFAULT_SUMMARIES_DIR / "hypernym_eos_comparison_detailed.csv"),
        help="CSV path for per-task EOS vs non-EOS rows.",
    )
    parser.add_argument(
        "--summary-output",
        default=str(DEFAULT_SUMMARIES_DIR / "hypernym_eos_comparison_summary.csv"),
        help="CSV path for grouped mean-delta table.",
    )
    args = parser.parse_args()

    detailed_out = Path(args.detailed_output)
    summary_out = Path(args.summary_output)
    detailed_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    # Reuse summarize_scores.py pipeline to compute the same metrics.
    summary = discover_and_summarize(
        outputs_dir=args.outputs_dir,
        existing_filenames=None,
        file_pattern="scores_*hypernym*.csv",
        model_filter=None,
        epoch_filter=None,
    )

    if summary.empty:
        raise RuntimeError("No rows produced by summarize pipeline.")

    detailed = _prepare_pairs(summary)
    grouped = _build_group_summary(detailed)

    detailed.to_csv(detailed_out, index=False)
    grouped.to_csv(summary_out, index=False)

    print(f"Detailed rows: {len(detailed)}")
    print(f"Grouped rows: {len(grouped)}")
    print(f"Detailed table written to: {detailed_out}")
    print(f"Grouped table written to: {summary_out}")

    # Quick console view: largest absolute gen_roc deltas.
    preview = detailed.copy()
    preview["abs_delta_gen_roc"] = np.abs(preview["delta_gen_roc"])
    preview = preview.sort_values("abs_delta_gen_roc", ascending=False).head(12)
    show_cols = [
        "model",
        "task",
        "training_label",
        "tc_mode",
        "eval_variant",
        "gen_roc_noeos",
        "gen_roc_eos",
        "delta_gen_roc",
        "val_roc_noeos",
        "val_roc_eos",
        "delta_val_roc",
        "delta_val_acc",
        "delta_corr",
    ]
    print("\nTop |delta_gen_roc| pairs:")
    print(preview[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()

