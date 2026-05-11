"""Plot gsm8k-v1 composition by model and strategy (gsm8k-v2 version of
humaneval_v1_model_strategy_composition.py).

Outputs:
  - results/gsm8k_v2_model_composition_train_test_pos_neg.png
  - results/gsm8k_v2_strategy_composition_train_test_pos_neg.png
  - results/gsm8k_v2_model_split_label_shares.csv
  - results/gsm8k_v2_strategy_split_label_shares.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLS = {"correct", "model", "strategy"}


def normalize_label(x: str) -> str:
    s = str(x).strip().lower()
    if s in {"yes", "1", "true"}:
        return "positive"
    if s in {"no", "0", "false"}:
        return "negative"
    return "unknown"


def load_v1_data(input_dir: Path) -> pd.DataFrame:
    rows = []
    for fp in sorted(input_dir.glob("*.csv")):
        if fp.name == "train_old.csv":
            continue
        split = "train" if fp.name == "train.csv" else "test"
        df = pd.read_csv(fp, usecols=list(REQUIRED_COLS))
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"{fp} missing required columns: {sorted(missing)}")
        df = df.copy()
        df["split"] = split
        df["label"] = df["correct"].map(normalize_label)
        df["source_file"] = fp.name
        rows.append(df)
    if not rows:
        raise ValueError(f"No CSV files found in {input_dir}")
    out = pd.concat(rows, ignore_index=True)
    out = out[out["label"].isin(["positive", "negative"])].copy()
    return out


def composition_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    counts = (
        df.groupby(["split", "label", group_col], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    counts["label_total"] = counts.groupby(["split", "label"])["count"].transform("sum")
    counts["share"] = counts["count"] / counts["label_total"]
    return counts.sort_values(["split", "label", "count"], ascending=[True, True, False])


def plot_stacked_composition(
    comp: pd.DataFrame,
    group_col: str,
    out_path: Path,
    title: str,
) -> None:
    categories = (
        comp.groupby(group_col)["count"].sum().sort_values(ascending=False).index.tolist()
    )
    splits = ["train", "test"]
    labels = ["positive", "negative"]
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(categories))))[: len(categories)]
    color_map = {cat: colors[i] for i, cat in enumerate(categories)}

    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(categories) * 0.35)))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, split in zip(axes, splits):
        y = np.arange(len(labels))
        left = np.zeros(len(labels))
        for cat in categories:
            vals = []
            for label in labels:
                row = comp[
                    (comp["split"] == split)
                    & (comp["label"] == label)
                    & (comp[group_col] == cat)
                ]
                vals.append(float(row["share"].iloc[0]) * 100 if len(row) else 0.0)
            vals = np.array(vals)
            ax.barh(y, vals, left=left, color=color_map[cat], label=cat)
            left += vals

        for i, label in enumerate(labels):
            total = int(
                comp[(comp["split"] == split) & (comp["label"] == label)]["count"].sum()
            )
            ax.text(101.0, i, f"n={total}", va="center", fontsize=10)

        ax.set_xlim(0, 108)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Composition within label (%)")
        ax.set_title(f"{split.capitalize()} split")
        ax.grid(axis="x", alpha=0.2)

    handles, labels_leg = axes[1].get_legend_handles_labels()
    uniq = dict(zip(labels_leg, handles))
    fig.legend(
        uniq.values(),
        uniq.keys(),
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
        frameon=False,
    )
    plt.tight_layout(rect=[0, 0, 0.86, 0.95])
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def parse_param_size_b(model_name: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", model_name)
    if match:
        return float(match.group(1))
    return None


def is_large_model(model_name: str) -> bool:
    m = model_name.lower()
    if ("gpt-5" in m or "gpt-4" in m) and "mini" not in m:
        return True
    if "claude" in m:
        return True
    if "deepseek-v3" in m:
        return True
    if "qwen3-coder" in m:
        return True
    size_b = parse_param_size_b(model_name)
    return bool(size_b is not None and size_b >= 20.0)


def summarize_balance(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["split", "label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    summary["split_total"] = summary.groupby("split")["count"].transform("sum")
    summary["share"] = summary["count"] / summary["split_total"]
    return summary.sort_values(["split", "label"])


def summarize_large_model_representation(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["is_large_model"] = data["model"].map(is_large_model)
    out = (
        data.groupby(["split", "label", "is_large_model"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    out["label_total"] = out.groupby(["split", "label"])["count"].transform("sum")
    out["share"] = out["count"] / out["label_total"]
    return out.sort_values(["split", "label", "is_large_model"], ascending=[True, True, False])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="data/gsm8k/v2/test",
        help="Directory with train.csv + gsm8k_test_*.csv",
    )
    parser.add_argument("--output-dir", default="results", help="Where to write outputs")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_v1_data(input_dir)
    balance = summarize_balance(df)
    model_comp = composition_table(df, "model")
    strategy_comp = composition_table(df, "strategy")
    large_rep = summarize_large_model_representation(df)

    model_comp.to_csv(output_dir / "gsm8k_v2_model_split_label_shares.csv", index=False)
    strategy_comp.to_csv(output_dir / "gsm8k_v2_strategy_split_label_shares.csv", index=False)
    balance.to_csv(output_dir / "gsm8k_v2_split_balance.csv", index=False)
    large_rep.to_csv(output_dir / "gsm8k_v2_large_model_representation.csv", index=False)

    plot_stacked_composition(
        model_comp,
        group_col="model",
        out_path=output_dir / "gsm8k_v2_model_composition_train_test_pos_neg.png",
        title="gsm8k-v1 composition by model\n(positive vs negative, train vs test)",
    )
    plot_stacked_composition(
        strategy_comp,
        group_col="strategy",
        out_path=output_dir / "gsm8k_v2_strategy_composition_train_test_pos_neg.png",
        title="gsm8k-v1 composition by strategy\n(positive vs negative, train vs test)",
    )

    print("Wrote:")
    print(f"- {output_dir / 'gsm8k_v2_model_composition_train_test_pos_neg.png'}")
    print(f"- {output_dir / 'gsm8k_v2_strategy_composition_train_test_pos_neg.png'}")
    print(f"- {output_dir / 'gsm8k_v2_model_split_label_shares.csv'}")
    print(f"- {output_dir / 'gsm8k_v2_strategy_split_label_shares.csv'}")
    print(f"- {output_dir / 'gsm8k_v2_split_balance.csv'}")
    print(f"- {output_dir / 'gsm8k_v2_large_model_representation.csv'}")


if __name__ == "__main__":
    main()
