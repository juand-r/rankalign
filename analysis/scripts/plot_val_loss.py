#!/usr/bin/env python3
"""Plot held-out val-loss curves across checkpoints for gemma-2-9b-it.

Reads the two val-loss CSVs produced by scripts/compute_val_loss.py:
  - analysis/tables/val_loss_gemma_ifeval.csv
        ifeval-concat task, evaluated on the held-out 50% of ifeval
  - analysis/tables/val_loss_gemma_mem.csv
        membership-sans-rosch-v0 task, evaluated on the rosch-all
        aggregated test set (the membership held-out distribution is rosch)

Each CSV has 5 settings (s1, s2, s3, s4, s7) x 4 checkpoints
(base, epoch0, epoch1, epoch2). The script produces a single figure with
2 rows (ifeval, membership/rosch) x 3 columns (preference loss, NLL-G,
NLL-V). The 'base' row is identical across settings (same untrained
gemma-2-9b-it), so it appears as a single horizontal reference line.

Output: analysis/plots/val_loss_curves.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
PLOTS_DIR = REPO / "analysis" / "plots"
TABLES_DIR = REPO / "analysis" / "tables"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SETTING_LABELS = {
    "s1": "s1 (SFT)",
    "s2": "s2 (RankAlign)",
    "s3": "s3 (Ours w/o TC)",
    "s4": "s4 (Ours)",
    "s7": "s7 (Ours, neg TC)",
}
SETTING_ORDER = ["s1", "s2", "s3", "s4", "s7"]
SETTING_COLORS = {
    "s1": "#888888",
    "s2": "#d62728",
    "s3": "#1f77b4",
    "s4": "#2ca02c",
    "s7": "#9467bd",
}

# Map raw checkpoint names -> integer x-axis index, with a friendly tick label.
CKPT_ORDER = ["base", "epoch0", "epoch1", "epoch2"]
CKPT_X = {name: i for i, name in enumerate(CKPT_ORDER)}
CKPT_TICK_LABELS = ["base", "ep0", "ep1", "ep2"]

# (column_name_in_csv, panel_title, y_label, y_scale)
METRIC_PANELS = [
    ("preference_loss",      "Preference loss", "preference loss", "linear"),
    ("nll_generator_loss",   "NLL-G",           "NLL-G",           "log"),
    ("nll_validator_loss",   "NLL-V",           "NLL-V",           "linear"),
]


def _plot_one_panel(ax, df, metric_col, title, ylabel, yscale):
    """Plot one (row, col) panel: settings overlaid as lines vs checkpoint."""
    base_df = df[df["checkpoint"] == "base"]
    if not base_df.empty:
        base_val = float(base_df[metric_col].iloc[0])
        ax.axhline(
            base_val,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
            label=f"base ({base_val:.2g})",
            zorder=1,
        )

    for setting in SETTING_ORDER:
        sub = df[df["setting"] == setting].copy()
        if sub.empty:
            continue
        sub["x"] = sub["checkpoint"].map(CKPT_X)
        sub = sub.sort_values("x")
        ax.plot(
            sub["x"].values,
            sub[metric_col].values,
            marker="o",
            color=SETTING_COLORS[setting],
            label=SETTING_LABELS[setting],
            linewidth=1.6,
            zorder=3,
        )

    ax.set_xticks(list(CKPT_X.values()))
    ax.set_xticklabels(CKPT_TICK_LABELS)
    ax.set_xlabel("checkpoint")
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def main():
    ifeval_csv = TABLES_DIR / "val_loss_gemma_ifeval.csv"
    mem_csv = TABLES_DIR / "val_loss_gemma_mem.csv"
    if not ifeval_csv.exists():
        raise FileNotFoundError(ifeval_csv)
    if not mem_csv.exists():
        raise FileNotFoundError(mem_csv)

    ifeval_df = pd.read_csv(ifeval_csv)
    mem_df = pd.read_csv(mem_csv)

    print(f"  ifeval rows: {len(ifeval_df)}  (settings: {sorted(ifeval_df['setting'].unique())})")
    print(f"  membership rows: {len(mem_df)}  (settings: {sorted(mem_df['setting'].unique())})")

    n_rows = 2
    n_cols = len(METRIC_PANELS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.2 * n_rows))

    rows = [
        ("Gemma / IFEval (held-out 50% of ifeval)", ifeval_df),
        ("Gemma / Membership (eval on rosch-all)", mem_df),
    ]

    for r, (row_title, df) in enumerate(rows):
        for c, (metric_col, panel_title, ylabel, yscale) in enumerate(METRIC_PANELS):
            ax = axes[r, c]
            _plot_one_panel(ax, df, metric_col, panel_title, ylabel, yscale)
            if c == 0:
                ax.set_ylabel(f"{row_title}\n{ylabel}", fontsize=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, -0.01),
        fontsize=9,
        frameon=False,
    )

    fig.suptitle(
        "Held-out validation loss across checkpoints (gemma-2-9b-it)",
        fontsize=13,
        y=1.00,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])

    out_path = PLOTS_DIR / "val_loss_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
