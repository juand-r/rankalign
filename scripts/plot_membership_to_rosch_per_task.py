#!/usr/bin/env python3
"""Per-task grouped bar plots of gen-ROC for membership->rosch (epoch2),
ordered along the x-axis by item-overlap with the membership training pool
(highest overlap on the left, rosch-sport on the right).

Two output plots, by default:
    per_task_gen_roc_self.png — base, rankalign, sft, offline+online self-TC
    per_task_gen_roc_neg.png  — base, rankalign, sft, offline+online neg-TC

Both use the matched-side eval ref (self for the first; neg for the second),
except offline-{self,neg}-TC which uses basetyp / basetypneg respectively
(the canonical "best" eval ref per the 4-table layout).

Usage:
    python scripts/plot_membership_to_rosch_per_task.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parents[1]
LONG_CSV = (ROOT / "outputs-quickiter" / "membership-sans-rosch-v0-to-rosch"
            / "quickiter_metrics_long_membership_to_rosch.csv")
OUT_DIR  = ROOT / "outputs-quickiter" / "membership-sans-rosch-v0-to-rosch"

MODEL      = "gemma-2-2b"
TRAIN_TASK = "membership-sans-rosch-v0"
EPOCH      = 2

# Tasks ordered by item-overlap with the membership training pool (descending);
# rosch-sport at the end (it has only ~9% overlap).
TASKS_BY_OVERLAP = [
    ("rosch-bird",            89),
    ("rosch-carpenters-tool", 61),
    ("rosch-fruit",           60),
    ("rosch-vehicle",         56),
    ("rosch-furniture",       45),
    ("rosch-vegetable",       44),
    ("rosch-toy",             42),
    ("rosch-clothing",        36),
    ("rosch-weapon",          36),
    ("rosch-sport",            9),
]

# Each "plot config" is a list of (variant_id, eval_ref, label, color).
# These are the variants and eval refs the user asked for.
SELF_PLOT = [
    ("0", "self",    "Base",            "tab:blue"),
    ("1", "self",    "RankAlign",       "tab:red"),
    ("6", "self",    "SFT",             "tab:purple"),
    ("2", "basetyp", "Offline self-TC", "tab:green"),
    ("3", "self",    "Online self-TC",  "darkgreen"),
]
NEG_PLOT = [
    ("0", "neg",        "Base",           "tab:blue"),
    ("1", "neg",        "RankAlign",      "tab:red"),
    ("6", "neg",        "SFT",            "tab:purple"),
    ("7", "basetypneg", "Offline neg-TC", "tab:green"),
    ("8", "neg",        "Online neg-TC",  "darkgreen"),
]

# The wide markdown table keeps its original 4+4 layout (only one TC variant
# per side, matching the canonical "best per row" eval ref). Bar plots show
# both offline and online TC, but the table stays compact.
SELF_TABLE = SELF_PLOT[:4]
NEG_TABLE  = NEG_PLOT[:4]

# Heatmap configs: ALL relevant variants per side, in a logical order
# (reference -> SFT -> preference no-TC -> TC × pair-selection grid).
# Eval refs match the canonical "best per row" used in the 4-table layout
# (offline TC -> basetyp[neg]; everything else -> self/neg).
SELF_HEATMAP = [
    ("0", "self",    "Base"),
    ("6", "self",    "SFT"),
    ("1", "self",    "RankAlign"),
    ("2", "basetyp", "+ offline self-TC"),
    ("3", "self",    "+ online self-TC"),
]
NEG_HEATMAP = [
    ("0", "neg",        "Base"),
    ("6", "neg",        "SFT"),
    ("1", "neg",        "RankAlign"),
    ("7", "basetypneg", "+ offline neg-TC"),
    ("8", "neg",        "+ online neg-TC"),
]

SIG_MAP = {
    "full-completion_force-same-x":                                                    "1",
    "tc-self_full-completion_force-same-x":                                            "2",
    "tc-self_full-completion_force-same-x_online-tc":                                  "3",
    "full-completion_force-same-x_online-pairs":                                       "4",
    "tc-self_full-completion_force-same-x_online-pairs_online-tc":                     "5",
    "full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x":                            "6",
    "tc-neg_full-completion_force-same-x":                                             "7",
    "tc-neg_full-completion_force-same-x_online-tc":                                   "8",
    "tc-neg_full-completion_force-same-x_online-pairs_online-tc":                      "9",
}


def parse_filename(fname: str):
    """Return (variant_id, eval_ref, eval_task) or None."""
    base_re = re.compile(
        r"^scores_(self|neg)-v6-google_" + re.escape(MODEL) +
        r"_(rosch-[a-z-]+?)_test_log-odds_tc_\d+\.csv$"
    )
    m = base_re.match(fname)
    if m:
        return "0", m.group(1), m.group(2)
    fine_re = re.compile(
        r"^scores_(basetypneg|basetyp|neg|self)-v6-google_" +
        re.escape(MODEL) + rf"-delta0\.15-epoch{EPOCH}_" +
        re.escape(TRAIN_TASK) + r"-all_d2g_random_alpha1\.0_(.+?)_"
        r"(rosch-[a-z-]+?)_test_log-odds_tc_\d+\.csv$"
    )
    m = fine_re.match(fname)
    if not m:
        return None
    eval_ref, sig, eval_task = m.group(1), m.group(2), m.group(3)
    if sig not in SIG_MAP:
        return None
    return SIG_MAP[sig], eval_ref, eval_task


def load_long(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["variant"] == "tc"].copy()
    parsed = df["file"].apply(parse_filename)
    keep = parsed.notna()
    df = df[keep].copy()
    df["id"]        = parsed[keep].map(lambda t: t[0])
    df["eval_ref"]  = parsed[keep].map(lambda t: t[1])
    df["eval_task"] = parsed[keep].map(lambda t: t[2])
    return df


def make_plot(long: pd.DataFrame, plot_config, out_path: Path,
              title_suffix: str) -> None:
    task_names = [t for t, _ in TASKS_BY_OVERLAP]
    overlaps   = [o for _, o in TASKS_BY_OVERLAP]

    n_groups = len(task_names)
    n_bars   = len(plot_config)
    bar_w    = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(13, 5))
    x_centers = np.arange(n_groups)

    for i, (vid, eval_ref, label, color) in enumerate(plot_config):
        vals = []
        for tname in task_names:
            sub = long[(long["id"] == vid) &
                       (long["eval_ref"] == eval_ref) &
                       (long["eval_task"] == tname)]
            if len(sub) == 0:
                vals.append(np.nan)
            else:
                vals.append(float(sub["gen_roc"].iloc[0]) * 100)
        offsets = (i - (n_bars - 1) / 2) * bar_w
        ax.bar(x_centers + offsets, vals, bar_w, label=label,
               color=color, edgecolor="black", linewidth=0.4)

    ax.set_ylabel("Generator ROC-AUC (× 100)")
    ax.set_title(
        f"membership-sans-rosch-v0 ({MODEL}, epoch{EPOCH}) → rosch — "
        f"{title_suffix}\n(x-axis ordered by item-overlap of rosch task "
        f"with membership training pool)"
    )
    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        [f"{n}\n({o}%)" for n, o in zip(task_names, overlaps)],
        rotation=20, ha="right"
    )
    ax.set_ylim(40, 100)
    ax.axhline(50, color="gray", linewidth=0.5, linestyle="--")
    ax.legend(loc="lower left", ncol=n_bars, frameon=False)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path.relative_to(ROOT)}")


def make_heatmap(long: pd.DataFrame, config, out_path: Path,
                 title_suffix: str, vmin: float = 40, vmax: float = 100) -> None:
    """Per-task heatmap: rows = rosch tasks (in TASKS_BY_OVERLAP order),
    columns = variants in `config`. Cells annotated with the gen-ROC × 100
    value; row max gets a small mark in the corner."""
    task_names = [t for t, _ in TASKS_BY_OVERLAP]
    overlaps   = [o for _, o in TASKS_BY_OVERLAP]
    col_labels = [c[2] for c in config]

    grid = np.full((len(task_names), len(config)), np.nan)
    for i, tname in enumerate(task_names):
        for j, (vid, eval_ref, _label) in enumerate(config):
            grid[i, j] = get_value(long, vid, eval_ref, tname)

    fig, ax = plt.subplots(figsize=(1.45 * len(config) + 1.5,
                                    0.55 * len(task_names) + 1.2))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(task_names)))
    ax.set_yticklabels([f"{n} ({o}%)" for n, o in zip(task_names, overlaps)])

    for i in range(grid.shape[0]):
        row = grid[i]
        if np.all(np.isnan(row)):
            continue
        max_j = int(np.nanargmax(row))
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center",
                        color="white", fontsize=8)
                continue
            text_color = "white" if v < vmin + 0.55 * (vmax - vmin) else "black"
            txt = f"{v:.1f}"
            if j == max_j:
                txt = f"$\\bf{{{v:.1f}}}$"
            ax.text(j, i, txt, ha="center", va="center",
                    color=text_color, fontsize=9)

    ax.set_title(
        f"membership-sans-rosch-v0 ({MODEL}, epoch{EPOCH}) → rosch — "
        f"{title_suffix}\nGen-ROC × 100 (rows ordered by item-overlap with "
        f"the membership training pool; bold = row max)"
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Gen-ROC × 100")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path.relative_to(ROOT)}")


def get_value(long: pd.DataFrame, vid: str, eval_ref: str, eval_task: str):
    sub = long[(long["id"] == vid) &
               (long["eval_ref"] == eval_ref) &
               (long["eval_task"] == eval_task)]
    if len(sub) == 0:
        return float("nan")
    return float(sub["gen_roc"].iloc[0]) * 100


def make_table(long: pd.DataFrame, out_path: Path) -> None:
    """Wide HTML-in-markdown table: 1 + 4 + 4 columns with a self/neg
    multi-column top header row, ordinary 4-variant header below it.
    Bold = row max across all 8 numeric cells."""
    self_labels = [label for _, _, label, _ in SELF_TABLE]
    neg_labels  = [label for _, _, label, _ in NEG_TABLE]

    parts = [
        "# membership-sans-rosch-v0 (gemma-2-2b, epoch2) → rosch — "
        "per-task gen-ROC × 100",
        "",
        "Rows: 10 rosch tasks ordered by item-overlap with the membership "
        "training pool (high overlap on top, rosch-sport at the bottom).",
        "",
        "Columns: 4 self-eval variants and 4 neg-eval variants of the same "
        "set (Base, RankAlign, SFT, Offline {self,neg}-TC).",
        "",
        "**Bold = highest value in the row across all 8 columns.** This "
        "mixes self and neg eval refs (different metrics on the same "
        "checkpoint) — interpret as a visual read, not a rigorous "
        "comparison.",
        "",
        "Long-form metrics: "
        "[quickiter_metrics_long_membership_to_rosch.csv]"
        "(quickiter_metrics_long_membership_to_rosch.csv)",
        "",
        "<table>",
        "<thead>",
        '<tr><th rowspan="2">task (overlap)</th>'
        f'<th colspan="{len(self_labels)}">self</th>'
        f'<th colspan="{len(neg_labels)}">neg</th></tr>',
        "<tr>"
        + "".join(f"<th>{lab}</th>" for lab in self_labels)
        + "".join(f"<th>{lab}</th>" for lab in neg_labels)
        + "</tr>",
        "</thead>",
        "<tbody>",
    ]

    for tname, overlap in TASKS_BY_OVERLAP:
        vals = []
        for vid, eval_ref, _label, _color in SELF_TABLE + NEG_TABLE:
            vals.append(get_value(long, vid, eval_ref, tname))
        idx_max = int(np.nanargmax(vals))
        cells = []
        for i, v in enumerate(vals):
            if np.isnan(v):
                cells.append("<td>—</td>")
            else:
                s = f"{v:.2f}"
                if i == idx_max:
                    cells.append(f"<td><strong>{s}</strong></td>")
                else:
                    cells.append(f"<td>{s}</td>")
        row_label = f"{tname} ({overlap}%)"
        parts.append("<tr><td>" + row_label + "</td>" + "".join(cells) + "</tr>")

    parts += ["</tbody>", "</table>", ""]

    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {out_path.relative_to(ROOT)}")


def main():
    long = load_long(LONG_CSV)

    for tname in [t for t, _ in TASKS_BY_OVERLAP]:
        seen = (
            long[long["eval_task"] == tname]
                .groupby(["id", "eval_ref"]).size()
        )
        missing = []
        for plot in (SELF_PLOT, NEG_PLOT):
            for vid, eval_ref, label, _ in plot:
                if (vid, eval_ref) not in seen.index:
                    missing.append(f"{tname}/{label}({vid},{eval_ref})")
        if missing:
            print(f"WARN missing cells for {tname}: {missing}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_plot(long, SELF_PLOT, OUT_DIR / "per_task_gen_roc_self.png",
              "self eval (Base / RankAlign / SFT / offline + online self-TC)")
    make_plot(long, NEG_PLOT, OUT_DIR / "per_task_gen_roc_neg.png",
              "neg eval (Base / RankAlign / SFT / offline + online neg-TC)")
    make_table(long, OUT_DIR / "per_task_gen_roc_table.md")
    make_heatmap(long, SELF_HEATMAP,
                 OUT_DIR / "per_task_gen_roc_heatmap_self.png",
                 "self eval, all variants")
    make_heatmap(long, NEG_HEATMAP,
                 OUT_DIR / "per_task_gen_roc_heatmap_neg.png",
                 "neg eval, all variants")


if __name__ == "__main__":
    main()
