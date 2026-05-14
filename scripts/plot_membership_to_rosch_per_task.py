#!/usr/bin/env python3
"""Per-task grouped bar plots of gen-ROC for membership->rosch (epoch2),
ordered along the x-axis by item-overlap with the membership training pool
(highest overlap on the left, rosch-sport on the right).

Two output plots, by default:
    per_task_gen_roc_self.png — base, rankalign, sft, offline-self-TC
    per_task_gen_roc_neg.png  — base, rankalign, sft, offline-neg-TC

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
]
NEG_PLOT = [
    ("0", "neg",        "Base",           "tab:blue"),
    ("1", "neg",        "RankAlign",      "tab:red"),
    ("6", "neg",        "SFT",            "tab:purple"),
    ("7", "basetypneg", "Offline neg-TC", "tab:green"),
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
    ax.legend(loc="lower left", ncol=4, frameon=False)
    ax.grid(axis="y", alpha=0.3)

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
    """Single wide markdown table with 8 columns (4 self + 4 neg)."""
    headers = (
        ["task (overlap)"]
        + [f"{label} (self)" for _, _, label, _ in SELF_PLOT]
        + [f"{label} (neg)"  for _, _, label, _ in NEG_PLOT]
    )

    lines = [
        "# membership-sans-rosch-v0 (gemma-2-2b, epoch2) → rosch — "
        "per-task gen-ROC × 100",
        "",
        "Rows: 10 rosch tasks ordered by item-overlap with the membership "
        "training pool (high overlap on top, rosch-sport at the bottom).",
        "",
        "Columns: 4 self-eval variants (Base, RankAlign, SFT, Offline "
        "self-TC) followed by 4 neg-eval variants (Base, RankAlign, SFT, "
        "Offline neg-TC). The Base-self / Base-neg columns and SFT-self / "
        "SFT-neg columns are different metrics on the same checkpoint.",
        "",
        "**Bold = highest value in the row across all 8 columns.** Note this "
        "comparison mixes self and neg eval refs, which are different "
        "metrics — interpret \"row max\" as a quick visual read, not a "
        "rigorous comparison.",
        "",
        "Long-form metrics: "
        "[quickiter_metrics_long_membership_to_rosch.csv]"
        "(quickiter_metrics_long_membership_to_rosch.csv)",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for tname, overlap in TASKS_BY_OVERLAP:
        vals = []
        for vid, eval_ref, _label, _color in SELF_PLOT + NEG_PLOT:
            vals.append(get_value(long, vid, eval_ref, tname))
        idx_max = int(np.nanargmax(vals))
        cells = []
        for i, v in enumerate(vals):
            if np.isnan(v):
                cells.append("—")
            else:
                s = f"{v:.2f}"
                if i == idx_max:
                    s = f"**{s}**"
                cells.append(s)
        first = f"{tname} ({overlap}%)"
        lines.append("| " + " | ".join([first] + cells) + " |")

    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
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
              "self eval (offline self-TC vs RankAlign / SFT / Base)")
    make_plot(long, NEG_PLOT, OUT_DIR / "per_task_gen_roc_neg.png",
              "neg eval (offline neg-TC vs RankAlign / SFT / Base)")
    make_table(long, OUT_DIR / "per_task_gen_roc_table.md")


if __name__ == "__main__":
    main()
