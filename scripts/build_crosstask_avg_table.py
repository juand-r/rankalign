#!/usr/bin/env python3
"""Average per-OOD-task quick-iter metrics across the 8 non-train rosch
categories and emit a single summary markdown.

Reads outputs-quickiter/rosch-furniture-and-bird-to-ood/quickiter_metrics_long_crosstask.csv
(the long CSV produced by summarize_scores_file.py over all 8 OOD score files),
computes mean and std per (variant_id, eval_ref) over the 8 OOD tasks, and
writes a markdown report mirroring build_quickiter_summary_tables.py but with
values aggregated across tasks.

All numeric cells are scaled by SCALE = 100 (percentage points), matching the
matched-task reports.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LONG_CSV = ROOT / "outputs-quickiter" / "rosch-furniture-and-bird-to-ood" / "quickiter_metrics_long_crosstask.csv"
OUT_MD   = ROOT / "outputs-quickiter" / "rosch-furniture-and-bird-to-ood" / "MEAN_across_8_OOD_tasks_gemma-2-2b.md"

MODEL      = "gemma-2-2b"
TRAIN_TASK = "rosch-furniture-and-bird"
OOD_TASKS  = [
    "rosch-carpenters-tool", "rosch-clothing", "rosch-fruit", "rosch-sport",
    "rosch-toy", "rosch-vegetable", "rosch-vehicle", "rosch-weapon",
]
SCALE = 100

SIG_MAP = {
    "full-completion_force-same-x":                                         ("1", "RankAlign baseline"),
    "tc-self_full-completion_force-same-x":                                 ("2", "+ offline self-TC"),
    "tc-self_full-completion_force-same-x_online-tc":                       ("3", "+ online self-TC"),
    "full-completion_force-same-x_online-pairs":                            ("4", "+ online pairs"),
    "tc-self_full-completion_force-same-x_online-pairs_online-tc":          ("5", "+ both online (self)"),
    "full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x":                 ("6", "SFT (NLL all)"),
    "tc-neg_full-completion_force-same-x":                                  ("7", "+ offline neg-TC"),
    "tc-neg_full-completion_force-same-x_online-tc":                        ("8", "+ online neg-TC"),
    "tc-neg_full-completion_force-same-x_online-pairs_online-tc":           ("9", "+ both online (neg)"),
}
NUMBERED_ORDER = [
    ("0", f"Base HF ({MODEL})"),
    ("1", "RankAlign baseline"),
    ("2", "+ offline self-TC"),
    ("3", "+ online self-TC"),
    ("4", "+ online pairs"),
    ("5", "+ both online (self)"),
    ("6", "SFT (NLL all)"),
    ("7", "+ offline neg-TC"),
    ("8", "+ online neg-TC"),
    ("9", "+ both online (neg)"),
]
EVAL_REFS = ["self", "neg", "basetyp", "basetypneg"]
METRICS_AND_TITLES = [
    ("gen_roc",  "Generator ROC-AUC — mean (std) across 8 OOD tasks — × 100"),
    ("val_roc",  "Validator ROC-AUC — mean (std) across 8 OOD tasks — × 100"),
    ("val_acc",  "Validator accuracy (thr 0) — mean (std) across 8 OOD tasks — × 100"),
    ("pearson",  "Pearson(gen, validator) — mean (std) across 8 OOD tasks — × 100"),
]


def parse_filename(fname: str):
    """Return (id, label, eval_ref, eval_task) or None."""
    m = re.match(
        r"^scores_(basetypneg|basetyp|neg|self)-v6-google_"
        + re.escape(MODEL)
        + r"-delta0\.15-epoch2_"
        + re.escape(TRAIN_TASK)
        + r"-all_d2g_random_alpha1\.0_(.+)_(rosch-[a-z\-]+)_test_log-odds_tc_\d+\.csv$",
        fname,
    )
    if m:
        eval_ref, sig, eval_task = m.group(1), m.group(2), m.group(3)
        if eval_task not in OOD_TASKS or sig not in SIG_MAP:
            return None
        num, label = SIG_MAP[sig]
        return num, label, eval_ref, eval_task
    m = re.match(
        r"^scores_(self|neg)-v6-google_"
        + re.escape(MODEL)
        + r"_(rosch-[a-z\-]+)_test_log-odds_tc_\d+\.csv$",
        fname,
    )
    if m:
        eval_ref, eval_task = m.group(1), m.group(2)
        if eval_task not in OOD_TASKS:
            return None
        return "0", f"Base HF ({MODEL})", eval_ref, eval_task
    return None


def main():
    df = pd.read_csv(LONG_CSV)
    df = df[df["variant"] == "tc"].copy()

    rows = []
    for _, r in df.iterrows():
        p = parse_filename(r["file"])
        if p is None:
            continue
        num, label, eval_ref, eval_task = p
        rows.append({
            "id": num, "train": label, "eval_ref": eval_ref, "eval_task": eval_task,
            "gen_roc": r["gen_roc"], "val_roc": r["val_roc"],
            "val_acc": r["val_acc"], "pearson":  r["pearson"],
        })
    long = pd.DataFrame(rows)
    if long.empty:
        raise SystemExit("No matching rows; check long CSV.")

    n_tasks_per_cell = (
        long.groupby(["id", "train", "eval_ref"])["eval_task"].nunique()
    )
    if (n_tasks_per_cell != 8).any():
        problem = n_tasks_per_cell[n_tasks_per_cell != 8]
        print("WARN: not every (id, eval_ref) cell has 8 tasks:")
        print(problem)

    parts = [
        f"# Cross-task mean across 8 OOD rosch tasks ({MODEL})\n",
        f"**Train task:** {TRAIN_TASK}-all (models trained here, {len(OOD_TASKS)} OOD eval tasks below).\n",
        f"**OOD tasks:** {', '.join(OOD_TASKS)}.\n",
        "**Validator:** log-odds (`--validator-log-odds`). "
        "**Generator column variant:** `tc` (TC-corrected gen score where applicable).\n",
        "**All numeric cells are raw values × 100** "
        "(percentage points for ROC/accuracy; 0–100 units for Pearson). "
        "Reported as **mean (std)** across the 8 OOD tasks.\n",
        f"Long-form metrics: [{LONG_CSV.name}]({LONG_CSV.name})\n",
    ]

    for metric, title in METRICS_AND_TITLES:
        agg = (
            long.groupby(["id", "train", "eval_ref"])[metric]
                .agg(["mean", "std", "count"])
                .reset_index()
        )
        wide_mean = agg.pivot_table(
            index=["id", "train"], columns="eval_ref", values="mean", aggfunc="first"
        )
        wide_std = agg.pivot_table(
            index=["id", "train"], columns="eval_ref", values="std", aggfunc="first"
        )
        for c in EVAL_REFS:
            if c not in wide_mean.columns:
                wide_mean[c] = np.nan
                wide_std[c]  = np.nan
        wide_mean = wide_mean[EVAL_REFS]
        wide_std  = wide_std[EVAL_REFS]

        present = set(wide_mean.index)
        out_rows = []
        for i, lab in NUMBERED_ORDER:
            if (i, lab) not in present:
                continue
            row = {"#": i, "trained model": lab}
            for c in EVAL_REFS:
                m = wide_mean.loc[(i, lab), c]
                s = wide_std.loc[(i, lab), c]
                if pd.isna(m):
                    row[c] = "—"
                else:
                    if pd.isna(s):
                        row[c] = f"{m * SCALE:.2f}"
                    else:
                        row[c] = f"{m * SCALE:.2f} ({s * SCALE:.2f})"
            out_rows.append(row)
        wdf = pd.DataFrame(out_rows)
        parts.append(f"### {title}\n")
        if wdf.empty:
            parts.append("_(no rows)_\n")
            continue
        parts.append("| " + " | ".join(wdf.columns) + " |")
        parts.append("| " + " | ".join(["---"] * len(wdf.columns)) + " |")
        for _, r in wdf.iterrows():
            parts.append("| " + " | ".join(str(r[c]) for c in wdf.columns) + " |")
        parts.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
