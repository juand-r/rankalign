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
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _table_format_4tables as tf4  # noqa: E402
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
    "full-completion_force-same-x":                                         ("1", "RankAlign+fsx"),
    "tc-self_full-completion_force-same-x":                                 ("2", "+ offline self-TC"),
    "tc-self_full-completion_force-same-x_online-tc":                       ("3", "+ online self-TC"),
    "full-completion_force-same-x_online-pairs":                            ("4", "+ online pairs"),
    "tc-self_full-completion_force-same-x_online-pairs_online-tc":          ("5", "+ both online (self)"),
    "full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x":                 ("6", "SFT+fsx (NLL all)"),
    "tc-neg_full-completion_force-same-x":                                  ("7", "+ offline neg-TC"),
    "tc-neg_full-completion_force-same-x_online-tc":                        ("8", "+ online neg-TC"),
    "tc-neg_full-completion_force-same-x_online-pairs_online-tc":           ("9", "+ both online (neg)"),
}
NUMBERED_ORDER = [
    ("0", f"Base HF ({MODEL})"),
    ("1", "RankAlign+fsx"),
    ("2", "+ offline self-TC"),
    ("3", "+ online self-TC"),
    ("4", "+ online pairs"),
    ("5", "+ both online (self)"),
    ("6", "SFT+fsx (NLL all)"),
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
        "Tables follow the canonical 4-table layout (see "
        "[docs/results_table_format.md](../../docs/results_table_format.md)): "
        "T1/T3 = baselines (Base HF + SFT) for the self/neg eval refs; "
        "T2/T4 = the 3×2 TC × pairs grids for self/neg. Cells in T2/T4 use the "
        "*best* eval ref per row (offline TC → `basetyp[neg]`; everything else "
        "→ `self`/`neg`). The (offline TC, online pairs) cell is always blank "
        "because that variant is not in the launcher.\n",
        f"Long-form metrics: [{LONG_CSV.name}]({LONG_CSV.name})\n",
    ]

    for metric, title in METRICS_AND_TITLES:
        agg = (
            long.groupby(["id", "eval_ref"])[metric]
                .agg(["mean", "std"])
                .reset_index()
        )
        val_map: dict[tuple[str, str], tuple[float, float]] = {}
        for _, r in agg.iterrows():
            val_map[(r["id"], r["eval_ref"])] = (r["mean"], r["std"])

        def get(vid: str, eval_ref: str, _vmap=val_map) -> str | None:
            v = _vmap.get((vid, eval_ref))
            if v is None:
                return None
            m, s = v
            if pd.isna(m):
                return None
            return tf4.fmt_mean_std(m, s, scale=SCALE)

        parts.append(tf4.emit_4_tables(get, f"{title} — × 100"))

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
