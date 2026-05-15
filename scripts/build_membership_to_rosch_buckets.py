#!/usr/bin/env python3
"""Build the headline membership->rosch tables.

Reads a long-form metrics CSV produced by summarize_scores_file.py over the
260 score CSVs from evaluating membership-sans-rosch-v0-trained models on all
10 rosch categories. Emits TWO summary markdowns:

    MEAN_across_10_rosch_tasks_gemma-2-2b.md
        Single mean (std) across all 10 rosch eval tasks.

    BUCKETED_by_overlap_gemma-2-2b.md
        Three buckets defined by item-overlap between rosch-positive items and
        the membership training pool:
          - High overlap (>=60%): rosch-bird, rosch-carpenters-tool, rosch-fruit
          - Mid overlap (35-55%): rosch-vehicle, rosch-furniture, rosch-vegetable,
                                  rosch-toy, rosch-clothing, rosch-weapon
          - Clean (<10%):         rosch-sport
        Each bucket reports mean (std) over its tasks. (rosch-sport has only
        one task; std reported as "—".)

All numeric cells scaled by SCALE = 100, matching matched-task / cross-task
reports.
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
LONG_CSV = ROOT / "outputs-quickiter" / "membership-sans-rosch-v0-to-rosch" / "quickiter_metrics_long_membership_to_rosch.csv"
OUT_DIR  = ROOT / "outputs-quickiter" / "membership-sans-rosch-v0-to-rosch"

MODEL      = "gemma-2-2b"
TRAIN_TASK = "membership-sans-rosch-v0"
SCALE      = 100

ROSCH_TASKS = [
    "rosch-bird", "rosch-carpenters-tool", "rosch-clothing", "rosch-fruit",
    "rosch-furniture", "rosch-sport", "rosch-toy", "rosch-vegetable",
    "rosch-vehicle", "rosch-weapon",
]
BUCKETS = [
    ("High overlap (>=60%)", ["rosch-bird", "rosch-carpenters-tool", "rosch-fruit"]),
    ("Mid overlap (35-55%)", ["rosch-vehicle", "rosch-furniture", "rosch-vegetable",
                              "rosch-toy", "rosch-clothing", "rosch-weapon"]),
    ("Clean (<10%)",         ["rosch-sport"]),
]

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
    ("gen_roc",  "Generator ROC-AUC"),
    ("val_roc",  "Validator ROC-AUC"),
    ("val_acc",  "Validator accuracy (thr 0)"),
    ("pearson",  "Pearson(gen, validator)"),
]


def parse_filename(fname: str):
    """Return (id, label, eval_ref, eval_task) or None.

    Matches both fine-tuned and base model file patterns. eval_task must be in
    ROSCH_TASKS for inclusion.
    """
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
        if eval_task not in ROSCH_TASKS or sig not in SIG_MAP:
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
        if eval_task not in ROSCH_TASKS:
            return None
        return "0", f"Base HF ({MODEL})", eval_ref, eval_task
    return None


def load_long(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
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
    return pd.DataFrame(rows)


def make_value_getter(long: pd.DataFrame, tasks: list[str], metric: str):
    """Build a (vid, eval_ref) -> formatted-string lookup for one (bucket, metric).

    For multi-task buckets the cell is "mean (std)"; single-task gets bare mean.
    """
    sub = long[long["eval_task"].isin(tasks)]
    n_tasks = len(tasks)
    agg = (
        sub.groupby(["id", "eval_ref"])[metric]
           .agg(["mean", "std"])
           .reset_index()
    )
    val_map: dict[tuple[str, str], tuple[float, float]] = {}
    for _, r in agg.iterrows():
        val_map[(r["id"], r["eval_ref"])] = (r["mean"], r["std"])

    def get(vid: str, eval_ref: str) -> str | None:
        v = val_map.get((vid, eval_ref))
        if v is None:
            return None
        m, s = v
        if pd.isna(m):
            return None
        if n_tasks == 1:
            return tf4.fmt_single(m, scale=SCALE)
        return tf4.fmt_mean_std(m, s, scale=SCALE)

    return get


def emit_bucket_section(long: pd.DataFrame, label: str, tasks: list[str]) -> str:
    """Emit one '## bucket' section: header + (4 metrics × 4 tables) underneath."""
    n_tasks = len(tasks)
    parts = [f"## {label}  (n_tasks={n_tasks})", ""]
    for metric, title in METRICS_AND_TITLES:
        get = make_value_getter(long, tasks, metric)
        parts.append(tf4.emit_4_tables(get, f"{title} — × 100"))
    return "\n".join(parts)


def main():
    long = load_long(LONG_CSV)
    if long.empty:
        raise SystemExit(f"No rows parsed from {LONG_CSV}.")

    n_per_cell = long.groupby(["id", "train", "eval_ref"])["eval_task"].nunique()
    if (n_per_cell != 10).any():
        problem = n_per_cell[n_per_cell != 10]
        print("WARN: not every (id, eval_ref) cell has 10 tasks:")
        print(problem.head())

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    table_layout_blurb = (
        "Tables follow the canonical 4-table layout (see "
        "[docs/results_table_format.md](../../docs/results_table_format.md)): "
        "T1/T3 = baselines (Base HF + SFT) for the self/neg eval refs; "
        "T2/T4 = the 3×2 TC × pairs grids for self/neg. Cells in T2/T4 use the "
        "*best* eval ref per row (offline TC → `basetyp[neg]`; everything else "
        "→ `self`/`neg`). The (offline TC, online pairs) cell is always blank "
        "because that variant is not in the launcher.\n"
    )

    # 1. Mean across all 10 rosch tasks.
    mean_md = OUT_DIR / f"MEAN_across_10_rosch_tasks_{MODEL}.md"
    parts = [
        f"# membership->rosch mean across 10 rosch tasks ({MODEL})\n",
        f"**Train task:** {TRAIN_TASK}-all (165 categories, 2k items, 5110 pair samples).\n",
        f"**Eval tasks:** all 10 rosch categories ({', '.join(ROSCH_TASKS)}).\n",
        "**Validator:** log-odds (`--validator-log-odds`). "
        "**Generator column variant:** `tc` (TC-corrected gen score where applicable).\n",
        "**All numeric cells are raw values × 100** "
        "(percentage points for ROC/accuracy; 0–100 units for Pearson). "
        "Reported as **mean (std)** across the 10 rosch tasks.\n",
        table_layout_blurb,
        f"Long-form metrics: [{LONG_CSV.name}]({LONG_CSV.name})\n",
    ]
    parts.append(emit_bucket_section(long, "All 10 rosch tasks", ROSCH_TASKS))
    mean_md.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {mean_md.relative_to(ROOT)}")

    # 2. Bucketed report.
    buck_md = OUT_DIR / f"BUCKETED_by_overlap_{MODEL}.md"
    parts = [
        f"# membership->rosch stratified by item-overlap ({MODEL})\n",
        f"**Train task:** {TRAIN_TASK}-all. **Validator:** log-odds. "
        "**Generator column:** `tc`. **Values × 100.**\n",
        "Buckets defined by what fraction of each rosch task's positive items "
        "appear in the membership training pool (pos+neg). High-overlap buckets "
        "are closer to a memorization probe; low-overlap buckets test genuine "
        "OOD generalization.\n",
        "Item-overlap fractions: "
        + ", ".join([
            "rosch-bird 89%", "rosch-carpenters-tool 61%", "rosch-fruit 60%",
            "rosch-vehicle 56%", "rosch-furniture 45%", "rosch-vegetable 44%",
            "rosch-toy 42%", "rosch-clothing 36%", "rosch-weapon 36%",
            "rosch-sport 9%",
        ]) + ".\n",
        table_layout_blurb,
        f"Long-form metrics: [{LONG_CSV.name}]({LONG_CSV.name})\n",
    ]
    for bucket_name, tasks in BUCKETS:
        parts.append(emit_bucket_section(long, bucket_name, tasks))
    buck_md.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {buck_md.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
