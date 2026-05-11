#!/usr/bin/env python3
"""Build rosch_quickiter_summary.md from outputs-quickiter score CSVs.

1. Run: python scripts/summarize_scores_file.py --glob 'outputs-quickiter/scores_*.csv' \\
       --csv outputs-quickiter/quickiter_metrics_long.csv
2. Run: python scripts/build_rosch_quickiter_summary_tables.py

Produces pivot tables (tc generator column only; no Spearman) grouped by
train variant and eval TC reference (self / neg / basetyp / basetypneg).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LONG = ROOT / "outputs-quickiter" / "quickiter_metrics_long.csv"
OUT_MD = ROOT / "outputs-quickiter" / "rosch_quickiter_summary.md"

SIG_MAP = {
    "full-completion_force-same-x": ("1", "RankAlign baseline"),
    "tc-self_full-completion_force-same-x": ("2", "+ offline self-TC"),
    "tc-self_full-completion_force-same-x_online-tc": ("3", "+ online self-TC"),
    "full-completion_force-same-x_online-pairs": ("4", "+ online pairs"),
    "tc-self_full-completion_force-same-x_online-pairs_online-tc": ("5", "+ both online (self)"),
    "full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x": ("6", "SFT (NLL all)"),
    "tc-neg_full-completion_force-same-x": ("7", "+ offline neg-TC"),
    "tc-neg_full-completion_force-same-x_online-tc": ("8", "+ online neg-TC"),
    "tc-neg_full-completion_force-same-x_online-pairs_online-tc": ("9", "+ both online (neg)"),
}

ORDER = [
    ("0", "Base HF (gemma-2-2b)"),
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

EVAL_COLS = ["self", "neg", "basetyp", "basetypneg"]

FNAME_RE = re.compile(
    r"^scores_(basetypneg|basetyp|neg|self)-v6-google_gemma-2-2b-delta0\.15-epoch2_"
    r"rosch-furniture-and-bird-all_d2g_random_alpha1\.0_(.+)_rosch-furniture-and-bird_test_log-odds_tc_20260510\.csv$"
)


def parse_filename(name: str):
    name = name.replace(".csv", "")
    if name == "scores_self-v6-google_gemma-2-2b_rosch-furniture-and-bird_test_log-odds_tc_20260510":
        return "0", "Base HF (gemma-2-2b)", "self"
    if name == "scores_neg-v6-google_gemma-2-2b_rosch-furniture-and-bird_test_log-odds_tc_20260510":
        return "0", "Base HF (gemma-2-2b)", "neg"
    m = FNAME_RE.match(name + ".csv")
    if not m:
        return None
    eval_ref, sig = m.group(1), m.group(2)
    if sig not in SIG_MAP:
        return None
    return (*SIG_MAP[sig], eval_ref)


def wide(piv: pd.DataFrame, metric: str) -> pd.DataFrame:
    t = piv.pivot_table(index=["id", "train"], columns="eval_ref", values=metric, aggfunc="first")
    for c in EVAL_COLS:
        if c not in t.columns:
            t[c] = float("nan")
    t = t[EVAL_COLS]
    present = set(zip(piv["id"], piv["train"]))
    out_rows = []
    for i, lab in ORDER:
        if (i, lab) not in present:
            continue
        row = {"#": i, "trained model": lab}
        sub = t.loc[(i, lab)]
        for c in EVAL_COLS:
            v = sub[c]
            row[c] = "—" if pd.isna(v) else f"{float(v):.4f}"
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def md_table(wdf: pd.DataFrame, title: str) -> str:
    lines = [f"### {title}", "", "| " + " | ".join(wdf.columns) + " |",
             "| " + " | ".join(["---"] * len(wdf.columns)) + " |"]
    for _, r in wdf.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in wdf.columns) + " |")
    lines.append("")
    return "\n".join(lines)


def main():
    df = pd.read_csv(LONG)
    df_tc = df[df["variant"] == "tc"].copy()
    rows = []
    for _, r in df_tc.iterrows():
        p = parse_filename(r["file"])
        if p is None:
            print("UNPARSED:", r["file"])
            continue
        num, train_label, eval_ref = p
        rows.append({
            "id": num, "train": train_label, "eval_ref": eval_ref,
            "gen_roc": r["gen_roc"], "val_roc": r["val_roc"],
            "val_acc": r["val_acc"], "pearson": r["pearson"],
        })
    piv = pd.DataFrame(rows)
    dup = piv.groupby(["id", "train", "eval_ref"]).size()
    if (dup > 1).any():
        raise SystemExit(f"Duplicate (id, train, eval_ref): {dup[dup > 1]}")

    parts = [
        "# Rosch quick-iter (gemma-2-2b) — rosch-furniture-and-bird\n",
        "**Task:** combined furniture + bird (186 items, 93/93 yes/no). "
        "**Validator:** log-odds (`--validator-log-odds`). "
        "**Metrics from `summarize_scores_file.py`, generator column variant `tc`** "
        "(TC-corrected gen score where applicable).\n",
        "Spearman omitted per your usual reporting preference.\n",
        "Long-form metrics: [quickiter_metrics_long.csv](quickiter_metrics_long.csv)\n",
    ]
    for metric, title in [
        ("gen_roc", "Generator ROC-AUC (`tc` column)"),
        ("val_roc", "Validator ROC-AUC (same across gen variants for a file; shown for reference)"),
        ("val_acc", "Validator accuracy (threshold 0)"),
        ("pearson", "Pearson(gen, validator) — `tc` gen vs val_score"),
    ]:
        parts.append(md_table(wide(piv, metric), title))

    OUT_MD.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
