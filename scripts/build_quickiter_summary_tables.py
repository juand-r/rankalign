#!/usr/bin/env python3
"""Build a quick-iter Generator-ROC-AUC summary markdown for one (model, task)
combination. Generalizes scripts/build_rosch_quickiter_summary_tables.py to
work for both rosch-furniture-and-bird and ambigqa-train-as-test, and for both
gemma-2-2b and gemma-2-2b-it.

Usage:
    python scripts/build_quickiter_summary_tables.py \\
        --model gemma-2-2b-it --task rosch-furniture-and-bird

Pipeline:
    1. summarize_scores_file.py glob → long CSV (already done by the morning
       report; this script just reads it).
    2. This script filters to the (model, task) pair, parses signatures from
       filenames, and pivots into per-metric tables.

Output is written to outputs-quickiter/{task}_quickiter_summary_{model}.md.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Per-variant signatures (the bit in the filename between alpha1.0_ and the
# trailing {task}_test_...). Values are (id, display label).
SIG_MAP = {
    "full-completion_force-same-x":
        ("1", "RankAlign baseline"),
    "tc-self_full-completion_force-same-x":
        ("2", "+ offline self-TC"),
    "tc-self_full-completion_force-same-x_online-tc":
        ("3", "+ online self-TC"),
    "full-completion_force-same-x_online-pairs":
        ("4", "+ online pairs"),
    "tc-self_full-completion_force-same-x_online-pairs_online-tc":
        ("5", "+ both online (self)"),
    "full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x":
        ("6", "SFT (NLL all)"),
    "tc-neg_full-completion_force-same-x":
        ("7", "+ offline neg-TC"),
    "tc-neg_full-completion_force-same-x_online-tc":
        ("8", "+ online neg-TC"),
    "tc-neg_full-completion_force-same-x_online-pairs_online-tc":
        ("9", "+ both online (neg)"),
}

# Display order. Base ("0") is appended dynamically using the model name.
NUMBERED_ORDER = [
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
    ("gen_roc",  "Generator ROC-AUC (`tc` column)"),
    ("val_roc",  "Validator ROC-AUC (same across gen variants; shown for reference)"),
    ("val_acc",  "Validator accuracy (threshold 0)"),
    ("pearson",  "Pearson(gen, validator) — `tc` gen vs val_score"),
]


def build_filename_regexes(model: str, task: str):
    """Return (fine_tune_re, base_self_re, base_neg_re) for the given model+task.

    The model token in the filename uses '_' as the family separator
    ("google_gemma-2-2b" or "google_gemma-2-2b-it"), which we mirror.
    """
    m = re.escape(model)
    t = re.escape(task)
    fine_re = re.compile(
        rf"^scores_(basetypneg|basetyp|neg|self)-v6-google_{m}-delta0\.15-epoch2_"
        rf"{t}-all_d2g_random_alpha1\.0_(.+)_{t}_test_log-odds_tc_\d+\.csv$"
    )
    base_self_re = re.compile(
        rf"^scores_self-v6-google_{m}_{t}_test_log-odds_tc_\d+\.csv$"
    )
    base_neg_re = re.compile(
        rf"^scores_neg-v6-google_{m}_{t}_test_log-odds_tc_\d+\.csv$"
    )
    return fine_re, base_self_re, base_neg_re


def parse_filename(fname: str, fine_re, base_self_re, base_neg_re, base_label: str):
    if base_self_re.match(fname):
        return "0", base_label, "self"
    if base_neg_re.match(fname):
        return "0", base_label, "neg"
    m = fine_re.match(fname)
    if not m:
        return None
    eval_ref, sig = m.group(1), m.group(2)
    if sig not in SIG_MAP:
        return None
    return (*SIG_MAP[sig], eval_ref)


def wide(piv: pd.DataFrame, metric: str, order) -> pd.DataFrame:
    t = piv.pivot_table(
        index=["id", "train"], columns="eval_ref", values=metric, aggfunc="first"
    )
    for c in EVAL_REFS:
        if c not in t.columns:
            t[c] = float("nan")
    t = t[EVAL_REFS]
    present = set(zip(piv["id"], piv["train"]))
    out_rows = []
    for i, lab in order:
        if (i, lab) not in present:
            continue
        row = {"#": i, "trained model": lab}
        sub = t.loc[(i, lab)]
        for c in EVAL_REFS:
            v = sub[c]
            row[c] = "—" if pd.isna(v) else f"{float(v):.4f}"
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def md_table(wdf: pd.DataFrame, title: str) -> str:
    if wdf.empty:
        return f"### {title}\n\n_(no rows)_\n"
    lines = [f"### {title}", "",
             "| " + " | ".join(wdf.columns) + " |",
             "| " + " | ".join(["---"] * len(wdf.columns)) + " |"]
    for _, r in wdf.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in wdf.columns) + " |")
    lines.append("")
    return "\n".join(lines)


def get_dataset_size_blurb(piv: pd.DataFrame, df_full: pd.DataFrame) -> str:
    """Pull n_total/n_pos/n_neg from any matching row in the long CSV."""
    if "file" not in df_full.columns:
        return ""
    files_in_piv = set()
    for f in piv.get("file", []):
        files_in_piv.add(f)
    sub = df_full[df_full["file"].isin(files_in_piv)]
    if sub.empty:
        return ""
    n_total = int(sub["n_total"].iloc[0])
    n_pos = int(sub["n_pos"].iloc[0])
    n_neg = int(sub["n_neg"].iloc[0])
    return f"{n_total:,} items, {n_pos:,}/{n_neg:,} yes/no"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    help="HF model slug (e.g. gemma-2-2b, gemma-2-2b-it).")
    ap.add_argument("--task", required=True,
                    help="Task name (e.g. rosch-furniture-and-bird, ambigqa-train-as-test).")
    ap.add_argument("--long-csv", default=None,
                    help="Long-form metrics CSV (default chosen by task).")
    ap.add_argument("--out", default=None,
                    help="Output markdown path (default: outputs-quickiter/{task_short}_quickiter_summary_{model}.md).")
    args = ap.parse_args()

    if args.long_csv is None:
        if "ambigqa" in args.task:
            long_csv = ROOT / "outputs-quickiter" / "quickiter_metrics_long_ambigqa.csv"
        else:
            long_csv = ROOT / "outputs-quickiter" / "quickiter_metrics_long.csv"
    else:
        long_csv = Path(args.long_csv).resolve()

    task_short = "ambigqa" if "ambigqa" in args.task else "rosch"
    out_md = (Path(args.out).resolve() if args.out
              else ROOT / "outputs-quickiter" / f"{task_short}_quickiter_summary_{args.model}.md")

    if not long_csv.exists():
        raise SystemExit(f"long CSV not found: {long_csv}")

    df = pd.read_csv(long_csv)
    df_tc = df[df["variant"] == "tc"].copy()

    fine_re, base_self_re, base_neg_re = build_filename_regexes(args.model, args.task)
    base_label = f"Base HF ({args.model})"

    rows = []
    n_unparsed = 0
    n_unmatched = 0
    sample_unparsed = []
    for _, r in df_tc.iterrows():
        p = parse_filename(r["file"], fine_re, base_self_re, base_neg_re, base_label)
        if p is None:
            # Could be (a) a different model/task combo from the same long CSV,
            # or (b) a real parse failure. Distinguish, using boundary-aware
            # checks (avoid substring false positives like
            # "gemma-2-2b" matching "gemma-2-2b-it").
            model_token_match = (
                f"_{args.model}-delta" in r["file"]
                or f"_{args.model}_" in r["file"]
            )
            task_token_match = f"_{args.task}-all_" in r["file"] or f"_{args.task}_test_" in r["file"]
            if model_token_match and task_token_match:
                n_unparsed += 1
                if len(sample_unparsed) < 3:
                    sample_unparsed.append(r["file"])
            else:
                n_unmatched += 1
            continue
        num, train_label, eval_ref = p
        rows.append({
            "id": num, "train": train_label, "eval_ref": eval_ref,
            "gen_roc": r["gen_roc"], "val_roc": r["val_roc"],
            "val_acc": r["val_acc"], "pearson": r["pearson"],
            "file": r["file"],
        })

    if n_unparsed:
        print(f"WARN: {n_unparsed} (model+task)-matching rows had unrecognized signatures.")
        for f in sample_unparsed:
            print(f"  e.g. {f}")
    print(f"INFO: {n_unmatched} rows skipped (different model/task in the same long CSV).")

    if not rows:
        raise SystemExit(f"No rows matched model={args.model} task={args.task} in {long_csv}.")

    piv = pd.DataFrame(rows)
    dup = piv.groupby(["id", "train", "eval_ref"]).size()
    if (dup > 1).any():
        raise SystemExit(f"Duplicate (id, train, eval_ref): {dup[dup > 1]}")

    order = [("0", base_label)] + NUMBERED_ORDER
    items_blurb = get_dataset_size_blurb(piv, df_tc)

    parts = [
        f"# {task_short.capitalize()} quick-iter ({args.model}) — {args.task}\n",
        f"**Task:** {args.task} ({items_blurb}). "
        f"**Validator:** log-odds (`--validator-log-odds`). "
        f"**Metrics from `summarize_scores_file.py`, generator column variant `tc`** "
        "(TC-corrected gen score where applicable).\n",
        "Train ≡ test by construction — these tables are a memorization probe; "
        "do NOT read them as cross-task generalization.\n",
        f"Long-form metrics: [{long_csv.name}]({long_csv.name})\n",
    ]
    for metric, title in METRICS_AND_TITLES:
        parts.append(md_table(wide(piv, metric, order), title))

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {out_md.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
