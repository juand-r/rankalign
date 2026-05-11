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

# All numeric cells are reported as raw_value × SCALE so that ROC / accuracy /
# Pearson read as percentage points (easier to compare deltas at a glance).
# If you change this, update scripts/check_summary_consistency.py too.
SCALE = 100

METRICS_AND_TITLES = [
    ("gen_roc",  "Generator ROC-AUC (`tc` column) — values × 100"),
    ("val_roc",  "Validator ROC-AUC (same across gen variants; shown for reference) — values × 100"),
    ("val_acc",  "Validator accuracy (threshold 0) — values × 100"),
    ("pearson",  "Pearson(gen, validator) — `tc` gen vs val_score — values × 100"),
]


def build_filename_regexes(model: str, train_task: str, eval_task: str | None = None):
    """Return (fine_tune_re, base_self_re, base_neg_re) for the given model and tasks.

    The model token in the filename uses '_' as the family separator
    ("google_gemma-2-2b" or "google_gemma-2-2b-it"), which we mirror.

    train_task = task baked into the model checkpoint path.
    eval_task  = task that appears just before _test_log-odds_; defaults to
                 train_task (matched-task evaluation, the original setting).
    """
    if eval_task is None:
        eval_task = train_task
    m = re.escape(model)
    tr = re.escape(train_task)
    ev = re.escape(eval_task)
    fine_re = re.compile(
        rf"^scores_(basetypneg|basetyp|neg|self)-v6-google_{m}-delta0\.15-epoch2_"
        rf"{tr}-all_d2g_random_alpha1\.0_(.+)_{ev}_test_log-odds_tc_\d+\.csv$"
    )
    base_self_re = re.compile(
        rf"^scores_self-v6-google_{m}_{ev}_test_log-odds_tc_\d+\.csv$"
    )
    base_neg_re = re.compile(
        rf"^scores_neg-v6-google_{m}_{ev}_test_log-odds_tc_\d+\.csv$"
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
            row[c] = "—" if pd.isna(v) else f"{float(v) * SCALE:.2f}"
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
                    help="Train task name (e.g. rosch-furniture-and-bird, ambigqa-train-as-test). "
                         "Identifies the task baked into the model checkpoint.")
    ap.add_argument("--eval-task", default=None,
                    help="Eval task; defaults to --task. Use a different value to build a "
                         "cross-task table (e.g. trained on rosch-furniture-and-bird, eval on rosch-toy).")
    ap.add_argument("--long-csv", default=None,
                    help="Long-form metrics CSV (default chosen by task).")
    ap.add_argument("--out", default=None,
                    help="Output markdown path (default: outputs-quickiter/{task_short}_quickiter_summary_{model}.md).")
    args = ap.parse_args()

    train_task = args.task
    eval_task  = args.eval_task or train_task
    matched    = (eval_task == train_task)

    if args.long_csv is None:
        if "ambigqa" in train_task:
            long_csv = ROOT / "outputs-quickiter" / "quickiter_metrics_long_ambigqa.csv"
        else:
            long_csv = ROOT / "outputs-quickiter" / "quickiter_metrics_long.csv"
    else:
        long_csv = Path(args.long_csv).resolve()

    task_short = "ambigqa" if "ambigqa" in train_task else "rosch"
    if args.out:
        out_md = Path(args.out).resolve()
    elif matched:
        out_md = ROOT / "outputs-quickiter" / f"{task_short}_quickiter_summary_{args.model}.md"
    else:
        out_md = (ROOT / "outputs-quickiter" /
                  f"{train_task}-to-ood" / f"{eval_task}_summary_{args.model}.md")

    if not long_csv.exists():
        raise SystemExit(f"long CSV not found: {long_csv}")

    df = pd.read_csv(long_csv)
    df_tc = df[df["variant"] == "tc"].copy()

    fine_re, base_self_re, base_neg_re = build_filename_regexes(
        args.model, train_task, eval_task
    )
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
            train_token_match = f"_{train_task}-all_" in r["file"] or (
                matched and f"_{train_task}_test_" in r["file"]
            )
            eval_token_match  = f"_{eval_task}_test_" in r["file"]
            if model_token_match and train_token_match and eval_token_match:
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
        raise SystemExit(
            f"No rows matched model={args.model} train_task={train_task} "
            f"eval_task={eval_task} in {long_csv}."
        )

    piv = pd.DataFrame(rows)
    dup = piv.groupby(["id", "train", "eval_ref"]).size()
    if (dup > 1).any():
        raise SystemExit(f"Duplicate (id, train, eval_ref): {dup[dup > 1]}")

    order = [("0", base_label)] + NUMBERED_ORDER
    items_blurb = get_dataset_size_blurb(piv, df_tc)

    if matched:
        title = f"# {task_short.capitalize()} quick-iter ({args.model}) — {train_task}\n"
        task_line = f"**Task:** {train_task} ({items_blurb}). "
        confound_blurb = ("Train ≡ test by construction — these tables are a memorization probe; "
                          "do NOT read them as cross-task generalization.\n")
    else:
        title = (f"# Cross-task quick-iter ({args.model}) — "
                 f"trained on {train_task}, evaluated on **{eval_task}**\n")
        task_line = (f"**Train task:** {train_task} (model checkpoints). "
                     f"**Eval task:** {eval_task} ({items_blurb}). ")
        confound_blurb = ("Models were trained on a *different* rosch category set "
                          "(furniture+bird) and are evaluated here as out-of-distribution. "
                          "This IS a generalization read.\n")

    parts = [
        title,
        task_line +
        f"**Validator:** log-odds (`--validator-log-odds`). "
        f"**Metrics from `summarize_scores_file.py`, generator column variant `tc`** "
        "(TC-corrected gen score where applicable).\n",
        confound_blurb,
        "**All numeric cells are raw values × 100** (i.e. ROC-AUC and accuracy "
        "are in percentage points; Pearson is in 0–100 units).\n",
        f"Long-form metrics: [{long_csv.name}]({long_csv.name})\n",
    ]
    for metric, title in METRICS_AND_TITLES:
        parts.append(md_table(wide(piv, metric, order), title))

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(parts), encoding="utf-8")
    try:
        rel = out_md.relative_to(ROOT)
        print(f"Wrote {rel}")
    except ValueError:
        print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
