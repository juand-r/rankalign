"""Tier 0 diagnostics for docs/DEBUGGING-PLAN.md.

For each (model, task) combination, this script reproduces the train-time
pair construction *offline* against the BASE model's per-item validator
scores (which are saved in the eval CSVs we already have on disk), then
reports:

  1. Pair-label correctness (the headline number).
  2. TC weight vs ground-truth correctness.
  3. Validator calibration (per-|Δv| bin accuracy).

What it does NOT do: rerun the model. The base eval CSVs already store
val_score, gen_score, gen_score_typcorr per (x, y) row. RankAlign builds
training pairs by (a) grouping by prompt (force-same-x), (b) within each
group forming all (i, j) pairs, (c) filtering by |val_score_i − val_score_j|
> delta, (d) calling the higher-val_score item the "winner". So pairs are
fully reconstructible from those CSVs.

Output: docs/diag_pair_correctness_report.md
        outputs-quickiter/diag_pairs_<model>_<task>.csv   (one per combo)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs-quickiter"
DOCS = ROOT / "docs"

# (label, model, task, csv_filename, prompt_col, label_col)
COMBOS = [
    ("rosch 2b",      "gemma-2-2b",
     "rosch-furniture-and-bird",
     "scores_self-v6-google_gemma-2-2b_rosch-furniture-and-bird_test_log-odds_tc_20260510.csv",
     "category", "label"),
    ("rosch 2b-it",   "gemma-2-2b-it",
     "rosch-furniture-and-bird",
     "scores_self-v6-google_gemma-2-2b-it_rosch-furniture-and-bird_test_log-odds_tc_20260510.csv",
     "category", "label"),
    ("ambigqa 2b",    "gemma-2-2b",
     "ambigqa-train-as-test",
     "scores_self-v6-google_gemma-2-2b_ambigqa-train-as-test_test_log-odds_tc_20260511.csv",
     "question", "gpt4_ground_truth"),
    ("ambigqa 2b-it", "gemma-2-2b-it",
     "ambigqa-train-as-test",
     "scores_self-v6-google_gemma-2-2b-it_ambigqa-train-as-test_test_log-odds_tc_20260511.csv",
     "question", "gpt4_ground_truth"),
]

DELTA = 0.15
CALIB_BINS = [0.15, 0.30, 0.50, 1.0, 2.0, 4.0, np.inf]
TC_BINS = [-np.inf, -2.0, -1.0, -0.3, 0.3, 1.0, 2.0, np.inf]


def build_pairs(df: pd.DataFrame, prompt_col: str, label_col: str, delta: float) -> pd.DataFrame:
    """Reproduce force-same-x + delta filter pair construction.

    For every prompt group with ≥2 items, emit every directed pair (winner,
    loser) with val_score_winner > val_score_loser and |Δv| > delta. We emit
    the *directed* form so 'winner' / 'loser' are unambiguous and we don't
    have to re-decide downstream.
    """
    rows = []
    for prompt, sub in df.groupby(prompt_col, sort=False):
        if len(sub) < 2:
            continue
        sub = sub.reset_index(drop=True)
        for i in range(len(sub)):
            for j in range(len(sub)):
                if i == j:
                    continue
                vi = sub.loc[i, "val_score"]
                vj = sub.loc[j, "val_score"]
                if not (vi > vj):
                    continue
                if (vi - vj) <= delta:
                    continue
                rows.append({
                    "prompt": prompt,
                    "winner_y": sub.loc[i, "answer_text"],
                    "loser_y":  sub.loc[j, "answer_text"],
                    "winner_label": sub.loc[i, label_col],
                    "loser_label":  sub.loc[j, label_col],
                    "val_winner": float(vi),
                    "val_loser":  float(vj),
                    "delta_v":    float(vi - vj),
                    "gen_winner": float(sub.loc[i, "gen_score"]),
                    "gen_loser":  float(sub.loc[j, "gen_score"]),
                    "gentc_winner": float(sub.loc[i, "gen_score_typcorr"]),
                    "gentc_loser":  float(sub.loc[j, "gen_score_typcorr"]),
                })
    if not rows:
        return pd.DataFrame()
    P = pd.DataFrame(rows)
    P["delta_gen"]   = P["gen_winner"]   - P["gen_loser"]
    P["delta_gentc"] = P["gentc_winner"] - P["gentc_loser"]
    P["tc_adj"]      = P["delta_gentc"] - P["delta_gen"]
    # Pair correctness: "yes" item ranked above "no" item.
    P["yesno_pair"]  = (P["winner_label"] != P["loser_label"])
    P["correct"]     = P["winner_label"].str.lower().eq("yes") & P["loser_label"].str.lower().eq("no")
    P["wrong"]       = P["winner_label"].str.lower().eq("no")  & P["loser_label"].str.lower().eq("yes")
    return P


def normalize_csv(df: pd.DataFrame, prompt_col: str, label_col: str) -> pd.DataFrame:
    """Pick out the columns we use, with stable names ('answer_text' for the y)."""
    if prompt_col == "category":
        df = df.rename(columns={"member": "answer_text"})
    else:
        df = df.rename(columns={"answer": "answer_text"})
    keep = [prompt_col, "answer_text", label_col,
            "val_score", "gen_score", "gen_score_typcorr"]
    df = df[keep].copy()
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()
    return df


def headline_stats(P: pd.DataFrame) -> dict:
    n_pairs = len(P)
    n_yesno = int(P["yesno_pair"].sum())
    n_corr  = int(P["correct"].sum())
    n_wrong = int(P["wrong"].sum())
    n_both_yes = int(((P["winner_label"] == "yes") & (P["loser_label"] == "yes")).sum())
    n_both_no  = int(((P["winner_label"] == "no")  & (P["loser_label"] == "no")).sum())
    accuracy = (n_corr / n_yesno) if n_yesno else float("nan")
    return {
        "n_pairs": n_pairs,
        "n_yesno": n_yesno,
        "n_correct": n_corr,
        "n_wrong": n_wrong,
        "n_both_yes": n_both_yes,
        "n_both_no":  n_both_no,
        "yesno_accuracy": accuracy,
    }


def calib_table(P: pd.DataFrame) -> pd.DataFrame:
    yn = P[P["yesno_pair"]].copy()
    yn["bin"] = pd.cut(yn["delta_v"], bins=CALIB_BINS, include_lowest=True)
    g = yn.groupby("bin", observed=True)
    out = pd.DataFrame({
        "n_pairs": g.size(),
        "accuracy": g["correct"].mean(),
        "mean_delta_v": g["delta_v"].mean(),
    })
    return out.reset_index().rename(columns={"bin": "|Δv| bin"})


def tc_table(P: pd.DataFrame) -> pd.DataFrame:
    yn = P[P["yesno_pair"]].copy()
    yn["bin"] = pd.cut(yn["tc_adj"], bins=TC_BINS)
    g = yn.groupby("bin", observed=True)
    out = pd.DataFrame({
        "n_pairs": g.size(),
        "accuracy": g["correct"].mean(),
        "mean_tc_adj": g["tc_adj"].mean(),
    })
    return out.reset_index().rename(columns={"bin": "TC adj bin"})


def fmt_md_table(df: pd.DataFrame, float_cols: list[str]) -> str:
    if df.empty:
        return "_(no rows)_\n"
    df = df.copy()
    for c in float_cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    if "n_pairs" in df.columns:
        df["n_pairs"] = df["n_pairs"].astype(int)
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delta", type=float, default=DELTA,
                    help="Pair filter on |val_score_i − val_score_j| (default 0.15, training default).")
    ap.add_argument("--out-md", default=str(DOCS / "diag_pair_correctness_report.md"))
    args = ap.parse_args()

    delta = args.delta
    out_md = Path(args.out_md).resolve()

    parts: list[str] = []
    parts.append("# Pair-label correctness diagnostic (Tier 0)")
    parts.append("")
    parts.append(
        f"Generated by `scripts/diag_pair_label_correctness.py` with `delta = {delta}`. "
        "Reproduces train-time pair construction offline against the BASE model's "
        "validator scores (`val_score` in the eval CSVs).\n"
    )
    parts.append(
        "**Headline:** for each (model, task), among pairs where the two items "
        "have *different* ground-truth labels (one yes, one no), how often does "
        "the validator's pair ordering agree with ground truth?\n"
    )

    headline_rows = []

    for label, model, task, csv_name, prompt_col, label_col in COMBOS:
        csv_path = OUTPUTS / csv_name
        if not csv_path.exists():
            parts.append(f"## {label}\n\n_CSV missing: `{csv_path.name}`_\n")
            continue
        df_raw = pd.read_csv(csv_path)
        df = normalize_csv(df_raw, prompt_col, label_col)
        P = build_pairs(df, prompt_col, label_col, delta)
        stats = headline_stats(P)

        out_pairs_csv = OUTPUTS / f"diag_pairs_{model}_{task}.csv"
        if not P.empty:
            P.to_csv(out_pairs_csv, index=False)

        headline_rows.append({
            "setting": label,
            "n_items": len(df),
            "n_pos": int((df[label_col] == "yes").sum()),
            "n_neg": int((df[label_col] == "no").sum()),
            "pairs (after Δv>{delta})": stats["n_pairs"],
            "yes-vs-no pairs": stats["n_yesno"],
            "validator accuracy on yes-vs-no": stats["yesno_accuracy"],
        })

        parts.append(f"## {label}  (`{csv_path.name}`)\n")
        parts.append(
            f"Items: **{len(df):,}** "
            f"({stats['n_both_yes'] + stats['n_yesno'] // 2 if False else int((df[label_col] == 'yes').sum()):,} yes / "
            f"{int((df[label_col] == 'no').sum()):,} no). "
            f"Δ filter: `|Δv| > {delta}`.\n"
        )
        parts.append(
            f"- **Total pairs after Δv-filter:** {stats['n_pairs']:,}\n"
            f"  - both-yes: {stats['n_both_yes']:,}\n"
            f"  - both-no:  {stats['n_both_no']:,}\n"
            f"  - yes-vs-no: **{stats['n_yesno']:,}**\n"
            f"- **Among yes-vs-no pairs:** validator-correct = "
            f"{stats['n_correct']:,}, validator-wrong = {stats['n_wrong']:,}\n"
            f"- **Validator pair-ordering accuracy (yes-vs-no only): "
            f"{stats['yesno_accuracy']:.4f}**\n"
        )
        parts.append("### Calibration (yes-vs-no pairs only)\n")
        parts.append(fmt_md_table(calib_table(P), ["accuracy", "mean_delta_v"]))
        parts.append("\n### TC-adjustment (Δgen_typcorr − Δgen) vs correctness\n")
        parts.append(
            "_Sign convention: positive `tc_adj` means TC up-weights this pair "
            "(TC-corrected gap is wider than raw gap). If TC is steering us, "
            "accuracy should rise with `tc_adj`. If `tc_adj` is uncorrelated "
            "with correctness, TC is amplifying noise._\n"
        )
        parts.append(fmt_md_table(tc_table(P), ["accuracy", "mean_tc_adj"]))
        parts.append("")

    parts.insert(4, "## Headline summary\n")
    head_df = pd.DataFrame(headline_rows)
    if not head_df.empty:
        parts.insert(5, fmt_md_table(head_df, ["validator accuracy on yes-vs-no"]) + "\n")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {out_md.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
