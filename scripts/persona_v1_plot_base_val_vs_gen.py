#!/usr/bin/env python3
"""
2x3 scatter grid: x = gen_score, y = val_score, one subplot per persona,
for the base model, persona-v1.

Score variants for the x-axis (gen):
  - raw    : gen_score, read from the self-CSV (raw is identical in self/neg)
  - tc-self: gen_score_typcorr, read from `scores_self-...csv`
  - tc-neg : gen_score_typcorr, read from `scores_neg-...csv`

The y-axis (val_score) is identical across the 3 score variants per persona;
only the x-axis (gen) changes, plus per-subplot Gen-AUC and Pearson r.

v1 vs v0:
- DROPPED `subscribes-to-moral-nihilism` and `believes-life-has-no-meaning`
  (poor validator accuracy on v0).
- Labels for `psychopathy`, `machiavellianism`, `narcissism` are FLIPPED
  upstream in the dataset builder so that "yes" = NOT antisocial.

Usage:
  python scripts/persona_v1_plot_base_val_vs_gen.py                       # all bases x all scores
  python scripts/persona_v1_plot_base_val_vs_gen.py --base 9b-it          # single base, all scores
  python scripts/persona_v1_plot_base_val_vs_gen.py --base 9b-it --score raw
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
OUT_DIR = REPO_ROOT / "plots"

# v0 list (kept for reference):
# PERSONAS_V0 = [
#     "psychopathy", "machiavellianism", "narcissism",
#     "subscribes-to-moral-nihilism", "believes-life-has-no-meaning",
#     "desire-to-create-allies", "interest-in-music", "interest-in-science",
# ]

PERSONAS = [
    # in-domain (label-flipped: yes = NOT antisocial)
    "psychopathy",
    "machiavellianism",
    "narcissism",
    # held-out (labels unchanged: yes = persona-aligned positive trait)
    "desire-to-create-allies",
    "interest-in-music",
    "interest-in-science",
]

ALL_BASES = ["9b-it", "2b-it", "2b"]
ALL_SCORES = ["raw", "tc-self", "tc-neg"]
DISC_TAGS = ["auto", "zero", "few"]  # auto = whatever is in the file, no filename suffix

# (eval_prefix, gen_column, x_label, score_label)
SCORE_SPEC = {
    "raw":     ("self", "gen_score",         "gen_score (raw, log-prob)",  "raw"),
    "tc-self": ("self", "gen_score_typcorr", "gen_score (self-TC)",        "self-TC"),
    "tc-neg":  ("neg",  "gen_score_typcorr", "gen_score (neg-TC)",         "neg-TC"),
}


def make_one_plot(base: str, score: str, disc_tag: str = "auto") -> Path | None:
    """Build the 2x3 grid for one (base, score) pair. Returns the output path,
    or None if no input files were found at all.

    disc_tag: 'auto' (default) plots whatever is in the file with no filename
              suffix. 'zero' / 'few' filters to rows whose strategy column
              matches and adds a '_disc-{zero|few}' suffix to the filename, so
              parallel runs (e.g. disc-zero and disc-few baseline evals on the
              same model) don't overwrite each other.
    """
    eval_prefix, gen_col, x_label, score_label = SCORE_SPEC[score]
    model_tag = f"v6-google_gemma-2-{base}"

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), sharex=False, sharey=False)
    axes = axes.flatten()
    any_found = False

    for ax, persona in zip(axes, PERSONAS):
        pattern = f"scores_{eval_prefix}-{model_tag}_persona-v1-{persona}_test_log-odds_tc_*.csv"
        candidates = list(OUTPUTS_DIR.glob(pattern))
        if not candidates:
            ax.set_title(f"{persona}\n(no file)", fontsize=10)
            ax.axis("off")
            continue
        # When disc_tag is 'zero' or 'few', prefer the latest CSV whose
        # strategy column matches; if no candidate matches, leave the subplot
        # empty so a mislabelled or mixed run is visible rather than silently
        # plotting the wrong data.
        chosen = None
        for f in sorted(candidates, reverse=True):
            df_try = pd.read_csv(f)
            strat_col = df_try["strategy"].iloc[0] if len(df_try) else ""
            if disc_tag == "auto" or f"disc:{disc_tag}" in str(strat_col):
                chosen = (f, df_try)
                break
        if chosen is None:
            ax.set_title(f"{persona}\n(no disc:{disc_tag} file)", fontsize=10)
            ax.axis("off")
            continue
        any_found = True
        _, df = chosen
        labels = (df["correct"].str.strip().str.lower() == "yes").astype(int)
        if gen_col not in df.columns:
            ax.set_title(f"{persona}\n(missing {gen_col})", fontsize=10)
            ax.axis("off")
            continue
        x = df[gen_col].values
        y = df["val_score"].values

        mask = ~(np.isnan(x) | np.isnan(y))
        x, y, labels_v = x[mask], y[mask], labels.values[mask]

        ax.scatter(x[labels_v == 1], y[labels_v == 1],
                   s=14, alpha=0.6, c="tab:red", label="yes")
        ax.scatter(x[labels_v == 0], y[labels_v == 0],
                   s=14, alpha=0.6, c="tab:blue", label="no")
        ax.axhline(0, color="gray", lw=0.6, alpha=0.5)

        try:
            auc = roc_auc_score(labels_v, x)
        except Exception:
            auc = float("nan")
        try:
            r, _ = pearsonr(x, y)
        except Exception:
            r = float("nan")

        ax.set_title(f"{persona}\nGen-AUC={auc:.2f}  r={r:.2f}  n={len(x)}",
                     fontsize=10)
        ax.set_xlabel(x_label)
        ax.set_ylabel("val_score (log-odds)")
        ax.grid(alpha=0.25)

    if not any_found:
        plt.close(fig)
        print(f"[skip] gemma-2-{base} / {score_label}: no input files found")
        return None

    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor="tab:red", markersize=7, label="label=yes"),
               plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor="tab:blue", markersize=7, label="label=no")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.995),
               ncol=2, frameon=False, fontsize=10)
    disc_str = "" if disc_tag == "auto" else f", disc-{disc_tag}"
    fig.suptitle(
        f"Base gemma-2-{base}{disc_str}: gen_score ({score_label}) vs val_score, per persona-v1 task (test split)",
        fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    score_slug = score.replace("-", "_")  # tc_self / tc_neg / raw
    disc_suffix = "" if disc_tag == "auto" else f"_disc-{disc_tag}"
    out_path = OUT_DIR / f"persona_v1_base_gemma-2-{base}_gen_{score_slug}_vs_val{disc_suffix}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="all",
                        choices=["all"] + ALL_BASES,
                        help="Base model to plot, or 'all' (default: all).")
    parser.add_argument("--score", default="all",
                        choices=["all"] + ALL_SCORES,
                        help="Gen-score variant on the x-axis, or 'all' (default: all).")
    parser.add_argument("--disc-shots", default="auto",
                        choices=DISC_TAGS,
                        help="Filter to a specific disc-shots strategy and tag the output "
                             "filename accordingly. 'auto' (default) plots whatever the "
                             "latest CSV contains and uses the unsuffixed filename.")
    args = parser.parse_args()

    bases = ALL_BASES if args.base == "all" else [args.base]
    scores = ALL_SCORES if args.score == "all" else [args.score]

    for base in bases:
        for score in scores:
            make_one_plot(base, score, disc_tag=args.disc_shots)


if __name__ == "__main__":
    main()
