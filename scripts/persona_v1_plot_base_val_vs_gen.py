#!/usr/bin/env python3
"""
2x3 scatter grid: x = gen_score (raw), y = val_score, one subplot per persona,
for the base model (default: gemma-2-9b-it), persona-v1.

v1 vs v0:
- DROPPED `subscribes-to-moral-nihilism` and `believes-life-has-no-meaning`
  (poor validator accuracy on v0).
- Labels for `psychopathy`, `machiavellianism`, `narcissism` are FLIPPED
  upstream in the dataset builder so that "yes" = NOT antisocial. The plot
  code itself is unchanged; it just reads the v1 CSVs.

Reads `scores_self-v6-google_<base>_persona-v1-<slug>_test_*.csv` files under
outputs/ (raw gen_score is identical between self- and neg- eval CSVs; we
just pick self-).

Usage:
  python scripts/persona_v1_plot_base_val_vs_gen.py                # 9b-it
  python scripts/persona_v1_plot_base_val_vs_gen.py --base 2b-it
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="9b-it",
                        choices=["9b-it", "2b-it", "2b"],
                        help="Which base model to plot (default: 9b-it).")
    args = parser.parse_args()

    model_tag = f"v6-google_gemma-2-{args.base}"
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), sharex=False, sharey=False)
    axes = axes.flatten()

    for ax, persona in zip(axes, PERSONAS):
        candidates = list(OUTPUTS_DIR.glob(
            f"scores_self-{model_tag}_persona-v1-{persona}_test_log-odds_tc_*.csv"))
        if not candidates:
            ax.set_title(f"{persona}\n(no file)", fontsize=10)
            ax.axis("off")
            continue
        df = pd.read_csv(sorted(candidates)[-1])
        labels = (df["correct"].str.strip().str.lower() == "yes").astype(int)
        # x = gen_score (raw), y = val_score
        x = df["gen_score"].values
        y = df["val_score"].values

        mask = ~(np.isnan(x) | np.isnan(y))
        x, y, labels_v = x[mask], y[mask], labels.values[mask]

        ax.scatter(x[labels_v == 1], y[labels_v == 1],
                   s=14, alpha=0.6, c="tab:red", label="yes")
        ax.scatter(x[labels_v == 0], y[labels_v == 0],
                   s=14, alpha=0.6, c="tab:blue", label="no")
        ax.axhline(0, color="gray", lw=0.6, alpha=0.5)

        try:
            # Gen-AUC is computed against the raw gen_score (now on x-axis).
            auc = roc_auc_score(labels_v, x)
        except Exception:
            auc = float("nan")
        try:
            r, _ = pearsonr(x, y)
        except Exception:
            r = float("nan")

        ax.set_title(f"{persona}\nGen-AUC={auc:.2f}  r={r:.2f}  n={len(x)}",
                     fontsize=10)
        ax.set_xlabel("gen_score (raw, log-prob)")
        ax.set_ylabel("val_score (log-odds)")
        ax.grid(alpha=0.25)

    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor="tab:red", markersize=7, label="label=yes"),
               plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor="tab:blue", markersize=7, label="label=no")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.995),
               ncol=2, frameon=False, fontsize=10)
    fig.suptitle(
        f"Base gemma-2-{args.base}: gen_score (raw) vs val_score, per persona-v1 task (test split)",
        fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"persona_v1_base_gemma-2-{args.base}_gen_raw_vs_val.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
