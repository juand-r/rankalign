"""plot_length_confound.py — histograms of answer character length, correct vs incorrect.

Two panels (shared x-axis):
  top    — v2.1 (original, untransformed)
  bottom — v2.1correct-multi (correct rows stylized)

Reveals whether the stylization transforms introduced a length confound
between the correct and incorrect classes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_RA = Path(__file__).resolve().parents[2]
V21 = _RA / "data/humaneval/v2.1"
MULTI = _RA / "data/humaneval/v2.1correct-multi"
OUT = Path(__file__).parent


def load_lengths(data_dir: Path) -> tuple[list[int], list[int]]:
    """Load answer character lengths for correct and incorrect rows."""
    correct, wrong = [], []
    for fp in sorted(data_dir.glob("humaneval_*.csv")):
        df = pd.read_csv(fp)
        for _, row in df.iterrows():
            label = str(row.get("correct", "")).strip().lower()
            length = len(str(row.get("answer", "")))
            if label == "yes":
                correct.append(length)
            else:
                wrong.append(length)
    return correct, wrong


def main() -> None:
    print("Loading v2.1 ...")
    v21_corr, v21_wrong = load_lengths(V21)
    print(f"  correct={len(v21_corr)}  wrong={len(v21_wrong)}")

    print("Loading v2.1correct-multi ...")
    mul_corr, mul_wrong = load_lengths(MULTI)
    print(f"  correct={len(mul_corr)}  wrong={len(mul_wrong)}")

    # shared bin edges across both datasets and both classes
    all_vals = v21_corr + v21_wrong + mul_corr + mul_wrong
    lo, hi = 0, int(np.percentile(all_vals, 99))   # clip 1% outliers for readability
    bins = np.linspace(lo, hi, 60)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    for ax, (corr, wrong), title in zip(
        axes,
        [(v21_corr, v21_wrong), (mul_corr, mul_wrong)],
        ["v2.1 (original)", "v2.1correct-multi (correct rows stylized)"],
    ):
        ax.hist(corr,  bins=bins, alpha=0.5, label="correct",   color="steelblue",  density=True)
        ax.hist(wrong, bins=bins, alpha=0.5, label="incorrect", color="darkorange", density=True)
        ax.set_ylabel("Density")
        ax.set_title(title, fontsize=11)
        ax.legend(framealpha=0.7)

        # annotate medians
        mc = int(np.median(corr))
        mw = int(np.median(wrong))
        ymax = ax.get_ylim()[1]
        ax.axvline(mc, color="steelblue",  ls="--", lw=1.2, alpha=0.8)
        ax.axvline(mw, color="darkorange", ls="--", lw=1.2, alpha=0.8)
        ax.text(mc, ymax * 0.85, f"med={mc}", color="steelblue",
                fontsize=8, ha="center", va="top")
        ax.text(mw, ymax * 0.70, f"med={mw}", color="darkorange",
                fontsize=8, ha="center", va="top")

    axes[-1].set_xlabel("Answer length (characters)")
    fig.suptitle(
        "Answer length distribution: correct vs incorrect\n"
        "(dashed lines = medians)",
        fontsize=12,
    )

    out_path = OUT / "length_confound.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
