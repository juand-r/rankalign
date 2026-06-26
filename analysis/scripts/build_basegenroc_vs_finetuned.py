#!/usr/bin/env python3
"""Predictor view #2: BASE (untrained model) GENERATOR ROC on the x-axis vs the FINAL fine-tuned
(epoch 2) Generator ROC (left) and Validator ROC (right) on the y-axis, one point per
(model, task, split) cell, colored by system.

Excludes the meaningless ifeval / 2b cells (base model is at chance there).
Reuses validated machinery from build_predictor_comprehensive.py. Systems scored in natural mode.
Run on mll. Writes analysis/plots/basegenroc_vs_finetuned.png.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_predictor_comprehensive as B

SELF = ["basetyp", "self"]; NEG = ["basetypneg", "neg"]; ANY = ["basetyp", "self", "basetypneg", "neg", "plain"]
SYS = [("RankAlign", "s2", SELF, "tab:red"), ("New+fsx", "s3", SELF, "tab:orange"),
       ("FLORA-PMI", "s4", SELF, "tab:blue"), ("FLORA-Neg", "s7", NEG, "tab:green")]
MODELS = ["gemma-2-2b", "gemma-2-2b-it", "gemma-2-9b-it", "gemma-2-27b-it", "qwen-3.5-9b", "gemma-4-31b"]
TASKS = ["ifeval", "hyponymy", "HE-upper", "HE-multi"]


def excluded(mdl, task):
    return ("2b" in mdl) and (task == "ifeval")  # meaningless: base at chance


def main():
    idx, _ = B.discover()
    data = {name: [] for name, *_ in SYS}
    for mdl in MODELS:
        for task in TASKS:
            if excluded(mdl, task):
                continue
            for split in ["train", "test"]:
                base_gen, _ = B.per_problem_roc(B.cell_files(idx, mdl, task, split, "base", "base", ANY), task, "gen_score_typcorr")
                if base_gen != base_gen:
                    continue
                for name, s, modes, _ in SYS:
                    files = B.cell_files(idx, mdl, task, split, s, "ep2", modes)
                    if not files:
                        continue
                    gen, _ = B.per_problem_roc(files, task, "gen_score_typcorr")
                    val, _ = B.per_problem_roc(files, task, "val_score")
                    if gen == gen or val == val:
                        data[name].append((base_gen, gen, val))

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True)
    for ax, (yi, ylab) in zip(axes, [(1, "Fine-tuned GENERATOR ROC (ep2)"), (2, "Fine-tuned VALIDATOR ROC (ep2)")]):
        ax.plot([45, 100], [45, 100], ls="--", color="gray", lw=1, zorder=1, label="y = x")
        for name, s, modes, col in SYS:
            pts = [(p[0], p[yi]) for p in data[name] if p[yi] == p[yi]]
            if not pts:
                continue
            x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
            ax.scatter(x, y, c=col, s=60, edgecolor="k", lw=0.5, alpha=0.85, label=f"{name} (n={len(pts)})", zorder=3)
        ax.set_xlabel("BASE generator ROC ($\\times$100)  [untrained, a-priori predictor]")
        ax.set_ylabel(ylab); ax.set_xlim(45, 100); ax.set_ylim(45, 100)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_title("Does base GENERATOR ROC predict fine-tuned GEN ROC?")
    axes[1].set_title("Does base GENERATOR ROC predict fine-tuned VAL ROC?")
    fig.suptitle("Base (untrained) GENERATOR ROC  vs  final fine-tuned Gen / Val ROC, per system\n"
                 "(each point = one model/task/split cell; ifeval-2b excluded)")
    fig.tight_layout()
    out = f"{B.PLOTS}/basegenroc_vs_finetuned.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"{'system':11} {'n':>3} {'r(baseGEN, ft-GEN)':>20} {'r(baseGEN, ft-VAL)':>20}")
    for name, *_ in SYS:
        d = data[name]
        if len(d) < 3:
            continue
        bg = np.array([p[0] for p in d]); g = np.array([p[1] for p in d]); v = np.array([p[2] for p in d])
        mg = g == g; mv = v == v
        rg = pearsonr(bg[mg], g[mg])[0] if mg.sum() >= 3 else float("nan")
        rv = pearsonr(bg[mv], v[mv])[0] if mv.sum() >= 3 else float("nan")
        print(f"{name:11} {len(d):>3} {rg:>20.2f} {rv:>20.2f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
