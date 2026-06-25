#!/usr/bin/env python3
"""Scatter: validator ROC (x) vs generator ROC (y), one point per (model,task,split) cell,
colored by SYSTEM. The y=x diagonal is the key reference:
  on the diagonal  -> generator ranks correctness exactly as well as the validator (gen tracks val)
  above            -> generator BEATS its own validator (RankAlign's regime)
  below            -> generator is worse than the validator (base; FLORA when validator is strong)

Reuses the validated discovery/scoring from build_predictor_comprehensive.py. Each system is
scored in its natural mode (non-TC + tc-self -> self/base-typ; tc-neg -> neg/base-typ).
Run on mll, qwen35 venv. Writes analysis/plots/val_vs_gen_scatter.png + a per-system summary.
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

SELF = ["basetyp", "self"]; NEG = ["basetypneg", "neg"]
ANY = ["basetyp", "self", "basetypneg", "neg", "plain"]
SYS = [("Base", "base", "base", ANY, "tab:gray"),
       ("RankAlign", "s2", "ep2", SELF, "tab:red"),
       ("New+fsx", "s3", "ep2", SELF, "tab:orange"),
       ("FLORA-PMI", "s4", "ep2", SELF, "tab:blue"),
       ("FLORA-Neg", "s7", "ep2", NEG, "tab:green")]
MODELS = ["gemma-2-2b", "gemma-2-2b-it", "gemma-2-9b-it", "gemma-2-27b-it", "qwen-3.5-9b", "gemma-4-31b"]
TASKS = ["ifeval", "hyponymy", "HE-upper", "HE-multi"]


def main():
    idx, _ = B.discover()
    pts = {name: [] for name, *_ in SYS}  # name -> [(val, gen, cell)]
    for mdl in MODELS:
        for task in TASKS:
            for split in ["train", "test"]:
                for name, s, ep, modes, _ in SYS:
                    files = B.cell_files(idx, mdl, task, split, s, ep, modes)
                    if not files:
                        continue
                    gen, _ = B.per_problem_roc(files, task, "gen_score_typcorr")
                    val, _ = B.per_problem_roc(files, task, "val_score")
                    if gen == gen and val == val:
                        pts[name].append((val, gen, f"{mdl}/{task}/{split[:2]}"))

    fig, ax = plt.subplots(figsize=(8.5, 8))
    ax.plot([40, 100], [40, 100], ls="--", color="k", lw=1, zorder=1, label="y = x (gen = val)")
    print(f"{'system':11} {'n':>3} {'mean gen-val':>12} {'pearson r(val,gen)':>20}")
    for name, s, ep, modes, col in SYS:
        P = pts[name]
        if not P:
            continue
        v = np.array([p[0] for p in P]); g = np.array([p[1] for p in P])
        ax.scatter(v, g, c=col, s=70, edgecolor="k", lw=0.6, alpha=0.85, label=f"{name} (n={len(P)})", zorder=3)
        gv = float(np.mean(g - v))
        r = pearsonr(v, g)[0] if len(P) >= 3 else float("nan")
        print(f"{name:11} {len(P):>3} {gv:>12.1f} {r:>20.2f}")
    ax.set_xlabel("Validator ROC ($\\times$100)   [discriminator: correct vs incorrect]")
    ax.set_ylabel("Generator ROC ($\\times$100)   [generator: correct vs incorrect]")
    ax.set_title("Validator ROC vs Generator ROC, per system\n(each point = one model/task/split cell; above y=x: gen beats its validator)")
    ax.set_xlim(45, 100); ax.set_ylim(45, 100); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = f"{B.PLOTS}/val_vs_gen_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
