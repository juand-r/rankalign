#!/usr/bin/env python3
"""Per-problem figures for report_fix.tex, using the SAME procedure as the HumanEval report
(analysis/scripts/build_he_report_figures.py): every aggregate is computed per problem (IFEval
prompt / membership category) then averaged, rather than on the pooled candidate set.

The original report's figures (analyze_all_combos.py / plot_tc_comparison.py) instead plot the
GLOBAL-POOL tc_gen_roc / a pooled pairwise gen_delta. This script replaces the aggregate ones so
both reports use one procedure. (gen-vs-val scatter and per-category were already per-problem /
raw-pooled in both reports; the scatter is regenerated here only for style parity.)

Run on mll, qwen35 venv:
    /datastor2/jdr/venvs/qwen35/bin/python analysis/scripts/build_report_fix_figures.py
Outputs into analysis/plots/:
    fix_unified_concordance.png   per-problem concordance bar (mean +- SE)
    fix_tc_dynamics.png           per-problem tc gen ROC vs epoch, 4 cells (s2/s3/s4/s7)
    fix_score_delta_hist.png      per-problem (tc-gen pos mean - neg mean) histograms
    fix_gen_vs_val_scatter.png    pooled raw gen(tc) vs val scatter (parity with HE)
    fix_tc_comparison.png         per-problem tc gen ROC: TC settings vs s3 (gemma membership)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_all_combos import COMBO_CONFIG, find_files_for_combo  # noqa: E402
from build_report_fix_metrics import GROUP_KEY, LABEL_COL, _labels, _concordance, CELLS  # noqa: E402

PLOTS = HERE.parent / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)
SET_LABEL = {"base": "Base", "s1": "SFT", "s2": "RankAlign", "s3": "New+fsx", "s4": "TC-self",
             "s5": "TC-self basic", "s6": "TC-self fsx", "s7": "Neg-TC", "s11": "TC-self vlo",
             "s12": "Neg-TC vlo"}
# natural eval mode per setting (neg-TC settings scored neg; rest self), as in the report
NEG_SETTINGS = {"s7", "s12"}
EPS = ["base", "ep0", "ep1", "ep2"]


def _index(combo):
    """(setting, epoch_str, tc_eval) -> filepath, for one combo."""
    inv = find_files_for_combo(combo)
    idx = {}
    for _, r in inv.iterrows():
        ep = "base" if r["epoch"] == -1 else f"ep{r['epoch']}"
        idx[(r["setting"], ep, r["tc_eval"])] = r["file"]
    return idx


IDX = {combo: _index(combo) for combo in COMBO_CONFIG}


def _file(combo, setting, ep):
    mode = "neg" if setting in NEG_SETTINGS else "self"
    if setting == "base":
        return IDX[combo].get(("base", "base", mode))
    return IDX[combo].get((setting, ep, mode))


def _per_problem(combo, setting, ep, fn):
    """Apply fn(group_df) per problem; return array of finite values."""
    f = _file(combo, setting, ep)
    if f is None:
        return np.array([])
    df = pd.read_csv(f)
    key, lc = GROUP_KEY[COMBO_CONFIG[combo]["task_label"]], LABEL_COL[COMBO_CONFIG[combo]["task_label"]]
    if key not in df.columns or lc not in df.columns:
        return np.array([])
    out = []
    for _, g in df.groupby(key):
        v = fn(g, lc)
        if v == v:
            out.append(v)
    return np.array(out)


def _roc(g, lc):
    y = _labels(g[lc]); gen = pd.to_numeric(g["gen_score_typcorr"], errors="coerce").to_numpy()
    ok = ~(np.isnan(y) | np.isnan(gen)); y, gen = y[ok], gen[ok]
    return roc_auc_score(y, gen) if (len(y) >= 2 and len(set(y)) == 2) else np.nan


def _conc(g, lc):
    gen = pd.to_numeric(g["gen_score_typcorr"], errors="coerce").to_numpy()
    val = pd.to_numeric(g["val_score"], errors="coerce").to_numpy()
    ok = ~(np.isnan(gen) | np.isnan(val))
    return _concordance(gen[ok], val[ok]) if ok.sum() >= 2 else np.nan


def _delta(g, lc):
    y = _labels(g[lc]); gen = pd.to_numeric(g["gen_score_typcorr"], errors="coerce").to_numpy()
    ok = ~(np.isnan(y) | np.isnan(gen)); y, gen = y[ok], gen[ok]
    if (y == 1).sum() and (y == 0).sum():
        return gen[y == 1].mean() - gen[y == 0].mean()
    return np.nan


def fig_unified_concordance():
    setts = ["base", "s1", "s2", "s3", "s4", "s7"]
    fig, ax = plt.subplots(figsize=(12, 5)); n = len(CELLS); w = 0.8 / len(setts)
    for si, s in enumerate(setts):
        means, ses = [], []
        for combo, _ in CELLS:
            a = _per_problem(combo, s, "ep2", _conc)
            means.append(a.mean() if len(a) else np.nan)
            ses.append(a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0)
        ax.bar(np.arange(n) + si * w, means, w, yerr=ses, capsize=3, label=s)
    ax.axhline(0.5, ls="--", color="gray", lw=1, label="chance")
    ax.set_xticks(np.arange(n) + 0.4 - w / 2); ax.set_xticklabels([d for _, d in CELLS])
    ax.set_ylabel("Concordance (per-problem mean $\\pm$ SE)"); ax.set_ylim(0.3, 0.9)
    ax.set_title("Generator--validator concordance by setting (ep2, per-problem)")
    ax.legend(ncol=7, fontsize=8, loc="lower right"); fig.tight_layout()
    fig.savefig(PLOTS / "fix_unified_concordance.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote fix_unified_concordance.png")


def fig_tc_dynamics():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9)); axes = axes.flatten()
    for ax, (combo, disp) in zip(axes, CELLS):
        for s in ["s2", "s3", "s4", "s7"]:
            ys = []
            for ep in EPS:
                a = _per_problem(combo, "base" if ep == "base" else s, ep, _roc)
                ys.append(a.mean() * 100 if len(a) else np.nan)
            ax.plot(EPS, ys, marker="o", label=SET_LABEL[s])
        ax.set_title(disp); ax.set_ylabel("tc gen ROC x100 (per-problem)"); ax.legend(fontsize=8)
    plt.suptitle("Per-epoch dynamics (per-problem tc gen ROC)")
    plt.tight_layout(); fig.savefig(PLOTS / "fix_tc_dynamics.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote fix_tc_dynamics.png")


def fig_score_delta_hist():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9)); axes = axes.flatten()
    for ax, (combo, disp) in zip(axes, CELLS):
        for s in ["s2", "s4", "s7"]:
            a = _per_problem(combo, s, "ep2", _delta)
            if len(a):
                ax.hist(a, bins=20, alpha=0.5, label=SET_LABEL[s])
        ax.axvline(0, color="k", lw=0.6); ax.set_title(disp); ax.legend(fontsize=8)
        ax.set_xlabel("per-problem (tc-gen pos mean $-$ neg mean)")
    plt.suptitle("Score-delta distributions (per-problem, ep2)")
    plt.tight_layout(); fig.savefig(PLOTS / "fix_score_delta_hist.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote fix_score_delta_hist.png")


def fig_scatter():
    """Pooled raw gen(tc) vs val, gemma membership base/s2/s4 (parity with HE scatter)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    combo = "gemma_membership"; lc = LABEL_COL["membership"]
    for ax, s in zip(axes, ["base", "s2", "s4"]):
        f = _file(combo, s, "ep2")
        if f is None:
            ax.set_title(f"{SET_LABEL[s]} (no data)"); continue
        df = pd.read_csv(f); y = _labels(df[lc])
        g = pd.to_numeric(df["gen_score_typcorr"], errors="coerce")
        v = pd.to_numeric(df["val_score"], errors="coerce")
        ax.scatter(g[y == 1], v[y == 1], s=8, alpha=.3, c="green", label="member")
        ax.scatter(g[y == 0], v[y == 0], s=8, alpha=.3, c="red", label="non-member")
        ax.set_xlabel("gen (tc)"); ax.set_ylabel("val"); ax.set_title(SET_LABEL[s]); ax.legend(fontsize=8)
    plt.suptitle("gemma-2-9b-it / membership gen(tc) vs val (ep2, pooled points)")
    plt.tight_layout(); fig.savefig(PLOTS / "fix_gen_vs_val_scatter.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote fix_gen_vs_val_scatter.png")


def fig_tc_comparison():
    """Per-problem tc gen ROC for TC settings vs s3 (gemma membership, ep2)."""
    combo = "gemma_membership"
    setts = ["s3", "s4", "s5", "s6", "s11", "s7", "s12"]
    means, ses, labs = [], [], []
    for s in setts:
        a = _per_problem(combo, s, "ep2", _roc)
        if not len(a):
            continue
        means.append(a.mean() * 100); ses.append(a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0)
        labs.append(f"{s} {SET_LABEL.get(s, '')}")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["gray" if l.startswith("s3") else ("steelblue" if l.split()[0] not in ("s7", "s12") else "indianred") for l in labs]
    y = np.arange(len(labs))
    ax.barh(y, means, xerr=ses, color=colors, alpha=0.85, capsize=3)
    ax.set_yticks(y); ax.set_yticklabels(labs)
    if means:
        s3v = means[labs.index(next(l for l in labs if l.startswith("s3")))]
        ax.axvline(s3v, color="k", ls="--", lw=1, label="s3 (no TC)"); ax.legend(fontsize=8)
    ax.set_xlabel("tc gen ROC x100 (per-problem mean $\\pm$ SE)")
    ax.set_title("TC settings vs s3 (gemma membership, ep2, per-problem)")
    plt.tight_layout(); fig.savefig(PLOTS / "fix_tc_comparison.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote fix_tc_comparison.png")


def main():
    fig_unified_concordance()
    fig_tc_dynamics()
    fig_score_delta_hist()
    fig_scatter()
    fig_tc_comparison()
    print("DONE")


if __name__ == "__main__":
    main()
