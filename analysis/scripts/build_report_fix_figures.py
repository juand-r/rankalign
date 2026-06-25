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


def _pairwise_deltas(combo, setting):
    """EXACT port of the original report's compute_delta_histograms: pool all candidates in the
    cell, sample ~100k random pairs, |score_i - score_j| for generator and validator."""
    f = _file(combo, setting, "ep2")
    if f is None:
        return None, None
    df = pd.read_csv(f)
    gen = pd.to_numeric(df["gen_score_typcorr"], errors="coerce").dropna().to_numpy()
    val = pd.to_numeric(df["val_score"], errors="coerce").dropna().to_numpy()
    np.random.seed(42)

    def d(a):
        n = len(a)
        if n < 2:
            return np.array([])
        npairs = min(100000, n * (n - 1) // 2)
        i = np.random.randint(0, n, npairs); j = np.random.randint(0, n, npairs); m = i != j
        return np.abs(a[i[m]] - a[j[m]])
    return d(gen), d(val)


def fig_score_delta_hist():
    """Pairwise |Δ| Gen vs Val, same procedure as the original report (score-spread/calibration)."""
    setts = ["s2", "s3", "s4", "s7"]
    fig, axes = plt.subplots(len(CELLS), len(setts), figsize=(4 * len(setts), 3 * len(CELLS)), squeeze=False)
    for ri, (combo, disp) in enumerate(CELLS):
        cache = {s: _pairwise_deltas(combo, s) for s in setts}
        allv = [x for gv in cache.values() for x in gv if x is not None and len(x)]
        xmax = float(np.percentile(np.concatenate(allv), 99)) if allv else 10.0
        for ci, s in enumerate(setts):
            ax = axes[ri][ci]; gd, vd = cache[s]
            if gd is None or not len(gd):
                ax.set_title(f"{disp} {SET_LABEL[s]}\n(no data)"); continue
            ax.hist(gd, bins=50, alpha=0.6, density=True, range=(0, xmax), color="blue",
                    label=f"Gen |$\\Delta$| ($\\mu$={gd.mean():.1f})")
            ax.hist(vd, bins=50, alpha=0.6, density=True, range=(0, xmax), color="orange",
                    label=f"Val |$\\Delta$| ($\\mu$={vd.mean():.1f})")
            ax.set_xlim(0, xmax); ax.set_title(f"{disp} {SET_LABEL[s]}")
            ax.legend(fontsize=6); ax.set_xlabel("|score$_i$ - score$_j$|")
    plt.suptitle("Pairwise score-delta distributions (ep2): Gen vs Val |$\\Delta$|")
    plt.tight_layout(); fig.savefig(PLOTS / "fix_score_delta_hist.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote fix_score_delta_hist.png")


def fig_scatter():
    """Generator vs validator scatter, gemma-2-9b-it / membership: rows = scoring variant
    (none / tc self / tc self base), cols = settings. NOTE: the original report only ran
    base-typicality evals (basetyp-), so "tc self" (own-model, self/no-base) has NO data and is
    shown as 'not evaluated' -- only "none" (raw gen_score) and "tc self base" are available."""
    combo = "gemma_membership"; lc = LABEL_COL["membership"]
    setts = ["base", "s2", "s3", "s4"]
    variants = [("none", "gen_score"), ("tc self", None), ("tc self base", "gen_score_typcorr")]
    fig, axes = plt.subplots(len(variants), len(setts),
                             figsize=(3 * len(setts), 2.8 * len(variants)), squeeze=False)
    for ri, (vname, col) in enumerate(variants):
        for ci, s in enumerate(setts):
            ax = axes[ri][ci]
            f = _file(combo, s, "ep2")  # basetyp (self/base-typ) file
            if col is None or f is None:
                msg = "not evaluated\n(no self/no-base scores)" if col is None else "no data"
                ax.set_title(f"{SET_LABEL[s]} / {vname}", fontsize=8)
                ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=7, color="gray",
                        transform=ax.transAxes); ax.set_xticks([]); ax.set_yticks([]); continue
            df = pd.read_csv(f); y = _labels(df[lc])
            g = pd.to_numeric(df[col], errors="coerce"); v = pd.to_numeric(df["val_score"], errors="coerce")
            ax.scatter(g[y == 1], v[y == 1], s=5, alpha=.25, c="green", label="member")
            ax.scatter(g[y == 0], v[y == 0], s=5, alpha=.25, c="red", label="non-member")
            ax.set_title(f"{SET_LABEL[s]} / {vname}", fontsize=8)
            if ci == 0:
                ax.set_ylabel(f"{vname}\nval", fontsize=8)
            if ri == len(variants) - 1:
                ax.set_xlabel("gen", fontsize=8)
            if ri == 0 and ci == 0:
                ax.legend(fontsize=6, markerscale=2)
    plt.suptitle("gemma-2-9b-it / membership: generator vs validator (ep2) "
                 "(rows: none / tc self / tc self base; cols: settings)")
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


METHOD_MAP_TT = {"Base": "base", "SFT labelonly 10%": "s1", "RankAlign": "s2",
                 "New + fsx [-TC]": "s3", "New + PMI + fsx": "s4"}


def _test_means(csv_path):
    """PMI-base per-problem test mean+-SE per setting from a precomputed long table."""
    if not csv_path.exists():
        return {}
    r = pd.read_csv(csv_path); r = r[r["column"] == "PMI base"].copy()
    r["setting"] = r["method"].map(METHOD_MAP_TT)
    out = {}
    for s, g in r.dropna(subset=["setting"]).groupby("setting"):
        v = g["value"].to_numpy()
        out[s] = (v.mean(), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
    return out


def fig_train_vs_test():
    """Two panels: membership TRAIN vs rosch TEST; ifeval TRAIN vs ifeval-OOD TEST (per-problem)."""
    REPO = HERE.parent.parent
    panels = [("gemma_membership", REPO / "metrics-from-scores-rerun-only" / "rosch_v7_9b-it_gen_roc_table_long.csv",
               "membership train vs rosch test"),
              ("gemma_ifeval", REPO / "metrics-from-scores" / "ifeval_v7_ood_9b-it_gen_roc_table_long.csv",
               "ifeval train vs ifeval-OOD test")]
    setts = ["base", "s1", "s2", "s3", "s4"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (combo, test_csv, title) in zip(axes, panels):
        test = _test_means(test_csv)
        tr_m, tr_e, te_m, te_e, labs = [], [], [], [], []
        for s in setts:
            a = _per_problem(combo, s, "ep2", _roc)
            if not len(a):
                continue
            labs.append(s)
            tr_m.append(a.mean()); tr_e.append(a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0)
            te_m.append(test.get(s, (np.nan, np.nan))[0]); te_e.append(test.get(s, (np.nan, np.nan))[1])
        x = np.arange(len(labs)); w = 0.38
        ax.bar(x - w / 2, tr_m, w, yerr=tr_e, capsize=3, label="train")
        ax.bar(x + w / 2, te_m, w, yerr=te_e, capsize=3, label="test (held-out)")
        ax.axhline(0.5, ls="--", color="gray", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylim(0.4, 1.0)
        ax.set_ylabel("gen ROC (per-problem mean $\\pm$ SE)"); ax.set_title(title); ax.legend(fontsize=9)
    plt.suptitle("Train vs held-out test (per-problem, PMI-base, gemma-2-9b-it)")
    plt.tight_layout(); fig.savefig(PLOTS / "fix_train_vs_test.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote fix_train_vs_test.png")


def main():
    fig_train_vs_test()
    fig_unified_concordance()
    fig_tc_dynamics()
    fig_score_delta_hist()
    fig_scatter()
    fig_tc_comparison()
    print("DONE")


if __name__ == "__main__":
    main()
