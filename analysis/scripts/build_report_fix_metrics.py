#!/usr/bin/env python3
"""Per-problem recomputation of the original report's headline metrics (the "fix").

The original report (analysis/report.tex) computes Gen ROC AUC, concordance, Spearman and
Pearson by pooling ALL candidates for a (model, task, setting, epoch) into one set and taking a
single statistic (analyze_all_combos.py::compute_metrics). That conflates problems. Here we
instead compute each metric PER PROBLEM, then average across problems with a standard error.

Problem unit (confirmed against the data):
  - IFEval     -> group by `prompt`   (label col `correct`); ~79 prompts x 40 candidates.
  - Membership -> group by `category` (label col `label`);   ~68 categories x 18-96 candidates.
SE = std(per-problem values, ddof=1) / sqrt(n_problems).

Data: pooled-per-cell score files (already contain the grouping column) under
/datastor2/jdr/rankalign/outputs-trainset-dynamics/. NO GPU — pure re-aggregation.

Run on mll in the qwen35 venv:
    /datastor2/jdr/venvs/qwen35/bin/python analysis/scripts/build_report_fix_metrics.py
Outputs:
    analysis/tables/report_fix_metrics.csv        tidy: all combos/settings/epochs/modes
    analysis/tables/fix_{genroc,concordance,spearman,pearson}.tex   ep2 self-TC tables (mean+-SE)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# Reuse the original analyzer's file->(setting,epoch,tc_eval) matching backbone (import-safe:
# analyze_all_combos guards main() under __main__).
from analyze_all_combos import COMBO_CONFIG, find_files_for_combo  # noqa: E402

TABLES = HERE.parent / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

GROUP_KEY = {"membership": "category", "ifeval": "prompt"}
LABEL_COL = {"membership": "label", "ifeval": "correct"}
# display order of the 4 model/task cells, matching the original report
CELLS = [("gemma_membership", "Gemma / Membership"), ("gemma_ifeval", "Gemma / IFEval"),
         ("qwen_membership", "Qwen / Membership"), ("qwen_ifeval", "Qwen / IFEval")]
SETTINGS = ["base", "s1", "s2", "s3", "s4"]


def _labels(s):
    return s.astype(str).str.strip().str.lower().map(
        {"yes": 1, "true": 1, "1": 1, "no": 0, "false": 0, "0": 0}).to_numpy(dtype=float)


def _concordance(gen, val):
    """Fraction of candidate pairs whose (gen, val) differences share sign (per problem)."""
    dg = np.subtract.outer(gen, gen)
    dv = np.subtract.outer(val, val)
    iu = np.triu_indices(len(gen), k=1)
    dg, dv = dg[iu], dv[iu]
    m = (dg != 0) & (dv != 0)
    return float(np.mean(np.sign(dg[m]) == np.sign(dv[m]))) if m.any() else np.nan


def per_problem_metrics(filepath, task_label, gen_col="gen_score_typcorr"):
    """Per-problem ROC/concordance/Spearman/Pearson, returned as mean+-SE across problems."""
    df = pd.read_csv(filepath)
    key, lc = GROUP_KEY[task_label], LABEL_COL[task_label]
    if key not in df.columns or lc not in df.columns:
        return None
    rocs, valrocs, concs, sps, prs = [], [], [], [], []
    for _, g in df.groupby(key):
        y = _labels(g[lc])
        gen = pd.to_numeric(g[gen_col], errors="coerce").to_numpy()
        val = pd.to_numeric(g["val_score"], errors="coerce").to_numpy()
        ok = ~(np.isnan(y) | np.isnan(gen) | np.isnan(val))
        y, gen, val = y[ok], gen[ok], val[ok]
        if len(y) >= 2 and len(set(y)) == 2:
            rocs.append(roc_auc_score(y, gen))
            valrocs.append(roc_auc_score(y, val))
            concs.append(_concordance(gen, val))
        if len(gen) >= 3 and np.std(gen) > 0 and np.std(val) > 0:
            sps.append(spearmanr(gen, val).correlation)
            prs.append(pearsonr(gen, val)[0])

    def ms(a):
        a = [x for x in a if x == x]
        if not a:
            return np.nan, np.nan, 0
        return float(np.mean(a)), (float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0), len(a)

    out = {}
    for nm, a in [("gen_roc", rocs), ("val_roc", valrocs), ("concordance", concs),
                  ("spearman", sps), ("pearson", prs)]:
        out[f"{nm}_mean"], out[f"{nm}_se"], out[f"{nm}_n"] = ms(a)
    return out


def build_tidy():
    rows = []
    for combo, cfg in COMBO_CONFIG.items():
        task = cfg["task_label"]
        inv = find_files_for_combo(combo)
        if inv.empty:
            print(f"  [warn] no files for {combo}")
            continue
        for _, r in inv.iterrows():
            ep = "base" if r["epoch"] == -1 else f"ep{r['epoch']}"
            m = per_problem_metrics(r["file"], task)
            if m is None:
                print(f"  [skip] bad cols: {Path(r['file']).name}")
                continue
            rows.append({"combo": combo, "model": cfg["model_label"], "task": task,
                         "setting": r["setting"], "epoch": ep, "tc_eval": r["tc_eval"], **m})
    df = pd.DataFrame(rows)
    out = TABLES / "report_fix_metrics.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} rows)")
    return df


def _fmt(mean, se, best):
    if mean != mean:
        return "---"
    body = f"\\mathbf{{{mean:.3f}}}" if best else f"{mean:.3f}"
    return f"${body}\\se{{{se:.3f}}}$"


def emit_table(df, metric, caption, label, path):
    """Emit a 4-cell x {base,s1..s4} table of per-problem mean+-SE at ep2, self-TC."""
    sub = df[(df.tc_eval == "self") & ((df.epoch == "ep2") | (df.setting == "base"))]
    lines = [f"% Auto-generated by build_report_fix_metrics.py. metric={metric}",
             "\\begin{tabular}{lccccc}", "\\toprule",
             "\\textbf{Model / Task} & \\textbf{Base} & \\textbf{s1} & \\textbf{s2} & "
             "\\textbf{s3} & \\textbf{s4} \\\\", "\\midrule"]
    for combo, disp in CELLS:
        c = sub[sub.combo == combo]
        means = {}
        for s in SETTINGS:
            row = c[c.setting == s]
            means[s] = row.iloc[0][f"{metric}_mean"] if len(row) else np.nan
        valid = {s: v for s, v in means.items() if v == v}
        best_s = max(valid, key=valid.get) if valid else None
        cells = []
        for s in SETTINGS:
            row = c[c.setting == s]
            if not len(row):
                cells.append("---")
            else:
                cells.append(_fmt(row.iloc[0][f"{metric}_mean"], row.iloc[0][f"{metric}_se"], s == best_s))
        lines.append(f"{disp} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


PLOTS = HERE.parent / "plots"
REPO = HERE.parent.parent
# Unified figure: natural eval mode per setting (s7 neg-TC, rest self-TC), as in the original.
UNIFIED_SETTINGS = ["base", "s1", "s2", "s3", "s4", "s7"]
NAT_MODE = {"base": "self", "s1": "self", "s2": "self", "s3": "self", "s4": "self", "s7": "neg"}
# rosch test table (already per-category): method name -> setting
ROSCH_METHOD_MAP = {"Base": "base", "SFT labelonly 10%": "s1", "RankAlign": "s2",
                    "New + fsx [-TC]": "s3", "New + PMI + fsx": "s4", "New + NegTC + fsx": "s7"}


def build_unified_figure(df):
    """Grouped bar of per-problem mean gen ROC +- SE, 4 cells x {base,s1,s2,s3,s4,s7}."""
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ncell, nset = len(CELLS), len(UNIFIED_SETTINGS)
    width = 0.8 / nset
    for si, s in enumerate(UNIFIED_SETTINGS):
        means, ses = [], []
        for combo, _ in CELLS:
            sub = df[(df.combo == combo) & (df.setting == s) & (df.tc_eval == NAT_MODE[s]) &
                     ((df.epoch == "ep2") | (df.setting == "base"))]
            if len(sub):
                means.append(sub.iloc[0]["gen_roc_mean"]); ses.append(sub.iloc[0]["gen_roc_se"])
            else:
                means.append(np.nan); ses.append(np.nan)
        x = np.arange(ncell) + si * width
        ax.bar(x, means, width, yerr=ses, capsize=3, label=s)
    ax.axhline(0.5, ls="--", color="gray", lw=1, label="chance")
    ax.set_xticks(np.arange(ncell) + 0.4 - width / 2)
    ax.set_xticklabels([d for _, d in CELLS])
    ax.set_ylabel("Gen ROC AUC (per-problem mean $\\pm$ SE)")
    ax.set_title("Generator ROC AUC by setting (ep2, per-problem; s7 neg-TC, rest self-TC)")
    ax.set_ylim(0.4, 1.0)
    ax.legend(ncol=7, fontsize=8, loc="lower right")
    fig.tight_layout()
    out = PLOTS / "fix_unified_genroc.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def build_train_vs_test(df):
    """gemma membership TRAIN (per-category) vs rosch TEST (per-category), PMI-base, mean+-SE."""
    rosch_csv = REPO / "metrics-from-scores-rerun-only" / "rosch_v7_9b-it_gen_roc_table_long.csv"
    if not rosch_csv.exists():
        print(f"  [skip train-vs-test] no {rosch_csv}")
        return
    r = pd.read_csv(rosch_csv)
    r = r[r["column"] == "PMI base"].copy()
    r["setting"] = r["method"].map(ROSCH_METHOD_MAP)
    test = {}
    for s, g in r.dropna(subset=["setting"]).groupby("setting"):
        v = g["value"].to_numpy()
        test[s] = (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0, len(v))
    # train: gemma membership, self (=PMI base on the basetyp- files), ep2 (base for base)
    tr = df[(df.combo == "gemma_membership") & (df.tc_eval == "self") &
            ((df.epoch == "ep2") | (df.setting == "base"))]
    settings = [s for s in ["base", "s1", "s2", "s3", "s4", "s7"] if s in test or s == "base"]
    lines = ["% Auto-generated by build_report_fix_metrics.py (train-vs-test, per-category PMI-base)",
             "\\begin{tabular}{lccc}", "\\toprule",
             "\\textbf{Setting} & \\textbf{Train (membership)} & \\textbf{Test (rosch)} & "
             "\\textbf{$\\Delta$} \\\\", "\\midrule"]
    for s in settings:
        trow = tr[tr.setting == s]
        if not len(trow):
            continue
        tm, tse = trow.iloc[0]["gen_roc_mean"], trow.iloc[0]["gen_roc_se"]
        if s in test:
            em, ese, _ = test[s]
            d = em - tm  # test - train (negative = generalization drop), matching original
            lines.append(f"{s} & ${tm:.3f}\\se{{{tse:.3f}}}$ & ${em:.3f}\\se{{{ese:.3f}}}$ & "
                         f"${d:+.3f}$ \\\\")
        else:
            lines.append(f"{s} & ${tm:.3f}\\se{{{tse:.3f}}}$ & --- & --- \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "fix_train_vs_test.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {TABLES / 'fix_train_vs_test.tex'}")


def main():
    df = build_tidy()
    if df.empty:
        print("No data."); return
    emit_table(df, "gen_roc", "Gen ROC AUC (TC, self-scoring), per-problem mean$\\pm$SE, ep2.",
               "tab:fix_main", TABLES / "fix_genroc.tex")
    emit_table(df, "concordance", "Concordance, per-problem mean$\\pm$SE, ep2.",
               "tab:fix_concordance", TABLES / "fix_concordance.tex")
    emit_table(df, "spearman", "Spearman, per-problem mean$\\pm$SE, ep2.",
               "tab:fix_spearman", TABLES / "fix_spearman.tex")
    emit_table(df, "pearson", "Pearson, per-problem mean$\\pm$SE, ep2.",
               "tab:fix_pearson", TABLES / "fix_pearson.tex")
    build_unified_figure(df)
    build_train_vs_test(df)
    # quick console view of the headline (gen ROC)
    print("\n=== Q1 gen ROC (per-problem mean +- SE, ep2, self-TC) ===")
    sub = df[(df.tc_eval == "self") & ((df.epoch == "ep2") | (df.setting == "base"))]
    for combo, disp in CELLS:
        c = sub[sub.combo == combo]
        cells = []
        for s in SETTINGS:
            row = c[c.setting == s]
            if len(row):
                cells.append(f"{s}={row.iloc[0]['gen_roc_mean']:.3f}+-{row.iloc[0]['gen_roc_se']:.3f}(n{int(row.iloc[0]['gen_roc_n'])})")
        print(f"  {disp:20s} " + "  ".join(cells))


if __name__ == "__main__":
    main()
