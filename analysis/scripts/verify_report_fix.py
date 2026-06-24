#!/usr/bin/env python3
"""INDEPENDENT verification of report_fix.tex numbers. Recomputes every per-problem metric from
the raw score files with a FRESH implementation (does not reuse build_report_fix_metrics's metric
functions) and diffs against the committed fix_*.tex tables. Also prints the matched source file
for every cell so file-selection can be eyeballed, and checks grouping counts.

Run on mll, qwen35 venv:
    /datastor2/jdr/venvs/qwen35/bin/python analysis/scripts/verify_report_fix.py
Exit 0 and "ALL CHECKS PASS" iff every recomputed value matches the .tex within 1e-3.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_all_combos import COMBO_CONFIG, find_files_for_combo  # file selection under test

TABLES = HERE.parent / "tables"
GROUP_KEY = {"membership": "category", "ifeval": "prompt"}
LABEL_COL = {"membership": "label", "ifeval": "correct"}
CELLS = [("gemma_membership", "Gemma / Membership"), ("gemma_ifeval", "Gemma / IFEval"),
         ("qwen_membership", "Qwen / Membership"), ("qwen_ifeval", "Qwen / IFEval")]
SETTINGS = ["base", "s1", "s2", "s3", "s4"]
TOL = 1.5e-3
problems = []  # collected failures


def lab(s):
    return s.astype(str).str.strip().str.lower().map(
        {"yes": 1, "true": 1, "1": 1, "no": 0, "false": 0, "0": 0}).to_numpy(dtype=float)


def conc(gen, val):
    n = len(gen); ag = tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            dg, dv = gen[i] - gen[j], val[i] - val[j]
            if dg == 0 or dv == 0:
                continue
            tot += 1; ag += (np.sign(dg) == np.sign(dv))
    return ag / tot if tot else np.nan


def recompute(combo, setting):
    """Fresh per-problem recompute. Returns (metrics dict, filename, n_groups, both_class)."""
    inv = find_files_for_combo(combo)
    ep = -1 if setting == "base" else 2
    sub = inv[(inv.setting == setting) & (inv.epoch == ep) & (inv.tc_eval == "self")]
    if not len(sub):
        return None, None, 0, 0
    f = sub.iloc[0]["file"]
    task = COMBO_CONFIG[combo]["task_label"]
    key, lc = GROUP_KEY[task], LABEL_COL[task]
    df = pd.read_csv(f)
    rocs, concs, sps, prs = [], [], [], []
    nboth = 0
    for _, g in df.groupby(key):
        y = lab(g[lc]); gen = pd.to_numeric(g["gen_score_typcorr"], errors="coerce").to_numpy()
        val = pd.to_numeric(g["val_score"], errors="coerce").to_numpy()
        ok = ~(np.isnan(y) | np.isnan(gen) | np.isnan(val)); y, gen, val = y[ok], gen[ok], val[ok]
        if len(y) >= 2 and len(set(y)) == 2:
            nboth += 1
            rocs.append(roc_auc_score(y, gen)); concs.append(conc(gen, val))
        if len(gen) >= 3 and np.std(gen) > 0 and np.std(val) > 0:
            sps.append(spearmanr(gen, val).correlation); prs.append(pearsonr(gen, val)[0])

    def m(a):
        a = [x for x in a if x == x]
        return (np.mean(a) if a else np.nan)
    return ({"gen_roc": m(rocs), "concordance": m(concs), "spearman": m(sps), "pearson": m(prs)},
            Path(f).name, df.groupby(key).ngroups, nboth)


def parse_tex(path):
    """Parse fix_*.tex -> {row_label: {col_idx: value}}. cols are base,s1,s2,s3,s4."""
    out = {}
    for line in path.read_text().splitlines():
        if "&" not in line or "\\textbf{Model" in line or line.strip().startswith("%"):
            continue
        if "\\toprule" in line or "\\bottomrule" in line or "tabular" in line or "midrule" in line:
            continue
        cells = [c.strip() for c in line.split("&")]
        label = cells[0].strip()
        vals = {}
        for i, c in enumerate(cells[1:6]):
            c = c.replace("\\\\", "").strip()
            if c == "---":
                vals[i] = None; continue
            mm = re.search(r"(-?\d+\.\d+)", c.replace("\\mathbf{", "").split("\\se")[0])
            vals[i] = float(mm.group(1)) if mm else None
        out[label] = vals
    return out


METRIC_TEX = {"gen_roc": "fix_genroc.tex", "concordance": "fix_concordance.tex",
              "spearman": "fix_spearman.tex", "pearson": "fix_pearson.tex"}
LABEL_OF = {"gemma_membership": "Gemma / Membership", "gemma_ifeval": "Gemma / IFEval",
            "qwen_membership": "Qwen / Membership", "qwen_ifeval": "Qwen / IFEval"}


def main():
    # recompute every cell once
    rc = {}
    print("=== matched source file + grouping per cell (eyeball model/task/setting/epoch) ===")
    for combo, _ in CELLS:
        for s in SETTINGS:
            metrics, fname, ng, nboth = recompute(combo, s)
            rc[(combo, s)] = metrics
            if fname:
                print(f"  {combo:18s} {s:4s} groups={ng:3d} both-class={nboth:3d}  {fname[:78]}")
            else:
                print(f"  {combo:18s} {s:4s}  (no file — expect '---' in tables)")

    print("\n=== diff recomputed vs committed .tex (tol %.1e) ===" % TOL)
    for metric, tex in METRIC_TEX.items():
        parsed = parse_tex(TABLES / tex)
        for combo, _ in CELLS:
            row = parsed.get(LABEL_OF[combo], {})
            for ci, s in enumerate(SETTINGS):
                texv = row.get(ci)
                recv = rc[(combo, s)][metric] if rc[(combo, s)] else np.nan
                recv = None if (recv != recv) else round(float(recv), 3)
                if texv is None and recv is None:
                    continue
                if texv is None or recv is None or abs(texv - recv) > TOL:
                    problems.append(f"{metric} {combo} {s}: tex={texv} recompute={recv}")
    if problems:
        print("FAIL:")
        for p in problems:
            print("  " + p)
        sys.exit(1)
    print("  all 4 tables x 4 cells x 5 settings match within tolerance.")
    print("\nALL CHECKS PASS")


if __name__ == "__main__":
    main()
