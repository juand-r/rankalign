#!/usr/bin/env python3
"""Does BASE validator ROC predict the FLORA-vs-RankAlign winner (by gen ROC)?

Non-circular test: the predictor is the BASE (untrained) validator's correctness ROC, which uses
NO generator (val_score is the discriminator log-odds, independent of the typicality mode). The
outcome is which method wins gen ROC (FLORA's best of s4/s7 vs RankAlign s2).

Cells (8): HumanEval gemma-4-31b / qwen-3.5-9b x {upper, multi}  +  original-report
gemma-2-9b-it / qwen-3.5-9b x {ifeval, hyponymy(=membership)}. TRAIN, epoch 2.

HE numbers are recomputed per-problem from outputs-he-trainset-perproblem/; ifeval+hyponymy come
from the committed analysis/tables/report_fix_metrics.csv. Writes:
  analysis/tables/predictor_base_valroc.csv / .tex
  analysis/plots/predictor_base_valroc.png   (x=base val_ROC, y=FLORA-RankAlign gen margin)
Run on mll, qwen35 venv.
"""
import re, glob, os, csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

REPO = "/datastor2/jdr/rankalign"
TRAIN_DIR = f"{REPO}/outputs-he-trainset-perproblem"
TABLES = f"{REPO}/analysis/tables"; PLOTS = f"{REPO}/analysis/plots"
NAT = {"base": "self/no-base", "s2": "neg/base-typ", "s3": "neg/base-typ",
       "s4": "self/base-typ", "s7": "neg/base-typ"}  # HE natural eval modes


def parse(fname):
    m = re.match(r"scores_(basetypneg|basetyp|self|neg)-", fname); pref = m.group(1) if m else "?"
    is_base = ("v6-" in fname) and ("-v7-" not in fname)
    mk = "qwen" if ("Qwen3.5-9B" in fname or "Qwen--Qwen3" in fname or "Qwen_Qwen3" in fname) else ("gemma" if "gemma-4-31" in fname.lower() else "?")
    ds = "upper" if "correct-upper" in fname else ("multi" if "correct-multi" in fname else "?")
    if is_base: s = "base"
    elif ("tc-self" in fname) or ("-tcs-" in fname): s = "s4"
    elif ("tc-neg" in fname) or ("-tcn-" in fname): s = "s7"
    elif "cft" in fname: s = "s13"
    elif ("force-same-x" in fname) or ("-fsx-" in fname): s = "s3"
    elif ("labelonly" in fname) or re.search(r"-p0-", fname) or ("pref0" in fname): s = "s1"
    else: s = "s2"
    ep = "base" if is_base else ("ep2" if re.search(r"epoch2|-e2-", fname) else ("ep1" if re.search(r"epoch1|-e1-", fname) else ("ep0" if re.search(r"epoch0|-e0-", fname) else "?")))
    typ = "neg" if pref in ("neg", "basetypneg") else "self"
    bt = "base-typ" if pref in ("basetyp", "basetypneg") else "no-base"
    return mk, ds, s, ep, f"{typ}/{bt}"


def he_collect():
    out = {}
    for p in glob.glob(f"{TRAIN_DIR}/scores_*train-humaneval_*.csv"):
        mk, ds, s, ep, mode = parse(os.path.basename(p))
        if "?" in (mk, ds, s, ep):
            continue
        out.setdefault((mk, ds, s, ep, mode), []).append(p)
    return out


def per_problem_roc(paths, col):
    vals = []
    for p in paths:
        df = pd.read_csv(p)
        y = df["correct"].astype(str).str.lower().isin(["yes", "true", "1"]).to_numpy().astype(int)
        x = pd.to_numeric(df.get(col), errors="coerce").to_numpy()
        ok = ~np.isnan(x); y, x = y[ok], x[ok]
        if len(set(y)) == 2 and len(y) >= 2:
            vals.append(roc_auc_score(y, x))
    return float(np.mean(vals) * 100) if vals else np.nan


def he_cells(TR):
    rows = []
    for mk in ["gemma", "qwen"]:
        for ds in ["upper", "multi"]:
            # base val_ROC: mode-independent -> any base file works
            base_paths = TR.get((mk, ds, "base", "base", NAT["base"]), [])
            base_val = per_problem_roc(base_paths, "val_score")
            def gen(s):
                return per_problem_roc(TR.get((mk, ds, s, "ep2", NAT[s]), []), "gen_score_typcorr")
            ra = gen("s2"); flora = np.nanmax([gen("s4"), gen("s7")])
            rows.append(dict(family="HumanEval", model=f"{mk}-{'4-31b' if mk=='gemma' else '3.5-9b'}",
                             task=ds, base_val_roc=base_val, rankalign_gen=ra, flora_gen=flora))
    return rows


def orig_cells():
    df = pd.read_csv(f"{TABLES}/report_fix_metrics.csv")
    def row(combo, setting, tc, col, ep="ep2"):
        e = "base" if setting == "base" else ep
        m = df[(df.combo == combo) & (df.setting == setting) & (df.epoch == e) & (df.tc_eval == tc)]
        return float(m.iloc[0][col]) * 100 if len(m) else np.nan
    cells = [("gemma_ifeval", "gemma-2-9b-it", "ifeval"), ("qwen_ifeval", "qwen-3.5-9b", "ifeval"),
             ("gemma_membership", "gemma-2-9b-it", "hyponymy"), ("qwen_membership", "qwen-3.5-9b", "hyponymy")]
    rows = []
    for combo, model, task in cells:
        base_val = row(combo, "base", "self", "val_roc_mean")
        ra = row(combo, "s2", "self", "gen_roc_mean")        # original self-scoring convention
        flora = np.nanmax([row(combo, "s4", "self", "gen_roc_mean"), row(combo, "s7", "neg", "gen_roc_mean")])
        rows.append(dict(family="original", model=model, task=task,
                         base_val_roc=base_val, rankalign_gen=ra, flora_gen=flora))
    return rows


def main():
    rows = he_cells(he_collect()) + orig_cells()
    for r in rows:
        r["margin"] = r["flora_gen"] - r["rankalign_gen"]
        r["winner"] = "FLORA" if r["margin"] > 0 else "RankAlign"
    rows.sort(key=lambda r: r["base_val_roc"])
    # CSV
    with open(f"{TABLES}/predictor_base_valroc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "model", "task", "base_val_roc", "rankalign_gen", "flora_gen", "margin", "winner"])
        w.writeheader()
        w.writerows(rows)
    # console + tex
    print(f"{'base_valROC':>11} {'model':13} {'task':10} {'RankAlign':>9} {'FLORA':>7} {'winner':>10}")
    tex = ["% auto: build_predictor_table.py", "\\begin{tabular}{rllrrl}", "\\toprule",
           "Base val ROC & Model & Task & RankAlign & FLORA & Winner \\\\", "\\midrule"]
    for r in rows:
        print(f"{r['base_val_roc']:11.1f} {r['model']:13} {r['task']:10} {r['rankalign_gen']:9.1f} {r['flora_gen']:7.1f} {r['winner']:>10}")
        tex.append(f"{r['base_val_roc']:.1f} & {r['model']} & {r['task']} & {r['rankalign_gen']:.1f} & {r['flora_gen']:.1f} & {r['winner']} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    open(f"{TABLES}/predictor_base_valroc.tex", "w").write("\n".join(tex) + "\n")
    # scatter
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for r in rows:
        c = "tab:green" if r["winner"] == "FLORA" else "tab:red"
        mk = "o" if r["family"] == "HumanEval" else "s"
        ax.scatter(r["base_val_roc"], r["margin"], c=c, marker=mk, s=90, edgecolor="k", zorder=3)
        ax.annotate(f"{r['model'][:5]}/{r['task'][:4]}", (r["base_val_roc"], r["margin"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.axhline(0, color="gray", lw=1)
    ax.axvline(87, color="navy", ls="--", lw=1, label="approx. threshold ~87")
    ax.set_xlabel("BASE validator ROC ($\\times$100)  [predictor; no generator]")
    ax.set_ylabel("gen-ROC margin: FLORA(best) $-$ RankAlign")
    ax.set_title("Does base validator ROC predict the winner?\n(green=FLORA wins, red=RankAlign; circle=HumanEval, square=original)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{PLOTS}/predictor_base_valroc.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("\nwrote predictor_base_valroc.{csv,tex,png}")


if __name__ == "__main__":
    main()
