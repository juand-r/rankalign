#!/usr/bin/env python3
"""Why does FLORA win concordance but lose ROC to RankAlign (HumanEval)?

Concordance = generator<->validator agreement. ROC = generator<->correctness.
Hypothesis: FLORA (TC) ties the generator to the VALIDATOR (high concordance) and so inherits the
validator's correctness ceiling; RankAlign pushes the generator past the validator on correctness
(high gen-ROC) at the cost of gen<->val agreement.

Test: compute per-problem (TRAIN, ep2, natural mode) for each method:
  gen_ROC (correct vs incorrect by gen_score_typcorr)
  val_ROC (correct vs incorrect by val_score)
  concordance (gen vs val pairwise sign agreement)
  gen-val Spearman
Averaged over problems (mean +- SE). If FLORA: gen_ROC ~ val_ROC, high concordance;
RankAlign: gen_ROC >> val_ROC, lower concordance -> hypothesis confirmed.

Run on mll, qwen35 venv:  python analysis/scripts/investigate_concordance_vs_roc.py
"""
import re, glob, os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

REPO = "/datastor2/jdr/rankalign"
TRAIN_DIR = f"{REPO}/outputs-he-trainset-perproblem"
NAT = {"base": "self/no-base", "s1": "self/base-typ", "s2": "neg/base-typ", "s3": "neg/base-typ",
       "s4": "self/base-typ", "s7": "neg/base-typ", "s13": "self/base-typ"}
LABEL = {"s2": "RankAlign", "s3": "New+fsx", "s4": "FLORA-PMI", "s7": "FLORA-Neg"}


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


def concordance(g, v):
    dg = np.subtract.outer(g, g); dv = np.subtract.outer(v, v)
    iu = np.triu_indices(len(g), k=1); dg, dv = dg[iu], dv[iu]
    m = (dg != 0) & (dv != 0)
    return float(np.mean(np.sign(dg[m]) == np.sign(dv[m]))) if m.any() else np.nan


def collect():
    out = {}
    for p in glob.glob(f"{TRAIN_DIR}/scores_*train-humaneval_*.csv"):
        mk, ds, s, ep, mode = parse(os.path.basename(p))
        if "?" in (mk, ds, s, ep):
            continue
        out.setdefault((mk, ds, s, ep, mode), []).append(p)
    return out


def main():
    TR = collect()
    print(f"{'model':10} {'data':6} {'setting':10} {'gen_ROC':>8} {'val_ROC':>8} {'gen-val':>8} {'concord':>8}  n")
    print("-" * 70)
    for mk in ["gemma", "qwen"]:
        for ds in ["upper", "multi"]:
            for s in ["s2", "s3", "s4", "s7"]:
                paths = TR.get((mk, ds, s, "ep2", NAT[s]), [])
                groc, vroc, gv, conc = [], [], [], []
                for p in paths:
                    df = pd.read_csv(p)
                    y = df["correct"].astype(str).str.lower().isin(["yes", "true", "1"]).to_numpy().astype(int)
                    g = pd.to_numeric(df.get("gen_score_typcorr"), errors="coerce").to_numpy()
                    v = pd.to_numeric(df.get("val_score"), errors="coerce").to_numpy()
                    ok = ~(np.isnan(g) | np.isnan(v)); y, g, v = y[ok], g[ok], v[ok]
                    if len(set(y)) == 2 and len(y) >= 2:
                        groc.append(roc_auc_score(y, g)); vroc.append(roc_auc_score(y, v))
                        conc.append(concordance(g, v))
                    if len(g) >= 3 and np.std(g) > 0 and np.std(v) > 0:
                        gv.append(spearmanr(g, v).correlation)

                def ms(a):
                    a = [x for x in a if x == x]
                    return (np.mean(a) * 100, np.std(a, ddof=1) / np.sqrt(len(a)) * 100) if a else (np.nan, 0)
                gr, _ = ms(groc); vr, _ = ms(vroc); gvm, _ = ms([x / 100 for x in gv]) if gv else (np.nan, 0); cc, _ = ms(conc)
                gvm = np.mean(gv) if gv else np.nan
                print(f"{mk:10} {ds:6} {LABEL[s]:10} {gr:8.1f} {vr:8.1f} {gvm:8.2f} {cc:8.1f}  {len(groc)}")
        print()


if __name__ == "__main__":
    main()
