#!/usr/bin/env python3
"""Re-verify qwen RankAlign ifeval gen ROC across ALL score columns + both modes, showing the
exact file used. gemma + FLORA-PMI for contrast. Run on mll, qwen35 venv."""
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_predictor_comprehensive as B

COLS = ["gen_score", "gen_score_typcorr", "gen_score_lenorm", "gen_score_typcorr_lenorm"]


def per_prob(files, col, lab="correct", key="prompt"):
    vals = []
    for f in files:
        df = pd.read_csv(f)
        if col not in df.columns or lab not in df.columns or key not in df.columns:
            return np.nan, 0
        y = df[lab].astype(str).str.strip().str.lower().map({"yes": 1, "true": 1, "1": 1, "no": 0, "false": 0, "0": 0})
        for _, g in df.groupby(key):
            yy = y.loc[g.index].to_numpy()
            xx = pd.to_numeric(g[col], errors="coerce").to_numpy()
            ok = ~(np.isnan(yy) | np.isnan(xx)); yy, xx = yy[ok], xx[ok]
            if len(set(yy)) == 2 and len(yy) >= 2:
                vals.append(roc_auc_score(yy, xx))
    return (np.mean(vals) * 100 if vals else np.nan), len(vals)


def main():
    idx, _ = B.discover()
    for mdl in ["qwen-3.5-9b", "gemma-2-9b-it"]:
        for lab, s in [("Base", "base"), ("RankAlign", "s2"), ("FLORA-PMI", "s4")]:
            for mode, pref in [("self", ["basetyp", "self"]), ("neg", ["basetypneg", "neg"])]:
                ep = "base" if s == "base" else "ep2"
                files = B.cell_files(idx, mdl, "ifeval", "train", s, ep, pref)
                if not files:
                    continue
                print(f"\n{mdl} / {lab} / {mode}-mode  ({len(files)} file(s))")
                print(f"   file: {os.path.basename(files[0])[:95]}")
                for c in COLS:
                    roc, n = per_prob(files, c)
                    print(f"     {c:26s} per-problem gen ROC = {roc:5.1f}  (n={n})")
                if s == "base":
                    break  # base mode doesn't matter much


if __name__ == "__main__":
    main()
