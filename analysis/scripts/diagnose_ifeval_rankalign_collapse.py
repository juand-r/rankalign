#!/usr/bin/env python3
"""Why does RankAlign collapse to chance on gemma-2-9b-it / ifeval but survive on qwen / ifeval?

Hypothesis: the NLL-explosion -- without an NLL anchor, RankAlign's preference loss drives the
generator log-prob scores to explode on gemma, degenerating the model; on qwen it's milder.
Check generator-score spread (std, range) for RankAlign vs FLORA-PMI vs Base on ifeval train ep2.

Run on mll, qwen35 venv.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_predictor_comprehensive as B

SELF = ["basetyp", "self"]


def stats(idx, mdl, s):
    f = B.cell_files(idx, mdl, "ifeval", "train", s, "base" if s == "base" else "ep2", SELF + ["basetypneg", "neg"])
    if not f:
        return None
    df = pd.concat([pd.read_csv(p) for p in f], ignore_index=True)
    g = pd.to_numeric(df["gen_score"], errors="coerce").dropna()
    gt = pd.to_numeric(df["gen_score_typcorr"], errors="coerce").dropna()
    v = pd.to_numeric(df["val_score"], errors="coerce").dropna()
    return dict(n=len(df), gen_std=g.std(), gen_min=g.min(), gen_max=g.max(),
                gtc_std=gt.std(), gtc_absmax=gt.abs().max(), val_std=v.std(), val_absmax=v.abs().max())


def main():
    idx, _ = B.discover()
    hdr = ("model / method", "gen_std", "gen_min", "gen_max", "gtc_std", "gtc_|max|", "val_std", "val_|max|")
    print("%-28s %9s %9s %9s %9s %10s %8s %9s" % hdr)
    for mdl in ["gemma-2-9b-it", "qwen-3.5-9b"]:
        for lab, s in [("Base", "base"), ("RankAlign s2", "s2"), ("FLORA-PMI s4", "s4")]:
            st = stats(idx, mdl, s)
            if st:
                print("%-28s %9.1f %9.1f %9.1f %9.1f %10.1f %8.2f %9.1f" % (
                    f"{mdl} / {lab}", st["gen_std"], st["gen_min"], st["gen_max"],
                    st["gtc_std"], st["gtc_absmax"], st["val_std"], st["val_absmax"]))
        print()


if __name__ == "__main__":
    main()
