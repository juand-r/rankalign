#!/usr/bin/env python3
"""Investigation (3, data-supported part): the length / lenorm mechanism on
humaneval-v2.1correct-upper, Base vs RankAlign vs RankAlign+TC.

The per-token-PMI lens (v1) needs per-token logprobs we don't have for
v2.1correct-upper. This does the part the aggregate scores CSVs DO support
(the other half of the v1 'why lenorm helps' report):

Per group, within correct / within wrong:
  - mean length N (num_tokens)
  - r(N, gen_score)            : the 'longer => less-negative sum' coupling
                                 that drives lenorm flips (v1 finding)
  - sep(raw), sep(lenorm), sep(tc)  : correct/wrong separation by each score
Goal: explain why lenorm is RankAlign's *best* variant but TC is
RankAlign+TC's best.

Usage:
  python scripts-more/diag_length_effect.py \
    --scores-dir /datastor1/jdr/gv-gap/rankalign/outputs_gemma4_3epoch_e2 \
    --base-dir   /datastor1/jdr/gv-gap/rankalign/outputs_gemma4_3epoch_e2/_base_model_eval
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

GROUPS = {
    "Base_self":      ("base", lambda n: n.startswith("scores_self-v6-google_gemma-4-31B-it")),
    "Base_neg":       ("base", lambda n: n.startswith("scores_neg-v6-google_gemma-4-31B-it")),
    "RankAlign_self": ("trained", lambda n: n.startswith("scores_basetyp-") and "tc-self" not in n),
    "RankAlign_neg":  ("trained", lambda n: n.startswith("scores_basetypneg-") and "tc-neg" not in n),
    "RankAlign+tc_self": ("trained", lambda n: n.startswith("scores_basetyp-") and "tc-self" in n),
    "RankAlign+negtc_neg": ("trained", lambda n: n.startswith("scores_basetypneg-") and "tc-neg" in n),
}


def sep(score, y):
    c, w = score[y == 1], score[y == 0]
    if len(c) < 2 or len(w) < 2:
        return np.nan
    psd = np.sqrt((c.var(ddof=1) + w.var(ddof=1)) / 2)
    return np.nan if psd == 0 else (c.mean() - w.mean()) / psd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", required=True)
    ap.add_argument("--base-dir", required=True)
    a = ap.parse_args()

    hdr = (f"{'group':22s} {'Ncorr':>6s} {'Nwrong':>6s} "
           f"{'r(N,g)|c':>9s} {'r(N,g)|w':>9s} "
           f"{'sep raw':>8s} {'sep len':>8s} {'sep tc':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for label, (kind, filt) in GROUPS.items():
        d = a.base_dir if kind == "base" else a.scores_dir
        files = [f for f in glob.glob(os.path.join(d, "scores_*.csv"))
                 if (kind == "base" or "epoch2" in os.path.basename(f))
                 and filt(os.path.basename(f))]
        frames = [pd.read_csv(f) for f in files if os.path.getsize(f) > 0]
        if not frames:
            print(f"{label:22s}  (no files)")
            continue
        df = pd.concat(frames, ignore_index=True)
        y = df["correct"].astype(str).str.strip().str.lower().map(
            {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0})
        m = (y.notna() & df["gen_score"].notna() & df["num_tokens"].notna()
             & df["gen_score_lenorm"].notna() & df["gen_score_typcorr"].notna())
        y = y[m].to_numpy(int)
        N = df.loc[m, "num_tokens"].to_numpy(float)
        g = df.loc[m, "gen_score"].to_numpy(float)
        ln = df.loc[m, "gen_score_lenorm"].to_numpy(float)
        tc = df.loc[m, "gen_score_typcorr"].to_numpy(float)
        cI, wI = y == 1, y == 0

        def rr(mask):
            if mask.sum() < 3:
                return np.nan
            return np.corrcoef(N[mask], g[mask])[0, 1]

        print(f"{label:22s} {N[cI].mean():6.1f} {N[wI].mean():6.1f} "
              f"{rr(cI):9.3f} {rr(wI):9.3f} "
              f"{sep(g, y):8.3f} {sep(ln, y):8.3f} {sep(tc, y):8.3f}")

    print("\nReading guide:")
    print("  Ncorr vs Nwrong: are correct solutions systematically longer/shorter?")
    print("  r(N,g): v1 found longer responses have less-negative SUM logp;")
    print("    strong r means raw sum is length-confounded -> lenorm helps by")
    print("    removing it. Compare across groups.")
    print("  Best sep among raw/len/tc should track the gen_roc tables'")
    print("    best variant per group.")


if __name__ == "__main__":
    main()
