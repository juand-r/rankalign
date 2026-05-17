#!/usr/bin/env python3
"""Investigation (2): why does eval-time TC help the Base model but HURT the
RankAlign-trained baseline on humaneval-v2.1correct-upper?

For each group (Base vs RankAlign, self & neg side) pool all per-task rows and
look at:
  - gen_score            : raw  log P(y|x)
  - gen_score_typcorr    : TC-corrected score
  - offset = gen - tc    : the typicality term subtracted (~ log P_base(y))
and how each separates correct vs wrong (point-biserial style: mean for
correct minus mean for wrong, in std units). If TC helps, the corrected
score should separate correct/wrong *better* than raw; if it hurts, worse.
Also reports how the offset itself correlates with correctness — an offset
that is large but unaligned with correctness is the over-correction signature.

Usage:
  python scripts-more/diag_tc_offset.py \
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
    # label -> (dir kind, filename filter)
    "Base_self":      ("base", lambda n: n.startswith("scores_self-v6-google_gemma-4-31B-it")),
    "Base_neg":       ("base", lambda n: n.startswith("scores_neg-v6-google_gemma-4-31B-it")),
    "RankAlign_self": ("trained", lambda n: n.startswith("scores_basetyp-") and "tc-self" not in n),
    "RankAlign_neg":  ("trained", lambda n: n.startswith("scores_basetypneg-") and "tc-neg" not in n),
}


def sep(score: np.ndarray, y: np.ndarray) -> float:
    """Standardized mean difference: (mean_correct - mean_wrong)/pooled_std.
    Higher = better correct/wrong separation by this score."""
    c, w = score[y == 1], score[y == 0]
    if len(c) < 2 or len(w) < 2:
        return np.nan
    psd = np.sqrt((c.var(ddof=1) + w.var(ddof=1)) / 2)
    return np.nan if psd == 0 else (c.mean() - w.mean()) / psd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", required=True)
    ap.add_argument("--base-dir", required=True)
    a = ap.parse_args()

    print(f"{'group':16s} {'n':>6s} {'sep(raw)':>9s} {'sep(tc)':>9s} "
          f"{'Δsep':>7s} {'|offset|':>9s} {'sep(offset)':>11s}")
    print("-" * 74)
    for label, (kind, filt) in GROUPS.items():
        d = a.base_dir if kind == "base" else a.scores_dir
        files = [f for f in glob.glob(os.path.join(d, "scores_*.csv"))
                 if "epoch2" in os.path.basename(f) or kind == "base"]
        files = [f for f in files if filt(os.path.basename(f))]
        frames = []
        for f in files:
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            if "correct" not in df or "gen_score" not in df:
                continue
            frames.append(df)
        if not frames:
            print(f"{label:16s}  (no files)")
            continue
        df = pd.concat(frames, ignore_index=True)
        y = df["correct"].astype(str).str.strip().str.lower().map(
            {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0})
        m = y.notna() & df["gen_score"].notna() & df["gen_score_typcorr"].notna()
        y = y[m].to_numpy(dtype=int)
        g = df.loc[m, "gen_score"].to_numpy(float)
        tc = df.loc[m, "gen_score_typcorr"].to_numpy(float)
        off = g - tc  # ~ the typicality term subtracted
        print(f"{label:16s} {len(y):6d} {sep(g, y):9.3f} {sep(tc, y):9.3f} "
              f"{sep(tc, y) - sep(g, y):7.3f} {np.abs(off).mean():9.2f} "
              f"{sep(off, y):11.3f}")

    print("\nReading guide:")
    print("  sep(raw)>0, sep(tc)>0  : score ranks correct above wrong (good)")
    print("  Δsep>0  : TC improves separation (helps).  Δsep<0 : TC hurts.")
    print("  |offset|: magnitude of the typicality term subtracted.")
    print("  sep(offset): if ~0 while |offset| is large => TC injects a big")
    print("    correctness-unaligned shift => over-correction signature.")


if __name__ == "__main__":
    main()
