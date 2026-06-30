#!/usr/bin/env python3
"""gen ROC, val ROC, val Acc, and gen-val Pearson for specific qwen IFEval (OOD) paper cells.

Metric definitions match scripts/summarize_scores_file.py exactly, computed per OOD prompt file
(each ifeval-prompt_N_test CSV is one problem), then averaged over the 20 prompts (mean ± SE,
SE = std(ddof=1)/sqrt(n)):
  gen_roc = ROC(correct, gen_score_typcorr)
  val_roc = ROC(correct, val_score)
  val_acc = accuracy(correct, val_score > 0)      # validator log-odds threshold 0
  pearson = pearsonr(gen_score_typcorr, val_score) # generator-validator agreement

Target runs (identified by recipe tokens + version dir; gen_roc confirms the match):
  s2 RankAlign self : outputs-rerun-wandb-v3 (gen ROC ~75.4)
  s4 FLORA-PMI self : outputs-rerun-wandb    (v1, 20260608, gen ROC ~79.6)
  s7 FLORA-Neg  neg : outputs-rerun-wandb (v1) AND outputs-rerun-wandb-v3 (both ~66.0)

Run on mll, qwen35 venv (numpy/pandas/scipy/sklearn). Read-only.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, roc_auc_score

D2 = "/datastor2/jdr/rankalign"
OOD_MAX = 21
EXCL_S2 = re.compile(r"nv1|ng1|fsx|tcs|tcn|lo0|-p0-|tc-self|tc-neg|force-same-x|labelonly")


def files_for(dirname, prefix, include=None, exclude=None, date=None):
    out = []
    for p in glob.glob(f"{D2}/{dirname}/scores_{prefix}-*Qwen3.5-9B*ifeval-prompt_*_test*"):
        n = os.path.basename(p)
        m = re.search(r"ifeval-prompt_(\d+)_test", n)
        if not m or int(m.group(1)) > OOD_MAX:
            continue
        if include and not re.search(include, n):
            continue
        if exclude and exclude.search(n):
            continue
        if date and f"_{date}" not in n:
            continue
        out.append(p)
    return sorted(out)


def per_file(path):
    df = pd.read_csv(path)
    y = (df["correct"].astype(str).str.strip().str.lower()
         .map({"yes": 1, "true": 1, "1": 1, "no": 0, "false": 0, "0": 0}))
    gen = pd.to_numeric(df["gen_score_typcorr"], errors="coerce")
    val = pd.to_numeric(df["val_score"], errors="coerce")
    ok = ~(y.isna() | gen.isna() | val.isna())
    y, gen, val = y[ok].to_numpy(), gen[ok].to_numpy(), val[ok].to_numpy()
    if len(set(y.tolist())) != 2 or len(y) < 2:
        return None
    return dict(
        gen_roc=roc_auc_score(y, gen),
        val_roc=roc_auc_score(y, val),
        val_acc=accuracy_score(y, (val > 0.0).astype(int)),
        pearson=pearsonr(gen, val)[0],
    )


def summarize(label, files):
    dates = sorted({re.search(r"_(20[0-9]{6})\.csv", os.path.basename(f)).group(1) for f in files})
    deltas = sorted({(re.search(r"-d([0-9.]+)-e", os.path.basename(f)) or re.search(r"(x)", "x")).group(1) for f in files})
    rows = [r for f in files if (r := per_file(f)) is not None]
    print(f"\n=== {label} ===")
    print(f"    files={len(files)}  usable_prompts={len(rows)}  dates={dates}  deltas={deltas}")
    if not rows:
        return
    for k, scale in [("gen_roc", 100), ("val_roc", 100), ("val_acc", 100), ("pearson", 1)]:
        v = np.array([r[k] for r in rows]) * scale
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
        unit = "" if k == "pearson" else ""
        print(f"    {k:9s} = {v.mean():7.3f} ± {se:.3f}{unit}")


def main():
    summarize("s2 RankAlign  self  (outputs-rerun-wandb-v3)",
              files_for("outputs-rerun-wandb-v3", "self", exclude=EXCL_S2))
    summarize("s4 FLORA-PMI  self  (outputs-rerun-wandb, v1 0608)",
              files_for("outputs-rerun-wandb", "self", include=r"tcs|tc-self", date="20260608"))
    summarize("s7 FLORA-Neg  neg   (outputs-rerun-wandb, v1 0608)",
              files_for("outputs-rerun-wandb", "neg", include=r"tcn|tc-neg", date="20260608"))
    summarize("s7 FLORA-Neg  neg   (outputs-rerun-wandb-v3)",
              files_for("outputs-rerun-wandb-v3", "neg", include=r"tcn|tc-neg"))


if __name__ == "__main__":
    main()
