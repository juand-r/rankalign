#!/usr/bin/env python3
"""HumanEval §2.4/2.7/2.9 diagnostics on the TRAIN set (runs ON mll), mirroring the original
report (which computed concordance/scatter/delta on the train-set dynamics scores, epoch 2).

Reads the per-problem TRAIN score CSVs in outputs-he-trainset/ (the --train split, N=50
STRATIFIED candidates/problem -- the same subsample used for GROUP B / trs2, because the full
train pool is ~2283/problem). For each (model, dataset, setting) at EPOCH 2 (base = untrained),
in its natural eval mode (gen_score_typcorr = tc gen, val_score, correct), computes:
  - Concordance: per problem, fraction of candidate pairs (i<j) with
    sign(gen_i-gen_j)==sign(val_i-val_j); mean +/- SE over problems -> he_concordance_train.csv
  - Scatter: gen(tc) vs val, correct(green)/incorrect(red), gemma-4 multi, Base/s2/s4 -> PNG
  - Score-delta histograms: per-problem (tc-gen pos mean - neg mean), per model x ds -> PNG

Currently available on train: gemma-4 base/s2/s4 (GROUP B) + whatever trs2 has finished
(gemma s3/s7, qwen). Settings with no train scores yet are simply skipped.
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/datastor2/jdr/rankalign")
TRAIN_DIR = REPO / "outputs-he-trainset"
PLOTS = REPO / "analysis/plots"; TABLES = REPO / "analysis/tables"
PLOTS.mkdir(parents=True, exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True)

NAT_PREFIX = {"base": "self", "s1": "basetyp", "s2": "basetypneg", "s3": "basetypneg",
              "s4": "basetyp", "s7": "basetypneg", "s13": "basetyp"}
SET_LABEL = {"base": "Base", "s1": "SFT", "s2": "RankAlign", "s3": "New+fsx",
             "s4": "FLORA-PMI", "s7": "FLORA-Neg", "s13": "Consistency FT"}


def parse(fname):
    m = re.match(r"scores_(basetypneg|basetyp|self|neg)-", fname); pref = m.group(1) if m else "?"
    is_base = ("v6-" in fname) and ("-v7-" not in fname)
    mk = "qwen" if ("Qwen3.5-9B" in fname or "Qwen--Qwen3" in fname or "Qwen_Qwen3" in fname) else ("gemma" if "gemma-4-31" in fname.lower() else "?")
    ds = "upper" if "correct-upper" in fname else ("multi" if "correct-multi" in fname else "?")
    if is_base: s = "base"
    elif ("tcs" in fname) or ("tc-self" in fname): s = "s4"
    elif ("tcn" in fname) or ("tc-neg" in fname): s = "s7"
    elif "cft" in fname: s = "s13"
    elif ("fsx" in fname) or ("force-same-x" in fname): s = "s3"
    elif ("lo0.1" in fname) or ("labelonly" in fname) or re.search(r"-p0-", fname) or ("pref0" in fname): s = "s1"
    else: s = "s2"
    if is_base: epoch = "base"
    elif re.search(r"epoch2|-e2-", fname): epoch = "ep2"
    elif re.search(r"epoch1|-e1-", fname): epoch = "ep1"
    elif re.search(r"epoch0|-e0-", fname): epoch = "ep0"
    else: epoch = "?"
    return mk, ds, s, pref, epoch


def collect():
    """dict[(mk,ds,setting)] -> [paths] for EPOCH 2 (or base), natural mode, TRAIN split."""
    out = {}
    for p in TRAIN_DIR.glob("scores_*_train_*.csv"):
        if "humaneval" not in p.name:
            continue
        mk, ds, s, pref, epoch = parse(p.name)
        if mk == "?" or ds == "?" or s == "?":
            continue
        if pref != NAT_PREFIX[s]:
            continue
        if s != "base" and epoch != "ep2":   # trained settings: epoch-2 final model only
            continue
        out.setdefault((mk, ds, s), []).append(p)
    return out


def concordance_one(df):
    g = pd.to_numeric(df.get("gen_score_typcorr"), errors="coerce").to_numpy()
    v = pd.to_numeric(df.get("val_score"), errors="coerce").to_numpy()
    ok = ~(np.isnan(g) | np.isnan(v)); g, v = g[ok], v[ok]
    n = len(g)
    if n < 2:
        return np.nan
    agree = tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            dg, dv = g[i] - g[j], v[i] - v[j]
            if dg == 0 or dv == 0:
                continue
            tot += 1; agree += (np.sign(dg) == np.sign(dv))
    return agree / tot if tot else np.nan


def _pos_mask(df):
    return df["correct"].astype(str).str.lower().isin(["yes", "true", "1"])


def main():
    files = collect()
    print(f"TRAIN groups found (epoch2/base, natural mode): {len(files)}")
    for k in sorted(files):
        print("  ", k, len(files[k]), "problems")

    # ---- concordance table ----
    rows = []
    for (mk, ds, s), paths in sorted(files.items()):
        vals = [concordance_one(pd.read_csv(p)) for p in paths]
        vals = [x for x in vals if not np.isnan(x)]
        if not vals:
            continue
        a = np.array(vals)
        rows.append(dict(model=mk, dataset=ds, setting=s, n=len(a),
                         concordance=a.mean(), se=a.std(ddof=1) / np.sqrt(len(a))))
    pd.DataFrame(rows).to_csv(TABLES / "he_concordance_train.csv", index=False)
    print("wrote he_concordance_train.csv\n", pd.DataFrame(rows).to_string())

    # ---- scatter: gemma multi, Base/s2/s4 (whatever present) ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, s in zip(axes, ["base", "s2", "s4"]):
        paths = files.get(("gemma", "multi", s), [])
        if not paths:
            ax.set_title(f"{SET_LABEL[s]} (no train data)"); continue
        df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        g = pd.to_numeric(df["gen_score_typcorr"], errors="coerce")
        v = pd.to_numeric(df["val_score"], errors="coerce")
        pos = _pos_mask(df)
        ax.scatter(g[pos], v[pos], s=10, alpha=0.3, c="green", label="correct")
        ax.scatter(g[~pos], v[~pos], s=10, alpha=0.3, c="red", label="incorrect")
        ax.set_xlabel("gen score (tc)"); ax.set_ylabel("val score"); ax.set_title(SET_LABEL[s])
        ax.legend(fontsize=8)
    plt.suptitle("gemma-4-31b / multi — gen(tc) vs val (TRAIN, N=50 subsample, epoch 2)")
    plt.tight_layout(); plt.savefig(PLOTS / "he_gen_vs_val_scatter_gemma_multi_train.png", dpi=150, bbox_inches="tight"); plt.close()
    print("wrote he_gen_vs_val_scatter_gemma_multi_train.png")

    # ---- delta histograms: per (model,ds), s2/s4/s7, per-problem (pos mean - neg mean) tc gen ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9)); axes = axes.flatten()
    for ax, (mk, ds) in zip(axes, [("gemma", "upper"), ("gemma", "multi"), ("qwen", "upper"), ("qwen", "multi")]):
        any_data = False
        for s in ["s2", "s4", "s7"]:
            deltas = []
            for p in files.get((mk, ds, s), []):
                df = pd.read_csv(p); g = pd.to_numeric(df["gen_score_typcorr"], errors="coerce"); pos = _pos_mask(df)
                if pos.sum() and (~pos).sum():
                    deltas.append(g[pos].mean() - g[~pos].mean())
            if deltas:
                ax.hist(deltas, bins=20, alpha=0.5, label=SET_LABEL[s]); any_data = True
        ax.axvline(0, color="k", lw=0.6); ax.set_title(f"{mk} / {ds}" + ("" if any_data else " (no train data yet)"))
        ax.set_xlabel("per-problem (tc-gen pos mean $-$ neg mean)")
        if any_data:
            ax.legend(fontsize=8)
    plt.suptitle("Score-delta distributions (TRAIN, N=50, epoch 2) — positives should separate above 0")
    plt.tight_layout(); plt.savefig(PLOTS / "he_score_delta_hist_train.png", dpi=150, bbox_inches="tight"); plt.close()
    print("wrote he_score_delta_hist_train.png\ndone.")


if __name__ == "__main__":
    main()
