#!/usr/bin/env python3
"""HumanEval §2.4/2.7/2.9 diagnostics from RAW per-candidate score CSVs (runs ON mll).

For each (model, dataset, setting) in its natural eval mode, reads the per-problem TEST score
CSVs (gen_score_typcorr = tc gen, val_score, correct), and computes:
  - Concordance: per problem, fraction of candidate pairs (i<j) where
    sign(gen_i-gen_j) == sign(val_i-val_j); mean +/- SE over problems. -> concordance CSV.
  - Scatter: gen_score_typcorr vs val_score, positives(green)/negatives(red), gemma-4 multi,
    settings s2/s4/s7. -> PNG.
  - Score-delta histograms: per-problem (mean tc-gen pos - mean tc-gen neg), per model x ds,
    settings s2/s4/s7. -> PNG.

Test score dirs: gemma upper=outputs_gemma4_mll_tmp, multi=outputs_gemma4_mll_tmp-multi,
qwen=outputs-rerun-wandb (humaneval files only). Natural eval mode per setting selects the
filename prefix (basetyp / basetypneg / self).
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/datastor2/jdr/rankalign")
PLOTS = REPO / "analysis/plots"; TABLES = REPO / "analysis/tables"
PLOTS.mkdir(parents=True, exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True)
DIRS = [REPO / "outputs_gemma4_mll_tmp", REPO / "outputs_gemma4_mll_tmp-multi", REPO / "outputs-rerun-wandb"]

# natural eval mode -> filename prefix
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
    return mk, ds, s, pref


def collect():
    """dict[(mk,ds,setting)] -> list of per-problem DataFrames (natural mode only)."""
    out = {}
    for d in DIRS:
        if not d.is_dir():
            continue
        for p in d.glob("scores_*_test_*.csv"):
            if "humaneval" not in p.name:
                continue
            mk, ds, s, pref = parse(p.name)
            if mk == "?" or ds == "?" or s == "?":
                continue
            if pref != NAT_PREFIX[s]:
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


def main():
    files = collect()
    print(f"groups found: {len(files)}")
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
    cdf = pd.DataFrame(rows)
    cdf.to_csv(TABLES / "he_concordance.csv", index=False)
    print("wrote he_concordance.csv\n", cdf.to_string())

    # ---- scatter: gemma multi, s2/s4/s7 ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, s in zip(axes, ["s2", "s4", "s7"]):
        paths = files.get(("gemma", "multi", s), [])
        if not paths:
            ax.set_title(f"{SET_LABEL[s]} (no data)"); continue
        df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        g = pd.to_numeric(df["gen_score_typcorr"], errors="coerce")
        v = pd.to_numeric(df["val_score"], errors="coerce")
        pos = df["correct"].astype(str).str.lower().isin(["yes", "true", "1"])
        ax.scatter(g[pos], v[pos], s=8, alpha=0.3, c="green", label="correct")
        ax.scatter(g[~pos], v[~pos], s=8, alpha=0.3, c="red", label="incorrect")
        ax.set_xlabel("gen score (tc)"); ax.set_ylabel("val score"); ax.set_title(f"{SET_LABEL[s]}")
        ax.legend(fontsize=8)
    plt.suptitle("gemma-4-31b / multi — gen(tc) vs val (TEST candidates)")
    plt.tight_layout(); plt.savefig(PLOTS / "he_gen_vs_val_scatter_gemma_multi.png", dpi=150, bbox_inches="tight"); plt.close()
    print("wrote he_gen_vs_val_scatter_gemma_multi.png")

    # ---- delta histograms: per (model,ds), s2/s4/s7, per-problem (pos mean - neg mean) tc gen ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9)); axes = axes.flatten()
    for ax, (mk, ds) in zip(axes, [("gemma", "upper"), ("gemma", "multi"), ("qwen", "upper"), ("qwen", "multi")]):
        for s in ["s2", "s4", "s7"]:
            paths = files.get((mk, ds, s), [])
            deltas = []
            for p in paths:
                df = pd.read_csv(p)
                g = pd.to_numeric(df["gen_score_typcorr"], errors="coerce")
                pos = df["correct"].astype(str).str.lower().isin(["yes", "true", "1"])
                if pos.sum() and (~pos).sum():
                    deltas.append(g[pos].mean() - g[~pos].mean())
            if deltas:
                ax.hist(deltas, bins=20, alpha=0.5, label=SET_LABEL[s])
        ax.axvline(0, color="k", lw=0.6); ax.set_title(f"{mk} / {ds}")
        ax.set_xlabel("per-problem (tc-gen pos mean $-$ neg mean)"); ax.legend(fontsize=8)
    plt.suptitle("Score-delta distributions (positive should separate above 0)")
    plt.tight_layout(); plt.savefig(PLOTS / "he_score_delta_hist.png", dpi=150, bbox_inches="tight"); plt.close()
    print("wrote he_score_delta_hist.png")
    print("done.")


if __name__ == "__main__":
    main()
