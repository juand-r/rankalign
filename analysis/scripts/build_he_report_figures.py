#!/usr/bin/env python3
"""Generate the remaining figures/tables to complete report_humaneval.tex to full structural
parity with analysis/report.tex. Runs ON mll (matplotlib + sklearn + the score dirs live there).

Produces (into analysis/plots/ and analysis/tables/):
  he_unified_genroc.png          (orig 1.3) test tc gen ROC per setting x model x dataset
  he_concordance_train.csv       (orig 2.4) per-problem gen/val concordance, TRAIN, mean+-SE -> table
  he_score_delta_hist.png        (orig 2.7) per-problem (tc-gen pos mean - neg mean), TRAIN
  he_train_dynamics.png          (orig 2.8) tc gen ROC vs epoch (base/ep0/ep1/ep2), TRAIN per-problem
  he_gen_vs_val_scatter.png      (orig 2.9) gen(tc) vs val, correct/incorrect, gemma multi base/s2/s4, TRAIN
  he_perproblem_improvement.png  (orig 2.10 substitute) per-problem TC effect (s7-s3) distribution, TEST

Train (per-problem) raw scores: outputs-he-trainset-perproblem/. Test raw scores:
outputs_gemma4_mll_tmp[-multi]/ (gemma) + outputs-rerun-wandb/ (qwen, humaneval only).
Setting parse uses the FULL 'force-same-x' token (NOT the abbreviated 'fsx').
"""
import re, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

REPO = "/datastor2/jdr/rankalign"
PLOTS = f"{REPO}/analysis/plots"; TABLES = f"{REPO}/analysis/tables"
os.makedirs(PLOTS, exist_ok=True); os.makedirs(TABLES, exist_ok=True)
TRAIN_DIR = f"{REPO}/outputs-he-trainset-perproblem"
TEST_DIRS = [f"{REPO}/outputs_gemma4_mll_tmp", f"{REPO}/outputs_gemma4_mll_tmp-multi", f"{REPO}/outputs-rerun-wandb"]

SETTINGS = ["base", "s1", "s2", "s3", "s4", "s7", "s13"]
SET_LABEL = {"base": "Base", "s1": "SFT", "s2": "RankAlign", "s3": "New+fsx",
             "s4": "FLORA-PMI", "s7": "FLORA-Neg", "s13": "ConsFT"}
NAT = {"base": "self/no-base", "s1": "self/base-typ", "s2": "neg/base-typ", "s3": "neg/base-typ",
       "s4": "self/base-typ", "s7": "neg/base-typ", "s13": "self/base-typ"}
NAT_PREFIX = {"self/no-base": "self", "self/base-typ": "basetyp",
              "neg/no-base": "neg", "neg/base-typ": "basetypneg"}


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


def per_problem_roc(df):
    y = df["correct"].astype(str).str.lower().isin(["yes", "true", "1"]).to_numpy().astype(int)
    g = pd.to_numeric(df.get("gen_score_typcorr"), errors="coerce").to_numpy()
    ok = ~np.isnan(g)
    if len(set(y[ok])) < 2 or ok.sum() < 2: return np.nan
    return roc_auc_score(y[ok], g[ok])


def concordance(df):
    g = pd.to_numeric(df.get("gen_score_typcorr"), errors="coerce").to_numpy()
    v = pd.to_numeric(df.get("val_score"), errors="coerce").to_numpy()
    ok = ~(np.isnan(g) | np.isnan(v)); g, v = g[ok], v[ok]; n = len(g)
    if n < 2: return np.nan
    ag = tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            dg, dv = g[i]-g[j], v[i]-v[j]
            if dg == 0 or dv == 0: continue
            tot += 1; ag += (np.sign(dg) == np.sign(dv))
    return ag/tot if tot else np.nan


def pos_mask(df):
    return df["correct"].astype(str).str.lower().isin(["yes", "true", "1"])


def collect(dirs, train):
    """dict[(mk,ds,setting,epoch,evalmode)] -> [paths]; train picks the per-problem dir naming."""
    out = {}
    pat = "scores_*train-humaneval_*.csv" if train else "scores_*_test_*.csv"
    for d in dirs:
        for p in glob.glob(f"{d}/{pat}"):
            if "humaneval" not in os.path.basename(p): continue
            mk, ds, s, ep, mode = parse(os.path.basename(p))
            if "?" in (mk, ds, s, ep): continue
            out.setdefault((mk, ds, s, ep, mode), []).append(p)
    return out


print("collecting TRAIN (per-problem) + TEST score files ...")
TR = collect([TRAIN_DIR], train=True)
TE = collect(TEST_DIRS, train=False)
print(f"  train groups={len(TR)}  test groups={len(TE)}")

# ---------- 2.4 concordance table (TRAIN, ep2, natural mode) ----------
rows = []
for mk in ["gemma", "qwen"]:
    for ds in ["upper", "multi"]:
        for s in SETTINGS:
            ep = "base" if s == "base" else "ep2"
            paths = TR.get((mk, ds, s, ep, NAT[s]), [])
            vals = [concordance(pd.read_csv(p)) for p in paths]
            vals = [x for x in vals if not np.isnan(x)]
            if vals:
                a = np.array(vals)
                rows.append(dict(model=mk, dataset=ds, setting=s, n=len(a),
                                 concordance=a.mean(), se=a.std(ddof=1)/np.sqrt(len(a))))
pd.DataFrame(rows).to_csv(f"{TABLES}/he_concordance_train.csv", index=False)
print("wrote he_concordance_train.csv", len(rows), "cells")

# ---------- 1.3 unified gen-ROC bar (TEST, tc, natural mode) ----------
def test_roc(mk, ds, s):
    ep = None  # test = epoch2 trained (or base)
    # test files: setting at ep2 (or base for base); pick natural mode
    key = None
    for (m2, d2, s2, e2, mode2), paths in TE.items():
        if m2 == mk and d2 == ds and s2 == s and mode2 == NAT[s] and (e2 == "ep2" or s == "base"):
            key = (m2, d2, s2, e2, mode2); break
    if key is None: return np.nan
    vals = [per_problem_roc(pd.read_csv(p)) for p in TE[key]]
    vals = [x for x in vals if not np.isnan(x)]
    return np.mean(vals)*100 if vals else np.nan

groups = [("gemma", "upper"), ("gemma", "multi"), ("qwen", "upper"), ("qwen", "multi")]
fig, ax = plt.subplots(figsize=(13, 5)); x = np.arange(len(groups)); w = 0.12
for i, s in enumerate(SETTINGS):
    vals = [test_roc(mk, ds, s) for mk, ds in groups]
    ax.bar(x + (i-3)*w, vals, w, label=SET_LABEL[s])
ax.set_xticks(x); ax.set_xticklabels([f"{m}\n{d}" for m, d in groups]); ax.set_ylim(50, 100)
ax.set_ylabel("tc gen ROC x100 (TEST)"); ax.legend(ncol=7, fontsize=8)
ax.set_title("Unified gen ROC by setting (TEST, tc, natural eval mode)")
plt.tight_layout(); plt.savefig(f"{PLOTS}/he_unified_genroc.png", dpi=150, bbox_inches="tight"); plt.close()
print("wrote he_unified_genroc.png")

# ---------- 2.8 train dynamics: tc gen ROC vs epoch (TRAIN per-problem) ----------
fig, axes = plt.subplots(2, 2, figsize=(13, 9)); axes = axes.flatten()
EPS = ["base", "ep0", "ep1", "ep2"]
for ax, (mk, ds) in zip(axes, groups):
    for s in ["s2", "s3", "s4", "s7"]:
        ys = []
        for ep in EPS:
            mode = NAT[s] if ep != "base" else NAT["base"]
            paths = TR.get((mk, ds, s, ep, mode), []) or TR.get((mk, ds, "base", "base", NAT["base"]), []) if ep == "base" else TR.get((mk, ds, s, ep, NAT[s]), [])
            vals = [per_problem_roc(pd.read_csv(p)) for p in paths]; vals = [v for v in vals if not np.isnan(v)]
            ys.append(np.mean(vals)*100 if vals else np.nan)
        ax.plot(EPS, ys, marker="o", label=SET_LABEL[s])
    ax.set_title(f"{mk} / {ds}"); ax.set_ylabel("tc gen ROC x100 (TRAIN)"); ax.legend(fontsize=8)
plt.suptitle("Per-epoch TRAIN dynamics (per-problem, tc gen ROC)")
plt.tight_layout(); plt.savefig(f"{PLOTS}/he_train_dynamics.png", dpi=150, bbox_inches="tight"); plt.close()
print("wrote he_train_dynamics.png")

# ---------- 2.7 score-delta hist (TRAIN, ep2) ----------
# EXACT port of the original report (analyze_gemma_membership.compute_delta_histograms):
# pool all candidates, sample ~100k random pairs, take pairwise |score_i - score_j| for the
# generator (gen_score_typcorr) and validator (val_score), and overlay the two density
# histograms per panel. This is a score-SPREAD / calibration diagnostic ("tighter = better
# calibrated"), Gen vs Val -- NOT a correct-vs-incorrect separation.
DELTA_SETTINGS = ["s2", "s3", "s4", "s7"]


def pairwise_deltas(mk, ds, s):
    ep = "base" if s == "base" else "ep2"
    paths = TR.get((mk, ds, s, ep, NAT[s]), [])
    if not paths:
        return None, None
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    gen = pd.to_numeric(df["gen_score_typcorr"], errors="coerce").dropna().to_numpy()
    val = pd.to_numeric(df["val_score"], errors="coerce").dropna().to_numpy()
    np.random.seed(42)

    def deltas(a):
        n = len(a)
        if n < 2:
            return np.array([])
        npairs = min(100000, n * (n - 1) // 2)
        i = np.random.randint(0, n, npairs); j = np.random.randint(0, n, npairs); m = i != j
        return np.abs(a[i[m]] - a[j[m]])
    return deltas(gen), deltas(val)


fig, axes = plt.subplots(len(groups), len(DELTA_SETTINGS),
                         figsize=(4 * len(DELTA_SETTINGS), 3 * len(groups)), squeeze=False)
for ri, (mk, ds) in enumerate(groups):
    cache = {s: pairwise_deltas(mk, ds, s) for s in DELTA_SETTINGS}
    allv = [d for gv in cache.values() for d in gv if d is not None and len(d)]
    xmax = float(np.percentile(np.concatenate(allv), 99)) if allv else 10.0
    for ci, s in enumerate(DELTA_SETTINGS):
        ax = axes[ri][ci]; gd, vd = cache[s]
        if gd is None or not len(gd):
            ax.set_title(f"{mk}/{ds} {SET_LABEL[s]}\n(no data)"); continue
        ax.hist(gd, bins=50, alpha=0.6, density=True, range=(0, xmax), color="blue",
                label=f"Gen |$\\Delta$| ($\\mu$={gd.mean():.1f})")
        ax.hist(vd, bins=50, alpha=0.6, density=True, range=(0, xmax), color="orange",
                label=f"Val |$\\Delta$| ($\\mu$={vd.mean():.1f})")
        ax.set_xlim(0, xmax); ax.set_title(f"{mk}/{ds} {SET_LABEL[s]}")
        ax.legend(fontsize=6); ax.set_xlabel("|score$_i$ - score$_j$|")
plt.suptitle("Pairwise score-delta distributions (TRAIN, ep2): Gen vs Val |$\\Delta$|")
plt.tight_layout(); plt.savefig(f"{PLOTS}/he_score_delta_hist.png", dpi=150, bbox_inches="tight"); plt.close()
print("wrote he_score_delta_hist.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, s in zip(axes, ["base", "s2", "s4"]):
    ep = "base" if s == "base" else "ep2"
    paths = TR.get(("gemma", "multi", s, ep, NAT[s]), [])
    if not paths: ax.set_title(f"{SET_LABEL[s]} (no data)"); continue
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    g = pd.to_numeric(df["gen_score_typcorr"], errors="coerce"); v = pd.to_numeric(df["val_score"], errors="coerce"); pm = pos_mask(df)
    ax.scatter(g[pm], v[pm], s=8, alpha=.3, c="green", label="correct")
    ax.scatter(g[~pm], v[~pm], s=8, alpha=.3, c="red", label="incorrect")
    ax.set_xlabel("gen (tc)"); ax.set_ylabel("val"); ax.set_title(SET_LABEL[s]); ax.legend(fontsize=8)
plt.suptitle("gemma-4 / multi gen(tc) vs val (TRAIN, ep2)")
plt.tight_layout(); plt.savefig(f"{PLOTS}/he_gen_vs_val_scatter.png", dpi=150, bbox_inches="tight"); plt.close()
print("wrote he_gen_vs_val_scatter.png")

# ---------- 2.10 substitute: per-problem TC effect (s7-s3) distribution, TEST ----------
fig, axes = plt.subplots(2, 2, figsize=(12, 8)); axes = axes.flatten()
for ax, (mk, ds) in zip(axes, groups):
    def roc_by_prob(s, mode):
        out = {}
        for (m2, d2, s2, e2, mo2), paths in TE.items():
            if (m2, d2, s2, mo2) == (mk, ds, s, mode) and (e2 == "ep2" or s == "base"):
                for p in paths:
                    pr = re.search(r"(humaneval_\d+)", os.path.basename(p))
                    if pr: out[pr.group(1)] = per_problem_roc(pd.read_csv(p))
        return out
    s7 = roc_by_prob("s7", "neg/base-typ"); s3 = roc_by_prob("s3", "neg/base-typ")
    common = [k for k in s7 if k in s3 and not np.isnan(s7[k]) and not np.isnan(s3[k])]
    if common:
        diff = np.array([(s7[k]-s3[k]) for k in common])*100
        ax.hist(diff, bins=20, color="steelblue", alpha=.85); ax.axvline(0, color="k")
        ax.axvline(diff.mean(), color="r", ls="--", label=f"mean {diff.mean():+.1f}"); ax.legend(fontsize=8)
    ax.set_title(f"{mk}/{ds}: per-problem s7$-$s3"); ax.set_xlabel("$\\Delta$ gen ROC x100 (TEST)")
plt.suptitle("Per-problem TC effect (neg-TC vs no-TC), TEST")
plt.tight_layout(); plt.savefig(f"{PLOTS}/he_perproblem_improvement.png", dpi=150, bbox_inches="tight"); plt.close()
print("wrote he_perproblem_improvement.png")
print("DONE")
