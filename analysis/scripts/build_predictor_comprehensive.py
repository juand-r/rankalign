#!/usr/bin/env python3
"""Comprehensive test: does BASE validator ROC predict the FLORA-vs-RankAlign winner?

Covers MANY models (gemma-2-2b, gemma-2-2b-it, gemma-2-9b-it, gemma-2-27b-it, qwen-3.5-9b on
ifeval + hyponymy; gemma-4-31b + qwen-3.5-9b on HumanEval) and BOTH splits (train + test).

Predictor (non-circular): base (untrained) validator ROC = roc(correct, val_score). val_score is
the discriminator log-odds, INDEPENDENT of the typicality mode, so it's unambiguous.
Outcome: winner by per-problem gen ROC (gen_score_typcorr), FLORA's best of s4/s7 vs RankAlign s2.
Scoring convention (uniform): s2/s3/s4 self-direction (prefer basetyp- then self-), s7 neg-direction
(prefer basetypneg- then neg-).

Per-problem ROC: group each file by its problem key (ifeval->prompt, hyponymy->category,
humaneval->task); for per-problem TEST files that's one group/file; for pooled TRAIN files it's
many. Average over problems.

Run on mll, qwen35 venv. Writes analysis/tables/predictor_comprehensive.{csv,tex} +
analysis/plots/predictor_comprehensive.png.  Use --validate to check 9b-it train vs known values.
"""
import re, glob, os, csv, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

D2 = "/datastor2/jdr/rankalign"; D1 = "/datastor1/jdr/gv-gap/rankalign"
TABLES = f"{D2}/analysis/tables"; PLOTS = f"{D2}/analysis/plots"


def model_of(n):
    for tok, lab in [("gemma-2-2b-it", "gemma-2-2b-it"), ("gemma-2-2b", "gemma-2-2b"),
                     ("gemma-2-9b-it", "gemma-2-9b-it"), ("gemma-2-27b-it", "gemma-2-27b-it"),
                     ("Qwen3.5-9B", "qwen-3.5-9b"), ("Qwen--Qwen3", "qwen-3.5-9b"),
                     ("Qwen_Qwen3", "qwen-3.5-9b"), ("gemma-4-31", "gemma-4-31b")]:
        if tok in n:
            return lab
    return None


def task_split_key_label(n):
    """Return (task, split, group_key, label_col, problem_id) or None."""
    if "humaneval" in n:
        ds = "upper" if "correct-upper" in n else ("multi" if "correct-multi" in n else None)
        if ds is None:
            return None
        m = re.search(r"(humaneval_\d+)", n)
        pid = m.group(1) if m else "?"
        if "train-humaneval_" in n:
            return (f"HE-{ds}", "train", "task", "correct", pid)
        if "_test_" in n or "_test." in n:
            return (f"HE-{ds}", "test", "task", "correct", pid)
        return None
    if ("rosch-" in n) and ("_test" in n):
        m = re.search(r"(rosch-[a-z-]+)_test", n)
        return ("hyponymy", "test", "category", "label", m.group(1) if m else "?")
    if ("membership" in n) and ("_train" in n):
        return ("hyponymy", "train", "category", "label", "ALL")
    if ("ifeval-prompt_" in n) and ("_test" in n):
        m = re.search(r"(ifeval-prompt_\d+)_test", n)
        return ("ifeval", "test", "prompt", "correct", m.group(1) if m else "?")
    if ("ifeval-concat" in n) and ("_train" in n):
        return ("ifeval", "train", "prompt", "correct", "ALL")
    return None


def setting_of(n):
    has_ep = re.search(r"epoch[0-9]|-e[0-9]-", n)
    if not has_ep and ("v6-" in n or "-d0." not in n):  # base = untrained (no epoch token)
        if not has_ep:
            return "base", "base"
    ep = "ep2" if re.search(r"epoch2|-e2-", n) else ("ep1" if re.search(r"epoch1|-e1-", n) else ("ep0" if re.search(r"epoch0|-e0-", n) else "?"))
    if ("tc-self" in n) or ("-tcs-" in n):
        s = "s4"
    elif ("tc-neg" in n) or ("-tcn-" in n):
        s = "s7"
    elif "cft" in n:
        s = "s13"
    elif ("force-same-x" in n) or ("-fsx-" in n):
        s = "s3"
    elif ("labelonly" in n) or ("pref0" in n) or re.search(r"-p0-", n):
        s = "s1"
    else:
        s = "s2"
    return s, ep


def prefix_of(n):
    m = re.match(r"scores_(basetypneg|basetyp|self|neg)-", n)
    return m.group(1) if m else "plain"


def discover():
    dirs = sorted(set(glob.glob(f"{D2}/outputs*") + glob.glob(f"{D1}/outputs*")))
    # index[(model,task,split,setting,ep,prefix)][problem_id] = path  (dedup: keep last by name)
    idx = {}
    nseen = 0
    for d in dirs:
        for path in glob.glob(f"{d}/scores_*.csv") + glob.glob(f"{d}/*/scores_*.csv"):
            n = os.path.basename(path)
            if not any(t in n for t in ("membership", "rosch-", "ifeval", "humaneval")):
                continue
            mdl = model_of(n)
            if mdl is None:
                continue
            tsk = task_split_key_label(n)
            if tsk is None:
                continue
            task, split, key, lab, pid = tsk
            s, ep = setting_of(n)
            if s != "base" and ep == "?":
                continue
            nseen += 1
            ck = (mdl, task, split, s, ep, prefix_of(n))
            idx.setdefault(ck, {})
            if pid not in idx[ck] or path > idx[ck][pid]:
                idx[ck][pid] = path
    return idx, nseen


GROUP = {"hyponymy": ("category", "label"), "ifeval": ("prompt", "correct")}


def keylab(task):
    if task.startswith("HE-"):
        return "task", "correct"
    return GROUP[task]


def per_problem_roc(files, task, scorecol):
    key, lab = keylab(task)
    vals = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if lab not in df.columns or scorecol not in df.columns:
            continue
        y = df[lab].astype(str).str.strip().str.lower().map(
            {"yes": 1, "true": 1, "1": 1, "no": 0, "false": 0, "0": 0})
        # HumanEval (and any per-problem file) has no in-file group column: the whole file IS one
        # problem. Otherwise group by the key column (prompt / category).
        groups = df.groupby(key) if key in df.columns else [(None, df)]
        for _, g in groups:
            yy = y.loc[g.index].to_numpy()
            xx = pd.to_numeric(g[scorecol], errors="coerce").to_numpy()
            ok = ~(np.isnan(yy) | np.isnan(xx)); yy, xx = yy[ok], xx[ok]
            if len(set(yy)) == 2 and len(yy) >= 2:
                vals.append(roc_auc_score(yy, xx))
    return float(np.mean(vals) * 100) if vals else np.nan, len(vals)


def cell_files(idx, mdl, task, split, s, ep, prefixes):
    for pfx in prefixes:
        files = list(idx.get((mdl, task, split, s, ep, pfx), {}).values())
        if files:
            return files
    return []


def main():
    idx, nseen = discover()
    print(f"indexed {nseen} relevant score files into {len(idx)} cell-keys")
    models = ["gemma-2-2b", "gemma-2-2b-it", "gemma-2-9b-it", "gemma-2-27b-it", "qwen-3.5-9b",
              "gemma-4-31b"]
    tasks = ["ifeval", "hyponymy", "HE-upper", "HE-multi"]
    SELF = ["basetyp", "self"]; NEG = ["basetypneg", "neg"]; ANY = ["basetyp", "self", "basetypneg", "neg", "plain"]
    rows = []
    for mdl in models:
        for task in tasks:
            for split in ["train", "test"]:
                base_val, nbase = per_problem_roc(cell_files(idx, mdl, task, split, "base", "base", ANY), task, "val_score")
                # RankAlign scored in BOTH modes (no natural TC mode); take its best shot
                ra_self, _ = per_problem_roc(cell_files(idx, mdl, task, split, "s2", "ep2", ["basetyp", "self"]), task, "gen_score_typcorr")
                ra_neg, _ = per_problem_roc(cell_files(idx, mdl, task, split, "s2", "ep2", ["basetypneg", "neg"]), task, "gen_score_typcorr")
                s4, n4 = per_problem_roc(cell_files(idx, mdl, task, split, "s4", "ep2", SELF), task, "gen_score_typcorr")
                s7, n7 = per_problem_roc(cell_files(idx, mdl, task, split, "s7", "ep2", NEG), task, "gen_score_typcorr")
                flora = np.nanmax([s4, s7]) if (s4 == s4 or s7 == s7) else np.nan
                ra_best = np.nanmax([ra_self, ra_neg]) if (ra_self == ra_self or ra_neg == ra_neg) else np.nan
                if base_val != base_val or (ra_best != ra_best and flora != flora):
                    continue
                if ra_best == ra_best and flora == flora:
                    winner = "FLORA" if flora > ra_best else "RankAlign"
                    # convention-sensitive: would the winner flip depending on which RA mode we use?
                    cs = (ra_self == ra_self and ra_neg == ra_neg and ((flora > ra_self) != (flora > ra_neg)))
                else:
                    winner, cs = "?", False
                rows.append(dict(model=mdl, task=task, split=split, base_val_roc=base_val,
                                 ra_self=ra_self, ra_neg=ra_neg, ra_best=ra_best, flora=flora,
                                 winner=winner, conv_sensitive=cs, nbase=nbase, n_flora=max(n4, n7)))
    rows.sort(key=lambda r: (r["base_val_roc"] if r["base_val_roc"] == r["base_val_roc"] else 999))
    os.makedirs(TABLES, exist_ok=True); os.makedirs(PLOTS, exist_ok=True)
    with open(f"{TABLES}/predictor_comprehensive.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\n{'model':14} {'task':9} {'split':6} {'baseVal':>7} {'RA_self':>7} {'RA_neg':>7} {'FLORA':>6} {'winner':>10} {'conv?':>5}")
    for r in rows:
        def fz(x): return f"{x:6.1f}" if x == x else "   ---"
        print(f"{r['model']:14} {r['task']:9} {r['split']:6} {fz(r['base_val_roc'])} {fz(r['ra_self'])} {fz(r['ra_neg'])} {fz(r['flora'])} {r['winner']:>10} {'YES' if r['conv_sensitive'] else '':>5}")
    # plot
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for r in rows:
        if r["winner"] == "?" or r["base_val_roc"] != r["base_val_roc"]:
            continue
        c = "tab:green" if r["winner"] == "FLORA" else "tab:red"
        mk = "o" if r["split"] == "train" else "^"
        ec = "blue" if r["conv_sensitive"] else "k"; lw = 2.2 if r["conv_sensitive"] else 0.8
        y = r["flora"] - r["ra_best"]
        ax.scatter(r["base_val_roc"], y, c=c, marker=mk, s=80, edgecolor=ec, linewidth=lw, zorder=3)
        ax.annotate(f"{r['model'].replace('gemma-2-','g2').replace('gemma-4-31b','g4').replace('qwen-3.5-9b','qw')}/{r['task'][:5]}",
                    (r["base_val_roc"], y), fontsize=6, xytext=(3, 2), textcoords="offset points")
    ax.axhline(0, color="gray", lw=1); ax.axvline(87, color="navy", ls="--", lw=1, label="~87 threshold")
    ax.set_xlabel("BASE validator ROC (predictor, no generator)")
    ax.set_ylabel("gen-ROC margin FLORA - RankAlign(best of self/neg)")
    ax.set_title("Base validator ROC vs winner across models+tasks+splits\n"
                 "(green=FLORA, red=RankAlign; o=train, ^=test; blue ring=convention-sensitive)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(f"{PLOTS}/predictor_comprehensive.png", dpi=150, bbox_inches="tight")
    print("\nwrote predictor_comprehensive.{csv,png}")


if __name__ == "__main__":
    main()
