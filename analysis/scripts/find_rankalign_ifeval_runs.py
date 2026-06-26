#!/usr/bin/env python3
"""Find ALL RankAlign (s2) ifeval runs and compute each run's per-problem gen ROC.

s2 = trained ifeval eval whose recipe has NONE of the other settings' tokens (no nll/fsx/
force-same-x/tc/pref/labelonly/cft/vallogodds). A RUN = (model, dir, delta, epoch, eval-prefix,
split, date); per-prompt test files of one run share that signature. Reports each run's gen ROC
(gen_score_typcorr) so the run-to-run variance of RankAlign is visible.

Run on mll, qwen35 venv.
"""
import glob, os, re
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

D2 = "/datastor2/jdr/rankalign"; D1 = "/datastor1/jdr/gv-gap/rankalign"
EXCLUDE = re.compile(r"nllv|nllg|force-same-x|-fsx-|tc-self|tc-neg|-tcs-|-tcn-|pref0|-p0-|labelonly|cft|vallogodds")


def model_of(n):
    for t, l in [("gemma-2-9b-it", "gemma-2-9b-it"), ("gemma-2-2b-it", "gemma-2-2b-it"),
                 ("gemma-2-2b", "gemma-2-2b"), ("Qwen3.5-9B", "qwen-3.5-9b")]:
        if t in n:
            return l
    return None


def parse(path):
    n = os.path.basename(path)
    if EXCLUDE.search(n) or "ifeval" not in n:
        return None
    if not re.search(r"epoch[0-9]|-e[0-9]-", n):
        return None  # trained only
    mdl = model_of(n)
    if mdl is None:
        return None
    ep = re.search(r"epoch([0-9])|-e([0-9])-", n)
    ep = (ep.group(1) or ep.group(2))
    delta = re.search(r"delta([0-9.]+)|-d([0-9.]+)-", n)
    delta = (delta.group(1) or delta.group(2)) if delta else "?"
    pfx = re.match(r"scores_([a-z]+)-", n); pfx = pfx.group(1) if pfx else "?"
    date = re.search(r"(20[0-9]{6})", n); date = date.group(1) if date else "?"
    if "ifeval-concat_train" in n:
        split, pid = "concat_train", "ALL"
    else:
        m = re.search(r"(ifeval-prompt_\d+)_test|ifeval-(id|ood)", n)
        if m and m.group(1):
            split, pid = "prompt_test", m.group(1)
        elif m:
            split, pid = m.group(2), "ALL"
        else:
            return None
    d = os.path.basename(os.path.dirname(path))
    return mdl, d, delta, ep, pfx, split, date, pid, path


def roc_of(files, split):
    vals = []
    for f in files:
        df = pd.read_csv(f)
        if "correct" not in df.columns or "gen_score_typcorr" not in df.columns:
            continue
        y = df["correct"].astype(str).str.strip().str.lower().map({"yes": 1, "true": 1, "1": 1, "no": 0, "false": 0, "0": 0})
        groups = df.groupby("prompt") if "prompt" in df.columns else [(None, df)]
        for _, g in groups:
            yy = y.loc[g.index].to_numpy(); xx = pd.to_numeric(g["gen_score_typcorr"], errors="coerce").to_numpy()
            ok = ~(np.isnan(yy) | np.isnan(xx)); yy, xx = yy[ok], xx[ok]
            if len(set(yy)) == 2 and len(yy) >= 2:
                vals.append(roc_auc_score(yy, xx))
    return (np.mean(vals) * 100 if vals else np.nan), len(vals)


def main():
    files = []
    for base in (D2, D1):
        files += glob.glob(f"{base}/outputs*/scores_*ifeval*") + glob.glob(f"{base}/outputs*/*/scores_*ifeval*")
    runs = defaultdict(list)  # (model,dir,delta,ep,pfx,split,date) -> [paths]
    for p in files:
        r = parse(p)
        if r:
            mdl, d, delta, ep, pfx, split, date, pid, path = r
            runs[(mdl, d, delta, ep, pfx, split, date)].append(path)
    # only ep2; report gen ROC per run
    rows = []
    for key, paths in runs.items():
        mdl, d, delta, ep, pfx, split, date = key
        if ep != "2":
            continue
        roc, nprob = roc_of(paths, split)
        if roc == roc:
            rows.append((mdl, split, pfx, delta, date, d, len(paths), nprob, roc))
    rows.sort(key=lambda r: (r[0], r[1], r[8]))
    print(f"{'model':14} {'split':12} {'pfx':12} {'delta':6} {'date':9} {'dir':26} {'nf':>3} {'np':>3} {'genROC':>7}")
    for r in rows:
        print(f"{r[0]:14} {r[1]:12} {r[2]:12} {r[3]:6} {r[4]:9} {r[5][:26]:26} {r[6]:>3} {r[7]:>3} {r[8]:>7.1f}")
    # summary: per (model, split) spread across runs
    print("\n=== RankAlign gen ROC spread across runs, per (model, split) ===")
    bym = defaultdict(list)
    for r in rows:
        bym[(r[0], r[1])].append(r[8])
    for (mdl, split), v in sorted(bym.items()):
        v = np.array(v)
        print(f"  {mdl:14} {split:12} n_runs={len(v):2}  min={v.min():5.1f}  max={v.max():5.1f}  mean={v.mean():5.1f}  std={v.std():4.1f}")


if __name__ == "__main__":
    main()
