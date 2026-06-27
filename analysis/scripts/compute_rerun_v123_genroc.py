#!/usr/bin/env python3
"""v1 vs v2 vs v3 per-method per-mode gen-ROC comparison for the qwen IFEval reruns.

Three independent train->eval reruns of Qwen3.5-9B on IFEval (OOD test, prompts 1..21):
  v1 = outputs-rerun-wandb       (the original rerun batch -- MESSY: multiple deltas/dates)
  v2 = outputs-rerun-wandb-v2    (clean single rerun)
  v3 = outputs-rerun-wandb-v3    (this session's third rerun)

Goal: per-method run-to-run spread of gen ROC, RankAlign (s2) especially.

Metric: per-PROBLEM gen ROC. Each ifeval-prompt_N_test CSV is ONE problem (one IFEval
instruction over many completions); compute ROC of gen_score_typcorr vs `correct` within
the file, then average over the ~20 OOD prompt files. SE = std(ddof=1)/sqrt(n_prompts).

gen_score_typcorr DEPENDS on the typicality eval mode, so we keep the four prefixes separate:
  self       = self-typicality, no-base normalization
  neg        = neg-typicality,  no-base normalization
  basetyp    = self-typicality, base-model normalization
  basetypneg = neg-typicality,  base-model normalization

Setting classification (robust to both long `force-same-x` and short `fsx` token schemes):
  s4 FLORA-PMI  : has tc-self / -tcs-
  s7 FLORA-Neg  : has tc-neg  / -tcn-
  s1 SFT        : (no tc) has labelonly / -lo0.x-
  s3 New+fsx    : (no tc, no labelonly) has force-same-x / -fsx-
  s2 RankAlign  : none of the above (vallogodds + semi only)

v1 dedup: v1 has several (delta, date) runs per (setting, mode). The summary picks, per setting,
the v1 run whose delta MATCHES v3's delta for that setting (data-driven, read from v3 filenames);
if several dates share that delta, the LATEST date is used. The full enumeration is printed first
so every v1 run is visible -- nothing is hidden or silently averaged.

Run on mll, qwen35 venv (numpy/pandas/sklearn). Read-only over the three outputs dirs.
"""
import glob
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

D2 = "/datastor2/jdr/rankalign"
DIRS = {"v1": f"{D2}/outputs-rerun-wandb", "v2": f"{D2}/outputs-rerun-wandb-v2",
        "v3": f"{D2}/outputs-rerun-wandb-v3"}
PREFIXES = ["self", "neg", "basetyp", "basetypneg"]
SETTINGS = ["s1", "s2", "s3", "s4", "s7"]
SETTING_LABEL = {"s1": "SFT", "s2": "RankAlign", "s3": "New+fsx",
                 "s4": "FLORA-PMI", "s7": "FLORA-Neg"}
# which modes are meaningful per setting (matches the sbatch EVAL_MODES):
SETTING_MODES = {"s1": ["self", "neg", "basetyp", "basetypneg"],
                 "s2": ["self", "neg", "basetyp", "basetypneg"],
                 "s3": ["self", "neg", "basetyp", "basetypneg"],
                 "s4": ["self", "basetyp"],          # self-typicality only
                 "s7": ["neg", "basetypneg"]}        # neg-typicality only

OOD_MAX = 21  # OOD = ifeval prompts 1..21


def classify_setting(name: str) -> str | None:
    if re.search(r"tc-self|-tcs-", name):
        return "s4"
    if re.search(r"tc-neg|-tcn-", name):
        return "s7"
    if re.search(r"labelonly|-lo[0-9]", name):
        return "s1"
    if re.search(r"force-same-x|-fsx-", name):
        return "s3"
    return "s2"


def prefix_of(name: str) -> str | None:
    m = re.match(r"scores_(basetypneg|basetyp|neg|self)-", name)
    return m.group(1) if m else None


def is_qwen_ifeval_ood_test(name: str) -> int | None:
    if "Qwen3.5-9B" not in name and "Qwen--Qwen3.5-9B" not in name:
        return None
    if not re.search(r"-e2-|epoch2", name):  # epoch2 only
        return None
    m = re.search(r"ifeval-prompt_(\d+)_test", name)
    if not m:
        return None
    p = int(m.group(1))
    return p if p <= OOD_MAX else None


def delta_of(name: str) -> str:
    m = re.search(r"-d([0-9.]+)-e2|delta([0-9.]+)-epoch2", name)
    return (m.group(1) or m.group(2)) if m else "?"


def date_of(name: str) -> str:
    m = re.search(r"_(20[0-9]{6})\.csv", name)
    return m.group(1) if m else "?"


def file_roc(path: str) -> float | None:
    """ROC of gen_score_typcorr vs correct within one prompt-file (one problem)."""
    df = pd.read_csv(path)
    if "correct" not in df.columns or "gen_score_typcorr" not in df.columns:
        return None
    y = (df["correct"].astype(str).str.strip().str.lower()
         .map({"yes": 1, "true": 1, "1": 1, "no": 0, "false": 0, "0": 0}))
    x = pd.to_numeric(df["gen_score_typcorr"], errors="coerce")
    ok = ~(y.isna() | x.isna())
    y, x = y[ok].to_numpy(), x[ok].to_numpy()
    if len(set(y.tolist())) != 2 or len(y) < 2:
        return None
    return roc_auc_score(y, x)


def run_roc(files: list[str]) -> tuple[float, float, int]:
    rocs = [r for f in files if (r := file_roc(f)) is not None]
    if not rocs:
        return float("nan"), float("nan"), 0
    a = np.array(rocs) * 100
    se = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float("nan")
    return a.mean(), se, len(a)


def collect(d: str):
    """dir -> {(prefix, setting): {(delta, date): [files]}}"""
    runs = defaultdict(lambda: defaultdict(list))
    for path in glob.glob(f"{d}/scores_*ifeval-prompt_*_test*"):
        name = os.path.basename(path)
        pfx = prefix_of(name)
        if pfx is None or is_qwen_ifeval_ood_test(name) is None:
            continue
        st = classify_setting(name)
        runs[(pfx, st)][(delta_of(name), date_of(name))].append(path)
    return runs


def main():
    allruns = {v: collect(p) for v, p in DIRS.items()}

    # ---- (A) full enumeration: every distinct run ----
    print("=" * 100)
    print("(A) ENUMERATION — every distinct (delta,date) run per version/mode/setting")
    print("=" * 100)
    print(f"{'ver':3} {'mode':11} {'set':3} {'method':10} {'delta':6} {'date':9} {'nP':>3} {'genROC':>7} {'SE':>5}")
    for v in ["v1", "v2", "v3"]:
        for pfx in PREFIXES:
            for st in SETTINGS:
                groups = allruns[v].get((pfx, st), {})
                for (delta, date), files in sorted(groups.items()):
                    m, se, n = run_roc(files)
                    if n == 0:
                        continue
                    print(f"{v:3} {pfx:11} {st:3} {SETTING_LABEL[st]:10} {delta:6} {date:9} "
                          f"{n:>3} {m:>7.1f} {se:>5.1f}")
        print("-" * 100)

    # ---- v1 dedup: pick run matching v3's delta per (setting, mode); latest date ----
    def v3_delta(pfx, st):
        groups = allruns["v3"].get((pfx, st), {})
        ds = {delta for (delta, _date) in groups}
        return next(iter(ds)) if len(ds) == 1 else None

    def n_of(files):
        return run_roc(files)[2]

    def best(cands, groups):
        # pick the MOST COMPLETE run (max n_prompts); tie-break by latest date.
        # avoids selecting partial/broken fragments (e.g. v1's 2-prompt 20260609 runs).
        return max(cands, key=lambda k: (n_of(groups[k]), k[1]))

    def pick(v, pfx, st):
        groups = allruns[v].get((pfx, st), {})
        if not groups:
            return None
        if v in ("v2", "v3"):
            key = best(list(groups), groups)
            return key[0], key[1], groups[key]
        # v1: match v3's delta if known, else any; then most-complete run
        tgt = v3_delta(pfx, st)
        cand = [k for k in groups if tgt is None or k[0] == tgt]
        if not cand:
            cand = list(groups)  # no delta match -> fall back, will be noted
        key = best(cand, groups)
        return key[0], key[1], groups[key]

    # ---- (B) summary tables per mode ----
    for mode in ["self", "basetyp", "neg", "basetypneg"]:
        print("\n" + "=" * 100)
        print(f"(B) SUMMARY — gen ROC (mean ± SE over OOD prompts), mode = {mode}")
        print("=" * 100)
        print(f"{'method':12} {'v1 (delta/date)':24} {'v2 (delta/date)':24} {'v3 (delta/date)':24}")
        for st in SETTINGS:
            if mode not in SETTING_MODES[st]:
                continue
            cells = []
            for v in ["v1", "v2", "v3"]:
                p = pick(v, mode, st)
                if p is None:
                    cells.append(("--", ""))
                    continue
                delta, date, files = p
                m, se, n = run_roc(files)
                cells.append((f"{m:.1f}±{se:.1f} (n{n})", f"d{delta}/{date[4:]}"))
            row = f"{SETTING_LABEL[st]:12}"
            for val, tag in cells:
                row += f" {val:>14} {tag:9}"
            print(row)

        # RankAlign spread highlight
        vals = []
        for v in ["v1", "v2", "v3"]:
            p = pick(v, mode, "s2")
            if p:
                m, _se, n = run_roc(p[2])
                if n:
                    vals.append((v, m))
        if len(vals) >= 2:
            a = np.array([x for _v, x in vals])
            print(f"  RankAlign spread ({mode}): " +
                  ", ".join(f"{v}={x:.1f}" for v, x in vals) +
                  f"  -> min={a.min():.1f} max={a.max():.1f} range={a.max()-a.min():.1f} "
                  f"std={a.std(ddof=1):.1f}" if len(a) > 1 else "")


if __name__ == "__main__":
    main()
