"""Aggregate per-problem TRAIN-set metrics (over the 80 train problems, mean +/- SE) and build
the corrected train-side tables for analysis/report_humaneval.tex -- replacing the broken
global-pool numbers.

Train source: perproblem_train_perfile_metrics.csv (from scripts/harvest_he_perproblem.sh;
one row per train problem x gen-variant). Test source (for the train-vs-test table): the
he_harvest_2026-06-17 per-file metrics (82 test problems).

Emits:
  humaneval_TRAINPP_aggregated.csv          -- all cells: n, mean, se per metric/variant.
  he_traintest_perproblem.tex               -- train(80 problems) vs test(82 problems) gen ROC
                                               tc, epoch2, each method in its natural eval mode.
Cells with n<80 are marked incomplete (batch still running).
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TRAIN = HERE / "perproblem_train_perfile_metrics.csv"
TESTDIR = HERE.parent / "he_harvest_2026-06-17"
TEST_FILES = ["qwen_he_perfile_metrics.csv", "gemma_cu_perfile_metrics.csv", "gemma_cm_perfile_metrics.csv"]

SETTINGS = ["base", "s2", "s3", "s4", "s7"]
SET_LABEL = {"base": "Base", "s2": "RankAlign", "s3": "New+fsx", "s4": "FLORA-PMI", "s7": "FLORA-Neg"}
NAT = {"base": "self/no-base", "s2": "neg/base-typ", "s3": "neg/base-typ",
       "s4": "self/base-typ", "s7": "neg/base-typ"}


def parse(f, want_epoch):
    m = re.match(r"scores_(basetypneg|basetyp|self|neg)-", f)
    pref = m.group(1) if m else "?"
    is_base = ("v6-" in f) and ("-v7-" not in f)
    mk = "qwen" if ("Qwen3.5-9B" in f or "Qwen--Qwen3" in f or "Qwen_Qwen3" in f) else ("gemma" if "gemma-4-31" in f.lower() else "?")
    ds = "upper" if "correct-upper" in f else ("multi" if "correct-multi" in f else "?")
    if is_base:
        s = "base"
    elif ("tcs" in f) or ("tc-self" in f):
        s = "s4"
    elif ("tcn" in f) or ("tc-neg" in f):
        s = "s7"
    elif "cft" in f:
        s = "s13"
    elif ("force-same-x" in f) or ("-fsx-" in f):   # FULL token (the abbreviated-only check is a bug)
        s = "s3"
    elif ("labelonly" in f) or re.search(r"-p0-", f) or ("pref0" in f):
        s = "s1"
    else:
        s = "s2"
    typ = "neg" if pref in ("neg", "basetypneg") else "self"
    bt = "base-typ" if pref in ("basetyp", "basetypneg") else "no-base"
    out = [mk, ds, s, f"{typ}/{bt}"]
    if want_epoch:
        ep = "base" if is_base else ("ep2" if re.search(r"epoch2|-e2-", f) else ("ep1" if re.search(r"epoch1|-e1-", f) else ("ep0" if re.search(r"epoch0|-e0-", f) else "?")))
        out.append(ep)
    return out


def agg(perfile, want_epoch, col="gen_roc", variant="tc"):
    idx = ["mk", "ds", "setting", "evalmode"] + (["epoch"] if want_epoch else [])
    meta = perfile.file.apply(lambda f: pd.Series(parse(f, want_epoch), index=idx))
    a = pd.concat([meta, perfile], axis=1)
    a = a[a.variant == variant]
    g = a.groupby(idx).agg(n=(col, "size"),
                           mean=(col, "mean"),
                           se=(col, lambda s: s.std(ddof=1) / np.sqrt(s.notna().sum()))).reset_index()
    return g


def main():
    train = pd.read_csv(TRAIN)
    gtr = agg(train, want_epoch=True)
    gtr.to_csv(HERE / "humaneval_TRAINPP_aggregated.csv", index=False)
    test = pd.concat([pd.read_csv(TESTDIR / f) for f in TEST_FILES], ignore_index=True)
    test = test[test.file.str.contains("humaneval", na=False)]
    gte = agg(test, want_epoch=False)

    def tr(mk, ds, s):
        r = gtr[(gtr.mk == mk) & (gtr.ds == ds) & (gtr.setting == s) & (gtr.epoch == ("base" if s == "base" else "ep2")) & (gtr.evalmode == NAT[s])]
        return None if r.empty else (r["mean"].iloc[0] * 100, r["se"].iloc[0] * 100, int(r["n"].iloc[0]))

    def te(mk, ds, s):
        r = gte[(gte.mk == mk) & (gte.ds == ds) & (gte.setting == s) & (gte.evalmode == NAT[s])]
        return None if r.empty else (r["mean"].iloc[0] * 100, r["se"].iloc[0] * 100, int(r["n"].iloc[0]))

    lines = [r"\begin{tabular}{lll ccc}", r"\toprule",
             r"Model & Data & Setting & Train (80 prob) & Test (82 prob) & $\Delta$ \\", r"\midrule"]
    for mk, ml in [("gemma", "gemma-4-31b"), ("qwen", "qwen-3.5-9b")]:
        for ds in ["upper", "multi"]:
            for s in SETTINGS:
                a, b = tr(mk, ds, s), te(mk, ds, s)
                def fmt(x):
                    if x is None: return "---"
                    return (f"{x[0]:.1f}\\stdv{{{x[1]:.1f}}}" + ("" if x[2] >= 80 else f"$^{{(n{x[2]})}}$"))
                d = "---" if (a is None or b is None) else f"{a[0]-b[0]:+.1f}"
                lines.append(f"{ml} & {ds} & {SET_LABEL[s]} & {fmt(a)} & {fmt(b)} & {d} " + r"\\")
            lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"; lines.append(r"\end{tabular}")
    (HERE / "he_traintest_perproblem.tex").write_text("\n".join(lines) + "\n")
    print("wrote humaneval_TRAINPP_aggregated.csv + he_traintest_perproblem.tex")
    nfull = (gtr.n >= 80).sum()
    print(f"train cells aggregated: {len(gtr)}  (n>=80 complete: {nfull})")


if __name__ == "__main__":
    main()
