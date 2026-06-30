#!/usr/bin/env python3
"""Self/neg/basetyp/basetypneg gen ROC for the 3 pod-env qwen IFEval RankAlign(s2) variance runs.

Per-prompt ROC of gen_score_typcorr vs `correct` over the 20 OOD ifeval-prompt_N_test CSVs,
averaged over prompts (mean ± SE). Pure-python AUC (no sklearn). Tests whether the original
83.2 self-mode result is reproducible in the original pod environment (=environment) or a
high-variance draw (=variance). Scores live under the monitor dir; point BASE at them.
"""
import csv, glob, os, statistics, math

BASE = os.path.expanduser("~/.claude/training_monitor/qwen_variance/scores")
MODES = ("self", "neg", "basetyp", "basetypneg")


def auc(rows):
    pos = [s for s, l in rows if l == 1]; neg = [s for s, l in rows if l == 0]
    if not pos or not neg:
        return None
    c = sum(1.0 if p > n else (0.5 if p == n else 0.0) for p in pos for n in neg)
    return c / (len(pos) * len(neg))


def file_roc(path, col="gen_score_typcorr"):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            cv = str(r.get("correct", "")).strip().lower()
            lab = 1 if cv in ("yes", "true", "1") else (0 if cv in ("no", "false", "0") else None)
            try:
                sc = float(r.get(col, ""))
            except ValueError:
                sc = None
            if lab is not None and sc is not None and not math.isnan(sc):
                rows.append((sc, lab))
    return auc(rows)


def mode_roc(run, prefix, col="gen_score_typcorr"):
    files = glob.glob(f"{BASE}/run{run}/scores_{prefix}-*ifeval-prompt_*_test*")
    vals = [v for f in files if (v := file_roc(f, col)) is not None]
    if not vals:
        return None
    m = statistics.mean(vals) * 100
    se = (statistics.stdev(vals) / math.sqrt(len(vals)) * 100) if len(vals) > 1 else float("nan")
    return m, se, len(vals)


def main():
    print(f"{'run':5} " + " ".join(f"{m:>14}" for m in MODES))
    selfs = []
    for run in ("1", "2", "3"):
        cells = []
        for pfx in MODES:
            r = mode_roc(run, pfx)
            cells.append(f"{r[0]:.1f}±{r[1]:.1f}(n{r[2]})" if r else "--")
            if pfx == "self" and r:
                selfs.append(r[0])
        print(f"run{run:2} " + " ".join(f"{c:>14}" for c in cells))
    if len(selfs) == 3:
        print(f"\nSELF gen ROC: run1={selfs[0]:.1f} run2={selfs[1]:.1f} run3={selfs[2]:.1f} "
              f"| mean={statistics.mean(selfs):.1f} std={statistics.pstdev(selfs):.2f} "
              f"range={min(selfs):.1f}-{max(selfs):.1f}")
        print("vs original pod 83.2 ; vs mll reruns 67.9/76.0/75.4 (mean 73.1)")


if __name__ == "__main__":
    main()
