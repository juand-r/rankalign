#!/usr/bin/env python3
"""Reproducibility check: diff the v2 fresh-retrain qwen IFEval-OOD metrics against the
current rerun-only numbers (the ones in docs/rerun_only_tables.pdf), per setting/column/metric.

v2  = metrics-from-scores-rerun-v2   (independent retrain: models2-rerun-wandb-v2,
       jobs 44680-44684, same flags/venv as the original rerun — verified)
cur = metrics-from-scores-rerun-only (the current rerun-only table source)

Both built by _build_ifeval_table_v7.py (qwen3.5-9b, ood). Flags |Δ|>5.

Finding (2026-06-10): SFT/New+fsx/FLORA-PMI/FLORA-Neg reproduce within ~±2.6 on every
metric (incl. the SFT validator: val_roc 82.8->82.4). RankAlign (s2) does NOT reproduce
(gen_roc +15, rho +44, validator +15-18) — genuinely high-variance run-to-run.

Run:  source ~/venvs/venv_lexcons/bin/activate && python scripts/_compare_v2_vs_rerun.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CUR = REPO / "metrics-from-scores-rerun-only"
V2 = REPO / "metrics-from-scores-rerun-v2"
LAB = {0: "Base", 1: "SFT", 2: "RankAlign", 3: "New+fsx", 4: "FLORA-PMI", 7: "FLORA-Neg"}


def _load(d: Path, met: str) -> pd.DataFrame:
    return pd.read_csv(d / f"ifeval_v7_ood_qwen3.5-9b_{met}_table_cells.csv")


def main() -> None:
    print("qwen IFEval OOD — v2 (fresh retrain) vs current rerun.  Δ = v2 - current ; * = |Δ|>5\n")
    for met in ("gen_roc", "spearman"):
        a, b = _load(CUR, met), _load(V2, met)
        m = a.merge(b, on=["method_num", "column"], suffixes=("_cur", "_v2"))
        m = m[m["mean_cur"].notna() & m["mean_v2"].notna()]
        m["d"] = m["mean_v2"] - m["mean_cur"]
        print(f"-- {met} --")
        for _, r in m.sort_values(["method_num", "column"]).iterrows():
            fl = " *" if abs(r["d"]) > 5 else ""
            print(f"  {LAB.get(r['method_num'], r['method_num']):10} {r['column']:9} "
                  f"cur={r['mean_cur']:6.1f}  v2={r['mean_v2']:6.1f}  Δ={r['d']:+6.1f}{fl}")
        print()
    for met in ("val_roc", "val_acc"):  # validator: eval-TC independent -> one per setting
        a, b = _load(CUR, met), _load(V2, met)
        print(f"-- {met} (validator) --")
        for num in sorted(set(a["method_num"]) & set(b["method_num"])):
            va = a[(a.method_num == num) & a["mean"].notna()]
            vb = b[(b.method_num == num) & b["mean"].notna()]
            if len(va) and len(vb):
                ca, cb = va.iloc[0]["mean"], vb.iloc[0]["mean"]
                d = cb - ca
                fl = " *" if abs(d) > 5 else ""
                print(f"  {LAB.get(num, num):10} cur={ca:6.1f}  v2={cb:6.1f}  Δ={d:+6.1f}{fl}")
        print()


if __name__ == "__main__":
    main()
