#!/usr/bin/env python3
"""Task #24 — original vs wandb-rerun comparison (Qwen3.5-9B).

For the qwen cells where BOTH an original checkpoint (recovered from HF latkes) AND an
independent wandb-rerun checkpoint exist — ifeval s1/s2 (OOD) and rosch s1/s7 — show the
two independently-trained models side by side, per eval-TC variant. Answers "how different
are two independent trains of the same setting?"

Sources (gen_roc + rho cells):
  orig  : metrics-from-scores-orig-hf/{ifeval_v7_ood,rosch_v7}_qwen3.5-9b_{metric}_*.csv
  rerun : metrics-from-scores-rerun-wandb/ifeval_v7_ood_qwen3.5-9b_*  +  metrics-from-scores/rosch_v7_qwen3.5-9b_*

Output: docs/orig_vs_rerun_qwen.tex  (+ standalone wrapper -> pdf)
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "orig_vs_rerun_qwen.tex"

# (task, setting_num, label) ; variants per setting follow the training template
CELLS = [("ifeval", 1, "SFT"), ("ifeval", 2, "RankAlign"),
         ("rosch", 1, "SFT"), ("rosch", 7, "FLORA-Neg")]
VARIANTS = {  # which eval-TC columns to show per setting
    1: ["PMI self", "PMI base", "Neg self", "Neg base"],
    2: ["PMI self", "PMI base", "Neg self", "Neg base"],
    7: ["Neg self", "Neg base"],
}
SRC = {
    ("orig", "ifeval"): "metrics-from-scores-orig-hf/ifeval_v7_ood_qwen3.5-9b_{m}_table_cells.csv",
    ("orig", "rosch"): "metrics-from-scores-orig-hf/rosch_v7_qwen3.5-9b_{m}_table_cells.csv",
    ("rerun", "ifeval"): "metrics-from-scores-rerun-wandb/ifeval_v7_ood_qwen3.5-9b_{m}_table_cells.csv",
    ("rerun", "rosch"): "metrics-from-scores/rosch_v7_qwen3.5-9b_{m}_table_cells.csv",
}
VLABEL = {"PMI self": "self", "PMI base": "self-base", "Neg self": "neg", "Neg base": "neg-base"}


def load(prov, task, metric):
    p = REPO / SRC[(prov, task)].format(m=metric)
    return pd.read_csv(p) if p.exists() else None


def val(df, num, col):
    if df is None:
        return None
    r = df[(df.method_num == num) & (df.column == col)]
    if len(r) and pd.notna(r.iloc[0]["mean"]):
        return float(r.iloc[0]["mean"])
    return None


def main():
    lines = [
        r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{ll cc c cc c}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{\textbf{ROC$_G$}} & \multicolumn{3}{c}{\textbf{$\rho$}} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"\textbf{Task/Method} & \textbf{eval-TC} & orig & rerun & $\Delta$ & orig & rerun & $\Delta$ \\",
        r"\midrule",
    ]
    og = {(t, "orig"): load("orig", t, "gen_roc") for t in ("ifeval", "rosch")}
    rg = {(t, "rerun"): load("rerun", t, "gen_roc") for t in ("ifeval", "rosch")}
    osp = {(t, "orig"): load("orig", t, "spearman") for t in ("ifeval", "rosch")}
    rsp = {(t, "rerun"): load("rerun", t, "spearman") for t in ("ifeval", "rosch")}
    for task, num, label in CELLS:
        tasklab = ("IFEval" if task == "ifeval" else "Hyp") + f"/{label}"
        rows = VARIANTS[num]
        for k, col in enumerate(rows):
            head = f"\\multirow{{{len(rows)}}}{{*}}{{{tasklab}}}" if k == 0 else ""
            def cells(ogd, rgd):
                o = val(ogd, num, col); r = val(rgd, num, col)
                if o is None and r is None:
                    return "--- & --- & ---"
                d = f"{o - r:+.1f}" if (o is not None and r is not None) else "--"
                return f"{o:.1f}" if o is not None else "---", \
                       (f"{r:.1f}" if r is not None else "---"), d
            o1, r1, d1 = cells(og[(task, "orig")], rg[(task, "rerun")])
            o2, r2, d2 = cells(osp[(task, "orig")], rsp[(task, "rerun")])
            lines.append(f"{head} & {VLABEL[col]} & {o1} & {r1} & {d1} & {o2} & {r2} & {d2} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}",
              r"\caption{\textbf{Original vs wandb-rerun} (Qwen3.5-9B): two independently-trained "
              r"checkpoints of the same setting, evaluated identically. \emph{orig} = paper checkpoint "
              r"recovered from HF (latkes); \emph{rerun} = the mll wandb rerun. $\Delta$ = orig $-$ rerun "
              r"(pp). IFEval is OOD (20 prompts); Hyponymy is the 10 rosch tasks. Only the four cells "
              r"with both checkpoints are shown (qwen ifeval s1/s2, rosch s1/s7). All epoch 2.}",
              r"\label{tab:orig-vs-rerun-qwen}", r"\end{table}"]
    OUT.write_text("% Task #24: orig vs rerun (qwen). Needs \\multirow.\n" + "\n".join(lines) + "\n")
    print(f"wrote {OUT}")
    # text summary
    print("\n--- ROC_G orig vs rerun (qwen) ---")
    for task, num, label in CELLS:
        for col in VARIANTS[num]:
            o = val(og[(task, "orig")], num, col); r = val(rg[(task, "rerun")], num, col)
            if o is not None or r is not None:
                os_ = f"{o:.1f}" if o is not None else "--"
                rs_ = f"{r:.1f}" if r is not None else "--"
                dd = f"{o-r:+.1f}" if (o is not None and r is not None) else ""
                print(f"  {task}/{label:9s} {VLABEL[col]:9s}: orig={os_:>5} rerun={rs_:>5} Δ={dd}")


if __name__ == "__main__":
    main()
