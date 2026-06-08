#!/usr/bin/env python3
"""Assemble paper-style LaTeX tables from the per-metric *_table_cells.csv files
produced by _build_rosch_table_v7.py / _build_ifeval_table_v7.py.

Two tables, for Hyponymy(rosch) + IFEval, columns = {gemma-2-9b-it, Qwen3.5-9B}:
  (1) Main  : ROC_G + rho  -- like tab:main-results-multi
  (2) Valid.: ROC_V + Acc_V -- like tab:main-results-multi-val

DIFFERENCE from main.tex: main.tex reports the per-task BEST eval-time correction.
Here we emit BOTH the self-TC and neg-TC eval explicitly, as separate rows
(Method . self / Method . neg). Per method, self-TC = whichever of {PMI self, PMI
base} is present, neg-TC = whichever of {Neg self, Neg base} (Base uses its own
typicality -> *self; trained rows use the base model's typicality -> *base).

Notes are auto-emitted for: incomplete cells (n < expected categories/prompts),
missing cells, and the epoch caveat. ROC_V / Acc_V are eval-TC-independent, so the
validator table has one row per method (no self/neg split).

Usage: python _build_paper_tables_latex.py [METRICS_DIR] [OUT.tex]
"""
import math
import sys
from pathlib import Path

import pandas as pd

# rosch builder writes cells to metrics-from-scores/; ifeval builder writes to
# metrics-from-scores-rerun-wandb/. Search both (rerun-wandb first = newest/correct).
MDIRS = [Path("metrics-from-scores-rerun-wandb"), Path("metrics-from-scores")]
if len(sys.argv) > 1:
    MDIRS = [Path(sys.argv[1])] + MDIRS
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("docs/qwen_rerun_tables.tex")

MODELS = [("G2-9b-it", "9b-it"), ("Q3.5-9b", "qwen3.5-9b")]
TASKS = [("Hyponymy", "rosch"), ("IFEval", "ifeval")]
NEXP = {"rosch": 10, "ifeval": 21}
# (paper label, method_num). qwen has no s13 (Consistency FT) -> cells absent -> "---".
ROWS = [("Base", 0), ("SFT", 1), ("RankAlign", 2),
        ("Consistency FT", 13), ("FLORA-PMI", 4), ("FLORA-Neg", 7)]
PMI_COLS = {"self": ["PMI self", "PMI base"], "neg": ["Neg self", "Neg base"]}

incomplete_notes: list[str] = []


def load(task: str, model: str, metric: str):
    for d in MDIRS:
        cands = [c for c in d.glob(f"{task}_v7_*{metric}_table_cells.csv")
                 if f"_{model}_" in c.name and "_id_" not in c.name]  # ood split for ifeval
        if cands:
            return pd.read_csv(sorted(cands)[-1])
    return None


def get(df, num, columns):
    if df is None:
        return None
    sub = df[df["method_num"] == num]
    for c in columns:
        r = sub[sub["column"] == c]
        if len(r) and pd.notna(r.iloc[0]["mean"]):
            return dict(mean=float(r.iloc[0]["mean"]), se=float(r.iloc[0]["se"]),
                        n=int(r.iloc[0]["n"]))
    return None


def fmt(c, task, model, label, metric, tc):
    if c is None:
        return "---"
    s = f"{c['mean']:.1f}\\stdv{{{c['se']:.1f}}}" if not math.isnan(c["se"]) else f"{c['mean']:.1f}"
    nexp = NEXP[task]
    if c["n"] < nexp:
        s += "$^{\\dagger}$"
        incomplete_notes.append(f"{label}{'/'+tc if tc else ''} {metric} {task}/{model}: {c['n']}/{nexp}")
    return s


def main_table() -> str:
    # cache cells frames
    cache = {(t, m, met): load(t, m, met)
             for _, t in TASKS for _, m in MODELS for met in ("gen_roc", "spearman")}
    lines = [
        r"\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{ll cccc cccc}",
        r"\toprule",
        r"& & \multicolumn{4}{c}{\textbf{Hyponymy}} & \multicolumn{4}{c}{\textbf{IFEval}} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
        r"& & \multicolumn{2}{c}{G2-9b-it} & \multicolumn{2}{c}{Q3.5-9b} & \multicolumn{2}{c}{G2-9b-it} & \multicolumn{2}{c}{Q3.5-9b} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
        r"\textbf{Method} & \textbf{eval} & ROC$_G$ & $\rho$ & ROC$_G$ & $\rho$ & ROC$_G$ & $\rho$ & ROC$_G$ & $\rho$ \\",
        r"\midrule",
    ]
    for label, num in ROWS:
        # which eval-TC rows to emit: only those with any data across the matrix
        for tc in ("self", "neg"):
            cells = []
            any_data = False
            for _, t in TASKS:
                for _, m in MODELS:
                    cg = get(cache[(t, m, "gen_roc")], num, PMI_COLS[tc])
                    cr = get(cache[(t, m, "spearman")], num, PMI_COLS[tc])
                    if cg or cr:
                        any_data = True
                    cells.append(fmt(cg, t, m, label, "ROC_G", tc))
                    cells.append(fmt(cr, t, m, label, "rho", tc))
            if not any_data:
                continue
            lines.append(f"{label} & {tc} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Generator results (ROC$_G$, $\rho$) on Hyponymy and IFEval, self-TC and "
              r"neg-TC eval shown as separate rows (cf.\ \texttt{tab:main-results-multi}, which "
              r"reports the per-task best). All rows are \textbf{epoch 2}. "
              r"$^{\dagger}$ = incomplete (see notes). \texttt{---} = setting not run for that model "
              r"(e.g.\ Consistency FT was not trained for Qwen). \textbf{NB:} the Qwen SFT and "
              r"RankAlign cells were trained \emph{with} \texttt{--validator-log-odds} (vlo); the gemma "
              r"ones were not (effect negligible when $P(\mathrm{Yes}){+}P(\mathrm{No})\approx1$). "
              r"disc-shots: membership=few, ifeval=zero.}",
              r"\label{tab:qwen-rerun-main}", r"\end{table*}"]
    return "\n".join(lines)


def val_table() -> str:
    cache = {(t, m, met): load(t, m, met)
             for _, t in TASKS for _, m in MODELS for met in ("val_roc", "val_acc")}
    lines = [
        r"\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l cccc cccc}",
        r"\toprule",
        r"& \multicolumn{4}{c}{\textbf{Hyponymy}} & \multicolumn{4}{c}{\textbf{IFEval}} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r"& \multicolumn{2}{c}{G2-9b-it} & \multicolumn{2}{c}{Q3.5-9b} & \multicolumn{2}{c}{G2-9b-it} & \multicolumn{2}{c}{Q3.5-9b} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"\textbf{Method} & ROC$_V$ & Acc$_V$ & ROC$_V$ & Acc$_V$ & ROC$_V$ & Acc$_V$ & ROC$_V$ & Acc$_V$ \\",
        r"\midrule",
    ]
    # validator metrics are eval-TC independent: take any non-missing column
    ANY = ["PMI self", "PMI base", "Neg self", "Neg base", "Raw"]
    for label, num in ROWS:
        cells = []
        for _, t in TASKS:
            for _, m in MODELS:
                cv = get(cache[(t, m, "val_roc")], num, ANY)
                ca = get(cache[(t, m, "val_acc")], num, ANY)
                cells.append(fmt(cv, t, m, label, "ROC_V", ""))
                cells.append(fmt(ca, t, m, label, "Acc_V", ""))
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Validator results (ROC$_V$, Acc$_V$) on Hyponymy and IFEval. These are "
              r"independent of the eval-time typicality correction (validator log-odds vs.\ gold "
              r"label), so there is no self/neg split. All rows are \textbf{epoch 2}.}",
              r"\label{tab:qwen-rerun-val}", r"\end{table*}"]
    return "\n".join(lines)


def main():
    m = main_table()
    v = val_table()
    notes = "\n".join(f"%   - {x}" for x in incomplete_notes) or "%   (none)"
    header = (
        "% Auto-generated by _build_paper_tables_latex.py. Requires \\stdv{} macro\n"
        "% (\\newcommand{\\stdv}[1]{{\\tiny$\\pm$#1}}). disc-shots: membership=few, ifeval=zero.\n"
        "% EPOCH NOTE: every cell here is epoch 2. In the paper's tab:main-results-multi /\n"
        "% -val, the RankAlign row is reported at EPOCH 1 (per project convention), so the\n"
        "% RankAlign rows below are NOT epoch-matched to the paper. Other methods: epoch 2.\n"
        "% Incomplete cells ($^\\dagger$):\n" + notes + "\n"
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(header + "\n" + m + "\n\n" + v + "\n")
    print(f"wrote {OUT}")
    print("\n--- incomplete cells ---")
    print("\n".join(incomplete_notes) or "(none)")


if __name__ == "__main__":
    main()
