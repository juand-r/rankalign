#!/usr/bin/env python3
"""Build the "original v7 results" tables — FULL v7 accounting, v7 delta-bins ONLY.

Shows all four eval-time typicality variants as separate rows per method
(self/neg own vs self/neg base), for Hyponymy + IFEval, gemma-2-9b-it + Qwen3.5-9B.

v7 ONLY — excludes v6 and v7b (fixed delta-0.15), and excludes the June wandb rerun.
Two provenance sources (the v7 evals were run in two places):
  [L] local-mll : docs/v7_rosch_all_metrics_20260525_003506.md  (gemma rosch; the
                  complete 4-variant x 5-metric table, matches the paper)
  [P] pod-v7    : metrics-from-scores/v7_recompute_eval_metrics.csv, recomputed from
                  the raw `eval_model_sN` scores in outputs_gemma4_from_pod-v7/
                  (gemma ifeval, qwen ifeval, qwen rosch).

IFEval split: own-typ exists for gemma only on the ID set (own-OOD was never run in
v7); base-typ exists OOD. Each ifeval cell is annotated with its split (o=OOD / i=ID).
Paper tables are OOD.

Output: docs/original_v7_results.tex  (+ standalone wrapper -> .pdf)
Usage:  python scripts/_build_original_v7_table.py [OUT.tex]
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
ROSCH_DOC = DOCS / "v7_rosch_all_metrics_20260525_003506.md"
RECOMPUTE = REPO / "metrics-from-scores" / "v7_recompute_eval_metrics.csv"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else DOCS / "original_v7_results.tex"

# variant key -> (display label, doc column name, recompute column name)
VARIANTS = [
    ("self (own)", "PMI self", "PMI self"),
    ("self-base", "PMI base", "PMI base"),
    ("neg (own)", "Neg self", "Neg self"),
    ("neg-base", "Neg base", "Neg base"),
]
METRIC_DOC = {"gen_roc": "gen_roc", "rho": "spearman", "val_roc": "val_roc", "val_acc": "val_acc"}
ROWS = [("Base", 0), ("SFT", 1), ("RankAlign", 2),
        ("Consistency FT", 13), ("FLORA-PMI", 4), ("FLORA-Neg", 7)]

# data[(task, model, num, metric, doccol)] = (mean, se, n, split, prov)
data: dict[tuple, tuple] = {}


def add(task, model, num, metric, col, mean, se, n, split, prov):
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return
    data[(task, model, num, metric, col)] = (float(mean), (None if se is None or math.isnan(se) else float(se)),
                                             int(n) if n else 0, split, prov)


# ---- source L: gemma rosch from v7_rosch_all_metrics ----------------------
def parse_rosch_doc():
    if not ROSCH_DOC.exists():
        print(f"WARN missing {ROSCH_DOC}", file=sys.stderr)
        return
    lines = ROSCH_DOC.read_text().splitlines()
    metric = None
    in9b = False
    cols = None  # ordered list of column names from the header row
    for ln in lines:
        mh = re.match(r"^## metric: (\w+)", ln)
        if mh:
            metric = mh.group(1)
            in9b = False
            continue
        if ln.startswith("### "):
            in9b = ln.strip() == "### gemma-2-9b-it"
            continue
        if not in9b or metric not in METRIC_DOC.values():
            continue
        if ln.startswith("| Method"):
            cols = [c.strip() for c in ln.strip().strip("|").split("|")][1:]  # drop 'Method'
            continue
        m = re.match(r"^\|\s*(\d+)\s+.*", ln)
        if not m or cols is None:
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        num = int(re.match(r"(\d+)", cells[0]).group(1))
        metric_key = next(k for k, v in METRIC_DOC.items() if v == metric)
        for ci, colname in enumerate(cols):
            cell = cells[1 + ci] if 1 + ci < len(cells) else ""
            vm = re.match(r"^(-?\d+\.?\d*)\s*±\s*(-?\d+\.?\d*)$", cell.strip())
            if vm:
                add("rosch", "9b-it", num, metric_key, colname,
                    float(vm.group(1)), float(vm.group(2)), 10, "all", "L")


# ---- source P: recompute CSV (gemma ifeval, qwen ifeval, qwen rosch) -------
def parse_recompute():
    if not RECOMPUTE.exists():
        print(f"WARN missing {RECOMPUTE}", file=sys.stderr)
        return
    df = pd.read_csv(RECOMPUTE)
    metric_map = {"gen_roc": "gen_roc", "spearman": "rho", "val_roc": "val_roc", "val_acc": "val_acc"}
    # for ifeval prefer OOD; fall back to ID. group rows and pick.
    for (model, task, num, col, metric), sub in df.groupby(["model", "task", "setting", "column", "metric"]):
        mk = metric_map.get(metric)
        if mk is None or col == "val":
            continue
        if task == "ifeval":
            ood = sub[sub.split == "ood"]
            pick = ood.iloc[0] if len(ood) else sub.iloc[0]
        else:
            pick = sub.iloc[0]
        add(task, model, int(num), mk, col, pick["mean"], pick["se"], pick["n"], pick["split"], "P")
    # validator metrics (column 'val'): apply to all variants for that cell
    for (model, task, num, metric), sub in df[df.column == "val"].groupby(["model", "task", "setting", "metric"]):
        mk = metric_map.get(metric)
        if mk not in ("val_roc", "val_acc"):
            continue
        if task == "ifeval":
            ood = sub[sub.split == "ood"]
            pick = ood.iloc[0] if len(ood) else sub.iloc[0]
        else:
            pick = sub.iloc[0]
        for _, _, col in VARIANTS:
            add(task, model, int(num), mk, col, pick["mean"], pick["se"], pick["n"], pick["split"], "P")
        add(task, model, int(num), mk, "PMI self", pick["mean"], pick["se"], pick["n"], pick["split"], "P")


def get(task, model, num, metric, col):
    return data.get((task, model, num, metric, col))


SPLIT_MARK = {"ood": "o", "id": "i", "all": ""}


def fmt(cell):
    if cell is None:
        return "---"
    mean, se, n, split, prov = cell
    s = f"{mean:.1f}\\stdv{{{se:.1f}}}" if se is not None else f"{mean:.1f}"
    sm = SPLIT_MARK.get(split, "")
    if sm:
        s += f"$_{{{sm}}}$"
    return s


MODELS = [("G2-9b-it", "9b-it"), ("Q3.5-9b", "qwen")]
TASKS = [("Hyponymy", "rosch"), ("IFEval", "ifeval")]


def main_table():
    lines = [
        r"\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{ll cccc cccc}",
        r"\toprule",
        r"& & \multicolumn{4}{c}{\textbf{Hyponymy}} & \multicolumn{4}{c}{\textbf{IFEval}} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
        r"& & \multicolumn{2}{c}{G2-9b-it} & \multicolumn{2}{c}{Q3.5-9b} & \multicolumn{2}{c}{G2-9b-it} & \multicolumn{2}{c}{Q3.5-9b} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
        r"\textbf{Method} & \textbf{eval-TC} & ROC$_G$ & $\rho$ & ROC$_G$ & $\rho$ & ROC$_G$ & $\rho$ & ROC$_G$ & $\rho$ \\",
        r"\midrule",
    ]
    for label, num in ROWS:
        emitted = []
        for vlabel, doccol, _ in VARIANTS:
            cells, any_data = [], False
            for _, task in [("Hyponymy", "rosch"), ("IFEval", "ifeval")]:
                for _, model in [("G2-9b-it", "9b-it"), ("Q3.5-9b", "qwen")]:
                    cg = get(task, model, num, "gen_roc", doccol)
                    cr = get(task, model, num, "rho", doccol)
                    if cg or cr:
                        any_data = True
                    cells += [fmt(cg), fmt(cr)]
            if any_data:
                emitted.append((vlabel, cells))
        if not emitted:
            continue
        n = len(emitted)
        for k, (vlabel, cells) in enumerate(emitted):
            head = f"\\multirow{{{n}}}{{*}}{{{label}}}" if k == 0 else ""
            lines.append(f"{head} & {vlabel} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}",
              r"\caption{\textbf{Original v7 results} (v7 delta-bins only; \emph{not} v6, v7b, "
              r"or the June rerun): ROC$_G$ and $\rho$, all four eval-time typicality corrections "
              r"as separate rows. Sources: \textbf{Hyponymy/G2-9b-it} from the complete local-mll "
              r"table \texttt{v7\_rosch\_all\_metrics}; everything else recomputed from the raw "
              r"\texttt{eval\_model\_sN} pod scores. IFEval split per cell: "
              r"$_o$=OOD (20 prompts), $_i$=ID (79). \textbf{Own-typ ifeval exists only on ID} "
              r"(own-OOD was never run in v7); base-typ ifeval is OOD. \texttt{---}=not on disk in v7. "
              r"All epoch 2.}",
              r"\label{tab:original-v7-main}", r"\end{table*}"]
    return "\n".join(lines)


def val_table():
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
    ANY = ["PMI self", "PMI base", "Neg self", "Neg base"]

    def anyget(task, model, num, metric):
        for c in ANY:
            v = get(task, model, num, metric, c)
            if v:
                return v
        return None

    for label, num in ROWS:
        cells = []
        for _, task in [("Hyponymy", "rosch"), ("IFEval", "ifeval")]:
            for _, model in [("G2-9b-it", "9b-it"), ("Q3.5-9b", "qwen")]:
                cells.append(fmt(anyget(task, model, num, "val_roc")))
                cells.append(fmt(anyget(task, model, num, "val_acc")))
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{Original v7 results}: validator metrics ROC$_V$, Acc$_V$ "
              r"(eval-TC independent, one row/method). Sources/splits as in "
              r"Table~\ref{tab:original-v7-main}. All epoch 2.}",
              r"\label{tab:original-v7-val}", r"\end{table*}"]
    return "\n".join(lines)


def main():
    parse_rosch_doc()
    parse_recompute()
    header = (
        "% ORIGINAL v7 results (v7 delta-bins ONLY; not v6/v7b/rerun). FULL accounting.\n"
        "% Auto-generated by scripts/_build_original_v7_table.py.\n"
        "% Needs \\stdv{} + \\multirow. Provenance: Hyp/G2-9b-it=local-mll (v7_rosch_all_metrics);\n"
        "% ifeval + qwen = pod-v7 recompute (v7_recompute_eval_metrics.csv). ifeval split: o=OOD,i=ID.\n"
    )
    OUT.write_text(header + "\n" + main_table() + "\n\n" + val_table() + "\n")
    print(f"wrote {OUT}")
    # coverage report
    filled = sum(1 for _ in data)
    print(f"populated (task,model,method,metric,variant) entries: {filled}")
    print("\n--- gen_roc cells per (task,model) ---")
    for _, task in TASKS:
        for _, model in MODELS:
            got = [(num, c) for (t, m, num, met, c) in data
                   if t == task and m == model and met == "gen_roc"]
            print(f"  {task}/{model}: {len(got)} cells")


if __name__ == "__main__":
    main()
