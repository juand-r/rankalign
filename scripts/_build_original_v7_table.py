#!/usr/bin/env python3
"""Build "original v7 results" LaTeX tables (NOT the wandb rerun).

These reconstruct what was actually on disk from the ORIGINAL v7 evals — the
canonical `delta-bins 10 · eval_model_sN` runs that the paper numbers came from —
showing ALL FOUR eval-time typicality variants as separate rows per method:

    self (own)  = PMI self  (`self-`     scores; model's own self-typicality)
    self-base   = PMI base  (`basetyp-`  scores; base model's self-typicality)
    neg  (own)  = Neg self  (`neg-`      scores; model's own neg-typicality)
    neg-base    = Neg base  (`basetypneg-` scores; base model's neg-typicality)

so we can see, per cell, which variant a paper number was taken from.

Sources (all on disk, original — no rerun):
  1. pod-results-{genroc,spearman,valroc,valacc}-20260525.md — the canonical
     `eval_model_sN` markdown tables for BOTH models × {rosch, ifeval OOD}, all
     four eval-TC columns. Predominantly base-typ for trained cells (+ some own).
  2. metrics-from-scores/{rosch_9b-it,ifeval_v7_ood_9b-it}_gen_roc_table_cells.csv
     — the deduped machine-readable gemma gen_roc cells. rosch carries the OWN-typ
     (PMI self / Neg self) values the paper Hyponymy column was taken from
     (e.g. FLORA-PMI = 92.56 ≈ paper 92.5). Overlaid as authoritative for gen_roc.

Output: docs/original_v7_results.tex  (Table 1 = ROC_G + rho with the four eval-TC
rows; Table 2 = ROC_V + Acc_V, eval-TC independent so one row per method).

Usage: python scripts/_build_original_v7_table.py [OUT.tex]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
METRICS = REPO / "metrics-from-scores"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else DOCS / "original_v7_results.tex"

# metric key -> (pod-results doc suffix)
METRIC_DOC = {
    "gen_roc": "genroc",
    "rho": "spearman",
    "val_roc": "valroc",
    "val_acc": "valacc",
}
MODEL_MAP = {"gemma-2-9b-it": "9b-it", "Qwen3.5-9B": "qwen"}
# only these tasks feed the paper-comparable tables (ifeval = OOD, like the paper)
TASK_MAP = {"rosch": "rosch", "ifeval OOD": "ifeval"}

# pod-results table columns, in order, after the leading "Setting" + "Train" cells:
#   Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self)
POD_COLS = ["raw", "pmi_base", "pmi_self", "neg_base", "neg_self"]

# eval-TC variant rows to emit (key -> (display label, dict field))
VARIANTS = [
    ("self (own)", "pmi_self"),
    ("self-base", "pmi_base"),
    ("neg (own)", "neg_self"),
    ("neg-base", "neg_base"),
]

# (paper label, method_num)
ROWS = [("Base", 0), ("SFT", 1), ("RankAlign", 2),
        ("Consistency FT", 13), ("FLORA-PMI", 4), ("FLORA-Neg", 7)]

# data[(metric, model_key, task_key, num)] = {field: (mean, se) or None}
data: dict[tuple, dict] = {}


def _parse_val(cell: str):
    """'83.6 ± 1.7' -> (83.6, 1.7); '90.1' -> (90.1, None); '--'/'N/A' -> None."""
    s = cell.strip()
    if s in ("--", "N/A", "—", ""):
        return None
    m = re.match(r"^(-?\d+\.?\d*)\s*(?:±|\\pm|\+/-)\s*(-?\d+\.?\d*)$", s)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    m = re.match(r"^(-?\d+\.?\d*)$", s)
    if m:
        return (float(m.group(1)), None)
    return None


_HDR = re.compile(
    r"^## (gemma-2-9b-it|Qwen3\.5-9B) × (rosch|ifeval OOD|ifeval ID|persona ID|persona OOD)"
    r" — delta-bins 10 · eval_model_sN \(canonical\)"
)


def parse_pod_doc(metric: str) -> None:
    doc = DOCS / f"pod-results-{METRIC_DOC[metric]}-20260525.md"
    lines = doc.read_text().splitlines()
    i = 0
    while i < len(lines):
        m = _HDR.match(lines[i])
        if not m:
            i += 1
            continue
        model_disp, task_disp = m.group(1), m.group(2)
        i += 1
        if model_disp not in MODEL_MAP or task_disp not in TASK_MAP:
            continue  # skip persona / ID for the paper tables
        model_key, task_key = MODEL_MAP[model_disp], TASK_MAP[task_disp]
        # advance to the table body (rows starting with '| <int> ')
        while i < len(lines) and not re.match(r"^\|\s*\d+\s", lines[i]):
            if lines[i].startswith("## "):
                break
            i += 1
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            i += 1
            if len(cells) < 7:
                continue
            mnum = re.match(r"^(\d+)\s", cells[0])
            if not mnum:
                continue
            num = int(mnum.group(1))
            vals = {field: _parse_val(cells[2 + j]) for j, field in enumerate(POD_COLS)}
            data[(metric, model_key, task_key, num)] = vals


def overlay_gemma_gen_roc_cells() -> None:
    """Overlay the deduped gemma gen_roc cells (authoritative). rosch carries OWN-typ
    (PMI self/Neg self) = the paper Hyponymy values; ifeval-ood carries base."""
    col_to_field = {"PMI self": "pmi_self", "PMI base": "pmi_base",
                    "Neg self": "neg_self", "Neg base": "neg_base", "Raw": "raw"}
    for task_key, fname in [("rosch", "rosch_9b-it_gen_roc_table_cells.csv"),
                            ("ifeval", "ifeval_v7_ood_9b-it_gen_roc_table_cells.csv")]:
        path = METRICS / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for num in df["method_num"].unique():
            sub = df[df["method_num"] == num]
            key = ("gen_roc", "9b-it", task_key, int(num))
            cur = data.setdefault(key, {f: None for f in POD_COLS})
            for _, r in sub.iterrows():
                field = col_to_field.get(r["column"])
                if field and pd.notna(r["mean"]):
                    se = float(r["se"]) if pd.notna(r["se"]) else None
                    cur[field] = (float(r["mean"]), se)


def get(metric: str, model_key: str, task_key: str, num: int, field: str):
    return data.get((metric, model_key, task_key, num), {}).get(field)


def fmt(v) -> str:
    if v is None:
        return "---"
    mean, se = v
    return f"{mean:.1f}\\stdv{{{se:.1f}}}" if se is not None else f"{mean:.1f}"


MODELS = [("G2-9b-it", "9b-it"), ("Q3.5-9b", "qwen")]
TASKS = [("Hyponymy", "rosch"), ("IFEval", "ifeval")]


def main_table() -> str:
    lines = [
        r"\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{ll cccc cccc}",
        r"\toprule",
        r"& & \multicolumn{4}{c}{\textbf{Hyponymy}} & \multicolumn{4}{c}{\textbf{IFEval (OOD)}} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
        r"& & \multicolumn{2}{c}{G2-9b-it} & \multicolumn{2}{c}{Q3.5-9b} & \multicolumn{2}{c}{G2-9b-it} & \multicolumn{2}{c}{Q3.5-9b} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
        r"\textbf{Method} & \textbf{eval-TC} & ROC$_G$ & $\rho$ & ROC$_G$ & $\rho$ & ROC$_G$ & $\rho$ & ROC$_G$ & $\rho$ \\",
        r"\midrule",
    ]
    for label, num in ROWS:
        emitted = []
        for vlabel, field in VARIANTS:
            cells, any_data = [], False
            for _, task_key in TASKS:
                for _, model_key in MODELS:
                    cg = get("gen_roc", model_key, task_key, num, field)
                    cr = get("rho", model_key, task_key, num, field)
                    if cg is not None or cr is not None:
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
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")
    lines += [
        r"\end{tabular}",
        r"\caption{\textbf{Original v7 results} (NOT the wandb rerun): generator metrics "
        r"ROC$_G$ and Spearman $\rho$ on Hyponymy and IFEval (OOD), reconstructed from the "
        r"canonical \texttt{delta-bins 10 $\cdot$ eval\_model\_sN} evals on disk "
        r"(\texttt{pod-results-*-20260525.md} + the deduped gemma \texttt{gen\_roc} cells). "
        r"All four eval-time typicality corrections are shown as separate rows: "
        r"\emph{self/neg (own)} = the model's own typicality (\texttt{self-}/\texttt{neg-}); "
        r"\emph{self/neg-base} = the base model's typicality (\texttt{basetyp-}/\texttt{basetypneg-}). "
        r"\texttt{---} = that variant was not computed on disk for that cell. "
        r"Hyponymy paper numbers were taken from the \emph{own} rows; IFEval from the "
        r"\emph{base} rows. All cells epoch 2 (paper RankAlign is reported at epoch 1).}",
        r"\label{tab:original-v7-main}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def val_table() -> str:
    lines = [
        r"\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l cccc cccc}",
        r"\toprule",
        r"& \multicolumn{4}{c}{\textbf{Hyponymy}} & \multicolumn{4}{c}{\textbf{IFEval (OOD)}} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r"& \multicolumn{2}{c}{G2-9b-it} & \multicolumn{2}{c}{Q3.5-9b} & \multicolumn{2}{c}{G2-9b-it} & \multicolumn{2}{c}{Q3.5-9b} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"\textbf{Method} & ROC$_V$ & Acc$_V$ & ROC$_V$ & Acc$_V$ & ROC$_V$ & Acc$_V$ & ROC$_V$ & Acc$_V$ \\",
        r"\midrule",
    ]
    # validator metrics are eval-TC independent: take any populated field
    ANY = ["pmi_base", "pmi_self", "neg_base", "neg_self", "raw"]

    def any_val(metric, model_key, task_key, num):
        for f in ANY:
            v = get(metric, model_key, task_key, num, f)
            if v is not None:
                return v
        return None

    for label, num in ROWS:
        cells = []
        for _, task_key in TASKS:
            for _, model_key in MODELS:
                cells.append(fmt(any_val("val_roc", model_key, task_key, num)))
                cells.append(fmt(any_val("val_acc", model_key, task_key, num)))
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{\textbf{Original v7 results}: validator metrics ROC$_V$, Acc$_V$ on "
        r"Hyponymy and IFEval (OOD). These are independent of the eval-time typicality "
        r"correction, so there is no self/neg split. Source as in Table~\ref{tab:original-v7-main}. "
        r"All cells epoch 2.}",
        r"\label{tab:original-v7-val}", r"\end{table*}",
    ]
    return "\n".join(lines)


def main() -> None:
    for metric in METRIC_DOC:
        parse_pod_doc(metric)
    overlay_gemma_gen_roc_cells()
    header = (
        "% ORIGINAL v7 results (NOT the wandb rerun).\n"
        "% Auto-generated by scripts/_build_original_v7_table.py.\n"
        "% Requires \\stdv{} (\\newcommand{\\stdv}[1]{{\\tiny$\\pm$#1}}) and \\multirow.\n"
        "% Source: pod-results-{genroc,spearman,valroc,valacc}-20260525.md (canonical\n"
        "% eval_model_sN) + metrics-from-scores gemma gen_roc cells (own-typ overlay).\n"
        "% Four eval-TC rows per method: self/neg (own) vs self/neg-base.\n"
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(header + "\n" + main_table() + "\n\n" + val_table() + "\n")
    print(f"wrote {OUT}")
    # quick coverage summary to stdout
    print("\n--- coverage (non-empty gen_roc cells) ---")
    for label, num in ROWS:
        for vlabel, field in VARIANTS:
            got = []
            for tname, task_key in TASKS:
                for mname, model_key in MODELS:
                    if get("gen_roc", model_key, task_key, num, field) is not None:
                        got.append(f"{mname}/{tname}")
            if got:
                print(f"  {label:16s} {vlabel:11s}: {', '.join(got)}")


if __name__ == "__main__":
    main()
