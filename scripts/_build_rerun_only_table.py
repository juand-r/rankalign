#!/usr/bin/env python3
"""Build a RERUN-ONLY results table (a brand-new, self-contained PDF).

Every value comes from score files in ONE directory — the wandb-rerun batch
`outputs-rerun-wandb` — aggregated into `metrics-from-scores-rerun-only/` by
running `_build_ifeval_table_v7.py` / `_build_rosch_table_v7.py` with
`EXTRA_SEARCH_DIR=.../outputs-rerun-wandb` (so NO v6-base or trained rows leak
in from the old `outputs/` dir). Any cell with no rerun source is `---`.

This differs from `_build_paper_tables_latex.py` (which searches BOTH
`outputs-rerun-wandb` and `outputs`, silently mixing new rerun numbers with old
ones). Here the input dir has a single fallback-free source, so the table is
honestly all-rerun.

The output is a COMPLETE standalone LaTeX document (defines its own \\stdv macro,
includes a provenance appendix listing, per column, the scores dir + the exact
checkpoint dirs each cell came from). Compile with `pdflatex`.

Usage:
    python scripts/_build_rerun_only_table.py [METRICS_DIR] [OUT.tex]
    (defaults: metrics-from-scores-rerun-only/  docs/rerun_only_tables.tex)
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
MDIR = (
    Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "metrics-from-scores-rerun-only"
)
OUT = (
    Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "docs" / "rerun_only_tables.tex"
)
SCORES_DIR = "outputs-rerun-wandb"  # the single source dir these cells were built from

MODELS = [("G2-9b-it", "9b-it"), ("Q3.5-9b", "qwen3.5-9b")]
TASKS = [("Hyponymy", "rosch"), ("IFEval", "ifeval")]
SPLIT = "ood"  # ifeval split shown (paper convention); rosch has no split
# (paper label, method_num). qwen has no s13 (Consistency FT) -> "---".
ROWS = [
    ("Base", 0),
    ("SFT", 1),
    ("RankAlign", 2),
    ("Consistency FT", 13),
    ("FLORA-PMI", 4),
    ("FLORA-Neg", 7),
]
# Show every eval-time typicality-correction variant present, each as its own row:
# own-typ (PMI self / Neg self) AND base-typ (PMI base / Neg base). A row is emitted
# only where the rerun batch actually has that variant for that setting.
EVAL_VARIANTS = ["PMI self", "PMI base", "Neg self", "Neg base"]

incomplete_notes: list[str] = []


def _safe_read(path: Path) -> pd.DataFrame | None:
    """Read a CSV, treating an empty file (a column with no rerun data) as None."""
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def load(task: str, model: str, metric: str) -> pd.DataFrame | None:
    """Load the single rerun-only cells CSV for (task, model, metric). No fallback."""
    want = "_id_" if SPLIT == "id" else "_ood_"
    cands = [
        c
        for c in MDIR.glob(f"{task}_v7_*{metric}_table_cells.csv")
        if f"_{model}_" in c.name
    ]
    if task == "ifeval":
        cands = [c for c in cands if want in c.name]
    if cands:
        return _safe_read(sorted(cands)[-1])
    return None


def get(df: pd.DataFrame | None, num: int, columns: list[str]) -> dict | None:
    if df is None:
        return None
    sub = df[df["method_num"] == num]
    for c in columns:
        r = sub[sub["column"] == c]
        if len(r) and pd.notna(r.iloc[0]["mean"]):
            return dict(
                mean=float(r.iloc[0]["mean"]),
                se=float(r.iloc[0]["se"]),
                n=int(r.iloc[0]["n"]),
                note=str(r.iloc[0]["note"]),
            )
    return None


def fmt(
    c: dict | None,
    task: str,
    model: str,
    label: str,
    metric: str,
    tc: str,
    bold: bool = False,
) -> str:
    if c is None:
        return "---"
    num = rf"\textbf{{{c['mean']:.1f}}}" if bold else f"{c['mean']:.1f}"
    s = num + (f"\\stdv{{{c['se']:.1f}}}" if not math.isnan(c["se"]) else "")
    if c.get("note", "OK") != "OK":
        s += "$^{\\dagger}$"
        incomplete_notes.append(
            f"{label}{'/' + tc if tc else ''} {metric} {task}/{model}: {c['note']}"
        )
    return s


def _col_maxes(rows: list[list[dict | None]], ncols: int) -> list[float | None]:
    """Per-column max of the displayed (1-decimal-rounded) mean across all rows.
    Comparing on the rounded value keeps bolding visually consistent on ties."""
    maxes: list[float | None] = []
    for j in range(ncols):
        vals = [round(r[j]["mean"], 1) for r in rows if r[j] is not None]
        maxes.append(max(vals) if vals else None)
    return maxes


def _is_max(c: dict | None, colmax: float | None) -> bool:
    return c is not None and colmax is not None and round(c["mean"], 1) == colmax


def main_table() -> str:
    cache = {
        (t, m, met): load(t, m, met)
        for _, t in TASKS
        for _, m in MODELS
        for met in ("gen_roc", "spearman")
    }
    lines = [
        r"\begin{center}\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{ll cccc cccc}",
        r"\toprule",
        r"& & \multicolumn{4}{c}{\textbf{Hyponymy}} & \multicolumn{4}{c}{\textbf{IFEval (OOD)}} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
        r"& & \multicolumn{2}{c}{G2-9b-it} & \multicolumn{2}{c}{Q3.5-9b} & \multicolumn{2}{c}{G2-9b-it} & \multicolumn{2}{c}{Q3.5-9b} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
        r"\textbf{Method} & \textbf{eval} & ROC$_G$ & $\rho$ & ROC$_G$ & $\rho$ & ROC$_G$ & $\rho$ & ROC$_G$ & $\rho$ \\",
        r"\midrule",
    ]
    # Pass 1: collect every emitted row's cell dicts (8 cols: per task x model -> ROC_G, rho).
    coords = [
        (t, m, met)
        for _, t in TASKS
        for _, m in MODELS
        for met in ("gen_roc", "spearman")
    ]
    collected = []  # (label, variant, [c-or-None x8])
    for label, num in ROWS:
        for variant in EVAL_VARIANTS:
            cells = [get(cache[(t, m, met)], num, [variant]) for (t, m, met) in coords]
            if any(c is not None for c in cells):
                collected.append((label, variant, cells))
    colmax = _col_maxes([cs for _, _, cs in collected], len(coords))

    # Pass 2: render, bolding the per-column max.
    first = True
    prev_label = None
    for label, variant, cells in collected:
        if not first and label != prev_label:
            lines.append(r"\midrule")
        first = False
        prev_label = label
        rendered = [
            fmt(
                c,
                t,
                m,
                label,
                met.replace("gen_roc", "ROC_G").replace("spearman", "rho"),
                variant,
                bold=_is_max(c, colmax[j]),
            )
            for j, (c, (t, m, met)) in enumerate(zip(cells, coords))
        ]
        lines.append(f"{label} & {variant} & " + " & ".join(rendered) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    return "\n".join(lines)


def val_table() -> str:
    cache = {
        (t, m, met): load(t, m, met)
        for _, t in TASKS
        for _, m in MODELS
        for met in ("val_roc", "val_acc")
    }
    lines = [
        r"\begin{center}\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l cccc cccc}",
        r"\toprule",
        r"& \multicolumn{4}{c}{\textbf{Hyponymy}} & \multicolumn{4}{c}{\textbf{IFEval (OOD)}} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r"& \multicolumn{2}{c}{G2-9b-it} & \multicolumn{2}{c}{Q3.5-9b} & \multicolumn{2}{c}{G2-9b-it} & \multicolumn{2}{c}{Q3.5-9b} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"\textbf{Method} & ROC$_V$ & Acc$_V$ & ROC$_V$ & Acc$_V$ & ROC$_V$ & Acc$_V$ & ROC$_V$ & Acc$_V$ \\",
        r"\midrule",
    ]
    ANY = ["PMI self", "PMI base", "Neg self", "Neg base", "Raw"]
    coords = [
        (t, m, met)
        for _, t in TASKS
        for _, m in MODELS
        for met in ("val_roc", "val_acc")
    ]
    collected = []  # (label, [c-or-None x8])
    for label, num in ROWS:
        cells = [get(cache[(t, m, met)], num, ANY) for (t, m, met) in coords]
        if any(c is not None for c in cells):
            collected.append((label, cells))
    colmax = _col_maxes([cs for _, cs in collected], len(coords))

    for label, cells in collected:
        rendered = [
            fmt(
                c,
                t,
                m,
                label,
                met.replace("val_roc", "ROC_V").replace("val_acc", "Acc_V"),
                "",
                bold=_is_max(c, colmax[j]),
            )
            for j, (c, (t, m, met)) in enumerate(zip(cells, coords))
        ]
        lines.append(f"{label} & " + " & ".join(rendered) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    return "\n".join(lines)


def _ckpt_family(fn: str) -> str:
    """Strip prefix/date/per-task suffix from a score filename -> checkpoint dir name."""
    b = fn[len("scores_") :] if fn.startswith("scores_") else fn
    b = re.sub(r"^(self|neg|basetyp|basetypneg|raw)-", "", b)
    b = re.sub(r"_(ifeval-prompt_\d+|rosch[-_].*?_test|membership.*?_test).*", "", b)
    b = re.sub(r"_\d{8}\.csv.*", "", b)
    return b


def provenance() -> str:
    """Per (task, model) column: scores dir + the distinct checkpoints each used,
    read from the rerun-only *_gen_roc_table_long.csv provenance files."""
    lines = [
        r"\subsection*{Provenance --- every value above is from \texttt{"
        + SCORES_DIR
        + r"/}}",
        r"{\footnotesize Cells were aggregated by \texttt{\_build\_ifeval\_table\_v7.py} / "
        r"\texttt{\_build\_rosch\_table\_v7.py} run with "
        r"\texttt{EXTRA\_SEARCH\_DIR=.../"
        + SCORES_DIR
        + r"} (single source, no fallback), "
        r"then assembled by \texttt{\_build\_rerun\_only\_table.py}. "
        r"Checkpoint dirs per column (\texttt{v6-*}=base model, \texttt{v7-*}=trained):}",
        r"\begin{itemize}\footnotesize\setlength{\itemsep}{1pt}",
    ]
    for tlabel, t in TASKS:
        for mlabel, m in MODELS:
            want = "_ood_" if t == "ifeval" else ""
            cands = [
                c
                for c in MDIR.glob(f"{t}_v7_*gen_roc_table_long.csv")
                if f"_{m}_" in c.name and (want in c.name if t == "ifeval" else True)
            ]
            head = f"{tlabel} / {mlabel}"
            df = _safe_read(sorted(cands)[-1]) if cands else None
            if df is None or "file" not in df:
                lines.append(
                    r"\item \textbf{"
                    + head
                    + r"}: \emph{no rerun data} ($\rightarrow$ all \texttt{---})"
                )
                continue
            fams = sorted({_ckpt_family(f) for f in df["file"].dropna()})
            lines.append(
                r"\item \textbf{" + head + r"}: " + str(len(fams)) + r" checkpoint(s):"
            )
            lines.append(r"\begin{itemize}\tiny")
            for fam in fams:
                lines.append(r"\item \texttt{" + fam.replace("_", r"\_") + r"}")
            lines.append(r"\end{itemize}")
    lines.append(r"\end{itemize}")
    return "\n".join(lines)


def main() -> None:
    body_main = main_table()
    body_val = val_table()
    prov = provenance()
    notes = "\n".join(rf"\item {x}" for x in incomplete_notes)
    notes_block = (
        (
            r"\subsection*{Incomplete cells ($^{\dagger}$)}\begin{itemize}\footnotesize"
            + "\n"
            + notes
            + "\n"
            + r"\end{itemize}"
        )
        if incomplete_notes
        else ""
    )

    doc = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.6in,landscape]{{geometry}}
\usepackage{{booktabs,amsmath,amssymb,xcolor}}
\newcommand{{\stdv}}[1]{{{{\tiny$\pm$#1}}}}
\pagestyle{{empty}}
\begin{{document}}
\section*{{RankAlign --- RERUN-ONLY results (wandb-rerun checkpoints)}}
{{\footnotesize Every value is from the \textbf{{{SCORES_DIR}}} batch (mll-retrained-for-wandb
checkpoints, evaluated 2026-06-07/08/09 in the \texttt{{qwen35}} venv). Cells with no rerun
source are \texttt{{---}}. The gemma IFEval Base row is blank because no rerun base eval
exists for it; the gemma Hyponymy Base row is blank because no rerun base eval exists for
gemma-2-9b-it on rosch. All rows epoch 2;
RankAlign is \emph{{not}} epoch-matched to the paper. Each eval-time typicality-correction
variant is its own row: \textbf{{PMI self}}/\textbf{{Neg self}} use the model's \emph{{own}}
typicality, \textbf{{PMI base}}/\textbf{{Neg base}} use the \emph{{base}} model's. A variant
row appears only where the rerun batch has it (e.g.\ Base has own only; FLORA-PMI has no Neg,
FLORA-Neg no PMI). disc-shots: membership=few, ifeval=zero.}}

\subsection*{{Generator (ROC$_G$, $\rho$)}}
{body_main}

\subsection*{{Validator (ROC$_V$, Acc$_V$)}}
{body_val}

{prov}

{notes_block}
\end{{document}}
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(doc)
    print(f"wrote {OUT}")
    print("incomplete cells:", len(incomplete_notes))


if __name__ == "__main__":
    main()
