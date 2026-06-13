#!/usr/bin/env python3
"""Side-by-side PDF: qwen IFEval-OOD, the current rerun vs an independent retrain (v2).

current = metrics-from-scores-rerun-only   (the numbers in docs/rerun_only_tables.pdf)
v2      = metrics-from-scores-rerun-v2      (jobs 44680-44684; identical flags+venv; new run)

Tests train-to-train variance of the mll rerun pipeline (NOT a comparison to the pod/paper).
Emits a complete standalone LaTeX doc (Generator + Validator tables) with cur / v2 / Δ and
bolds |Δ|>5. Compile with pdflatex.

Usage: python scripts/_build_v2_vs_current_table.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CUR = REPO / "metrics-from-scores-rerun-only"
V2 = REPO / "metrics-from-scores-rerun-v2"
OUT = REPO / "docs" / "v2_vs_current_qwen_ifeval.tex"

ROWS = [
    (1, "SFT"),
    (2, "RankAlign"),
    (3, "New+fsx"),
    (4, "FLORA-PMI"),
    (7, "FLORA-Neg"),
]
# Full eval-TC column set (own-typ self-/neg- + base-typ basetyp-/basetypneg- + Raw), present
# in BOTH cur and v2 after the own-typ eval pass (_qwen_ifeval_v2_owntyp_eval.sh). Validator is
# read with a fixed column-preference for BOTH sides, so the delta is apples-to-apples.
GEN_COLS = ["Raw", "PMI self", "PMI base", "Neg self", "Neg base"]
ANY = ["PMI base", "Neg base", "PMI self", "Neg self", "Raw"]


def _load(d: Path, met: str) -> pd.DataFrame:
    return pd.read_csv(d / f"ifeval_v7_ood_qwen3.5-9b_{met}_table_cells.csv")


def _val(df: pd.DataFrame, num: int, col: str) -> float | None:
    r = df[(df["method_num"] == num) & (df["column"] == col)]
    if len(r) and pd.notna(r.iloc[0]["mean"]):
        return float(r.iloc[0]["mean"])
    return None


def _val_any(df: pd.DataFrame, num: int) -> float | None:
    for c in ANY:
        v = _val(df, num, c)
        if v is not None:
            return v
    return None


def _cell(cur: float | None, v2: float | None) -> str:
    """Return 'cur & v2 & Δ' LaTeX, bolding Δ when |Δ|>5; '---' where absent."""
    if cur is None and v2 is None:
        return "--- & --- & ---"
    cs = f"{cur:.1f}" if cur is not None else "---"
    vs = f"{v2:.1f}" if v2 is not None else "---"
    if cur is None or v2 is None:
        return f"{cs} & {vs} & ---"
    d = v2 - cur
    ds = f"{d:+.1f}"
    if abs(d) > 5:
        ds = rf"\textbf{{{ds}}}"
    return f"{cs} & {vs} & {ds}"


def gen_table() -> str:
    g_cur, g_v2 = _load(CUR, "gen_roc"), _load(V2, "gen_roc")
    r_cur, r_v2 = _load(CUR, "spearman"), _load(V2, "spearman")
    lines = [
        r"\begin{center}\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{ll ccc ccc}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{\textbf{ROC}$_G$} & \multicolumn{3}{c}{\textbf{$\rho$}} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"\textbf{Method} & \textbf{eval} & cur & v2 & $\Delta$ & cur & v2 & $\Delta$ \\",
        r"\midrule",
    ]
    first = True
    for num, label in ROWS:
        rows = []
        for col in GEN_COLS:
            gc, gv = _val(g_cur, num, col), _val(g_v2, num, col)
            if gc is None and gv is None:
                continue
            rc, rv = _val(r_cur, num, col), _val(r_v2, num, col)
            rows.append(f"{label} & {col} & {_cell(gc, gv)} & {_cell(rc, rv)} " + r"\\")
        if not rows:
            continue
        if not first:
            lines.append(r"\midrule")
        first = False
        lines.extend(rows)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    return "\n".join(lines)


def val_table() -> str:
    vr_cur, vr_v2 = _load(CUR, "val_roc"), _load(V2, "val_roc")
    va_cur, va_v2 = _load(CUR, "val_acc"), _load(V2, "val_acc")
    lines = [
        r"\begin{center}\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l ccc ccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{\textbf{ROC}$_V$} & \multicolumn{3}{c}{\textbf{Acc}$_V$} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"\textbf{Method} & cur & v2 & $\Delta$ & cur & v2 & $\Delta$ \\",
        r"\midrule",
    ]
    for num, label in ROWS:
        lines.append(
            f"{label} & {_cell(_val_any(vr_cur, num), _val_any(vr_v2, num))} "
            f"& {_cell(_val_any(va_cur, num), _val_any(va_v2, num))} " + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    return "\n".join(lines)


def main() -> None:
    doc = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.7in]{{geometry}}
\usepackage{{booktabs,amsmath,amssymb}}
\pagestyle{{empty}}
\begin{{document}}
\section*{{qwen Qwen3.5-9B IFEval (OOD) --- reproducibility: current rerun vs.\ independent retrain (v2)}}
{{\footnotesize \textbf{{cur}} = the current wandb rerun (\texttt{{outputs-rerun-wandb}}, in
\texttt{{rerun\_only\_tables.pdf}}); \textbf{{v2}} = a second, independent retrain
(\texttt{{outputs-rerun-wandb-v2}}, jobs 44680--44684) with \emph{{identical}} flags, data,
trainer, 3 epochs, and the \texttt{{qwen35}} venv --- only the output dirs differ. No fixed
seed, so $\Delta$ is the natural train-to-train variance. $\Delta=$ v2$-$cur; \textbf{{bold}}
marks $|\Delta|>5$. Both baselines are mll reruns (this is \emph{{not}} a comparison to the
original pod/paper). All eval-time typicality variants (own: PMI self / Neg self; base: PMI base
/ Neg base; + Raw + validator) are present in both runs. Takeaway: SFT/New+fsx/FLORA reproduce
within $\sim\pm2.6$ (incl.\ the SFT validator); \textbf{{RankAlign does not}} (high run-to-run variance).}}

\subsection*{{Generator (ROC$_G$, $\rho$)}}
{gen_table()}

\subsection*{{Validator (ROC$_V$, Acc$_V$)}}
{val_table()}
\end{{document}}
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(doc)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
