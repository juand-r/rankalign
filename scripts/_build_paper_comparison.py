#!/usr/bin/env python3
"""Colored comparison: my ORIGINAL-v7 on-disk numbers vs the paper tables in main.tex.

For every paper cell (the 4 task-model columns I have data for: IFEval & Hyponymy ×
{G2-9b-it, Q3.5-9b}; HumanEval/G4 is excluded), compare the paper value against my
original-v7 variants (self/neg own + self/neg base) and classify:

  BLUE   : |paper - best-matching variant| < 1pp  -> faithful (just rounding).
  A1     : best match is an OWN variant (self-/neg-, no base). The paper took the
           own-typ number; the base-typ value differs. We print the base value too.
  GREEN  : RankAlign only, residual 1-5pp -> the paper reports RankAlign at EPOCH 1
           but my numbers are EPOCH 2; advise evaluating the other epoch.
  RED    : > 5pp from every available variant and not otherwise explained -> a genuine
           discrepancy we cannot attribute (paper copy error / old qwen training env /
           retraining noise -- cases b/c/d, indistinguishable).
  GREY † : the variant the paper most likely used is NOT on disk (e.g. trained-IFEval
           OOD own-typ was never computed) -> cannot verify here.

Known caveat surfaced as a footnote (paper main.tex line ~942, author note):
  FLORA (PMI/Neg) IFEval numbers in the paper are PRE-bug-fix (except Persona/Rosch),
  so a large IFEval-FLORA gap is expected, not a copy error.

Source of "mine": scripts/_build_original_v7_table.py (pod-results canonical + gemma
gen_roc own overlay). Paper: main.tex tab:main-results-multi / -val.

Usage: python scripts/_build_paper_comparison.py [MAIN_TEX] [OUT.tex]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# reuse the original-v7 data machinery
import importlib.util

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("_ov7", REPO / "scripts" / "_build_original_v7_table.py")
ov7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ov7)

MAIN_TEX = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/home/jdr/Projects/paper-rankalignv2/llm-gv-gap-research/main.tex")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "docs" / "paper_vs_original_v7.tex"

# Build the original-v7 data dict
for metric in ov7.METRIC_DOC:
    ov7.parse_pod_doc(metric)
ov7.overlay_gemma_gen_roc_cells()

# ---- paper parsing -------------------------------------------------------
# Paper column order (per task-model pair, each ROC + 2nd-metric):
#   IFEval-G2, IFEval-Q3.5, HumanEval-G4(skip), Hyponymy-G2, Hyponymy-Q3.5
# Indices into the 10 value cells -> (task_key, model_key); we keep ROC at 2i, 2nd at 2i+1
PAPER_COLS = [("ifeval", "9b-it"), ("ifeval", "qwen"), (None, None),
              ("rosch", "9b-it"), ("rosch", "qwen")]
PAPER_METHOD_NUM = {
    "Base": 0, "SFT": 1, "Consistency FT": 13, "RankAlign": 2,
    "\\FCPA{}-PMI": 4, "\\FCPA{}-Neg": 7,
}


def _strip(cell: str):
    """'\\B{80.3}\\stdv{2.7}' -> 80.3 ; '78.2\\stdv{2.2}' -> 78.2 ; else None."""
    s = cell.strip()
    s = re.sub(r"\\B\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\stdv\{[^}]*\}", "", s)
    s = s.strip()
    m = re.match(r"^(-?\d+\.?\d*)$", s)
    return float(m.group(1)) if m else None


def parse_paper_table(start_label: str) -> dict:
    """Return {(task, model, num): (metric1, metric2)} for the 4 cols I keep."""
    lines = MAIN_TEX.read_text().splitlines()
    # find the \label{...} line (NOT a \ref), then walk back to its \begin{table*}
    tag = "\\label{" + start_label + "}"
    li = next(i for i, l in enumerate(lines) if tag in l)
    # scan upward to \begin{tabular}, then parse body rows until \bottomrule
    out: dict = {}
    # body is above the label; collect rows containing ' & ' and \\ between toprule..bottomrule
    j = li
    # find enclosing \begin{table*}
    while j > 0 and "\\begin{table*}" not in lines[j]:
        j -= 1
    for l in lines[j:li]:
        if "\\midrule" in l or "\\toprule" in l or "\\cmidrule" in l or "multicolumn" in l:
            continue
        if "&" not in l or "\\\\" not in l:
            continue
        body = l.split("\\\\")[0]
        cells = [c.strip() for c in body.split("&")]
        name = cells[0].strip()
        if name not in PAPER_METHOD_NUM:
            continue
        num = PAPER_METHOD_NUM[name]
        vals = cells[1:]
        for ci, (task, model) in enumerate(PAPER_COLS):
            if task is None:
                continue
            m1 = _strip(vals[2 * ci]) if 2 * ci < len(vals) else None
            m2 = _strip(vals[2 * ci + 1]) if 2 * ci + 1 < len(vals) else None
            out[(task, model, num)] = (m1, m2)
    return out


PAPER1 = parse_paper_table("tab:main-results-multi")     # ROC_G, rho
PAPER2 = parse_paper_table("tab:main-results-multi-val") # ROC_V, Acc_V

# ---- classification ------------------------------------------------------
OWN = {"pmi_self": "self", "neg_self": "neg"}
BASE = {"pmi_base": "self-base", "neg_base": "neg-base"}
ALLV = list(OWN) + list(BASE)


def variants(metric, model, task, num):
    """Return list of (field, label, mean) available on disk for this cell+metric."""
    out = []
    for f in ALLV + ["raw"]:
        v = ov7.get(metric, model, task, num, f)
        if v is not None:
            out.append((f, f, v[0]))
    return out


def classify(paper_val, metric, model, task, num):
    """Return (code, mine_mean, mine_field, base_mean, note).

    The paper used OWN-typ (self/neg) for BOTH Hyponymy and IFEval (author note,
    main.tex ~l.942: "Rosch: Self", "IFEval: PMI[self]", "self generally better than
    base ... reporting those"). So we compare paper against the best-matching OWN
    variant when it exists. If own is NOT on disk (trained IFEval-OOD), we cannot
    verify the paper number from base -> GREY (would need an own-typ re-eval).
    """
    if paper_val is None:
        return ("none", None, None, None, "")
    vs = variants(metric, model, task, num)
    # Validator metrics (ROC_V, Acc_V) are eval-TC INDEPENDENT (paper caption): no
    # own/base distinction -> compare directly against any available variant value.
    if metric in ("val_roc", "val_acc"):
        if not vs:
            return ("nodata", None, None, None, "no on-disk variant")
        best = min(vs, key=lambda t: abs(t[2] - paper_val))
        field, mean = best[0], best[2]
        diff = abs(mean - paper_val)
        if num == 2 and 1.0 <= diff <= 5.0:
            return ("green", mean, field, None, "epoch1 vs my epoch2")
        if diff < 1.0:
            return ("blue", mean, field, None, "")
        if diff <= 5.0:
            return ("amber", mean, field, None, f"{diff:.1f}pp off")
        return ("red", mean, field, None, f"{diff:.1f}pp off")
    own = [v for v in vs if v[0] in OWN]
    base = [v for v in vs if v[0] in BASE]
    base_mean = min((v[2] for v in base), key=lambda m: abs(m - paper_val), default=None)
    flora_ifeval = (task == "ifeval" and num in (4, 7))

    if own:
        best = min(own, key=lambda t: abs(t[2] - paper_val))
        field, mean = best[0], best[2]
        diff = abs(mean - paper_val)
        if num == 2 and 1.0 <= diff <= 5.0:                 # RankAlign epoch1-vs-2
            return ("green", mean, field, base_mean, "epoch1 vs my epoch2")
        if diff < 1.0:
            if base_mean is not None and abs(base_mean - paper_val) >= 1.0:
                return ("a1", mean, field, base_mean, "paper=own")  # own matches, base differs
            return ("blue", mean, field, base_mean, "own match")
        if diff <= 5.0:
            return ("amber", mean, field, base_mean, f"own {diff:.1f}pp off")
        return ("red", mean, field, base_mean, f"own {diff:.1f}pp off")

    # no OWN on disk: paper number (own) cannot be verified here
    if base:
        note = "own not on disk"
        if base_mean is not None and abs(base_mean - paper_val) < 1.0:
            note += f"; base≈paper ({base_mean:.1f})"
        else:
            note += f"; base={base_mean:.1f}" if base_mean is not None else ""
        if flora_ifeval:
            note += "; paper pre-fix"
        return ("nodata", base_mean, (base[0][0] if base else None), base_mean, note)
    return ("nodata", None, None, None, "no on-disk variant")


COLOR = {"blue": "blue", "a1": "RoyalPurple", "green": "ForestGreen",
         "red": "red", "amber": "orange", "flora": "Gray", "nodata": "Gray"}

# ---- emit ----------------------------------------------------------------
MODELS = [("G2-9b-it", "9b-it"), ("Q3.5-9b", "qwen")]
TASKS = [("IFEval", "ifeval"), ("Hyponymy", "rosch")]
ROWS = [("Base", 0), ("SFT", 1), ("Consistency FT", 13),
        ("RankAlign", 2), ("FLORA-PMI", 4), ("FLORA-Neg", 7)]
flags: set[str] = set()


def cell_tex(paper_val, metric, model, task, num):
    code, mean, field, base_mean, note = classify(paper_val, metric, model, task, num)
    if code == "none":
        return "---"
    if paper_val is None:
        return "---"
    flags.add(code)
    pv = f"{paper_val:.1f}"
    fl = {"pmi_self": "s", "neg_self": "n", "pmi_base": "sb",
          "neg_base": "nb", "raw": "r"}.get(field, "")
    if code == "nodata":
        bref = f"\\,\\tiny[{base_mean:.1f}b]" if base_mean is not None else ""
        return f"\\textcolor{{Gray}}{{{pv}$^{{\\ddagger}}${bref}}}"
    col = COLOR.get(code, "black")
    inner = f"{pv}\\,\\tiny[{mean:.1f}{fl}]"
    if code == "a1" and base_mean is not None:
        inner = f"{pv}\\,\\tiny[{mean:.1f}{fl}/{base_mean:.1f}b]"
    mark = {"green": "$^{\\S}$", "red": "$^{!}$", "amber": "$^{\\sim}$"}.get(code, "")
    return f"\\textcolor{{{col}}}{{{inner}{mark}}}"


def table(title, paper, metric_idx, metric_names, label):
    lines = [
        r"\begin{table*}[t]\centering\scriptsize\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l cc cc cc cc}",
        r"\toprule",
        r"& \multicolumn{4}{c}{\textbf{IFEval}} & \multicolumn{4}{c}{\textbf{Hyponymy}} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r"& \multicolumn{2}{c}{G2-9b-it} & \multicolumn{2}{c}{Q3.5-9b} & \multicolumn{2}{c}{G2-9b-it} & \multicolumn{2}{c}{Q3.5-9b} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        f"\\textbf{{Method}} & {metric_names[0]} & {metric_names[1]} & {metric_names[0]} & {metric_names[1]} & {metric_names[0]} & {metric_names[1]} & {metric_names[0]} & {metric_names[1]} \\\\",
        r"\midrule",
    ]
    mkeys = metric_idx  # (metric_for_col0, metric_for_col1)
    for label_m, num in ROWS:
        cells = []
        for _, task in TASKS:
            for _, model in MODELS:
                pv = paper.get((task, model, num), (None, None))
                cells.append(cell_tex(pv[0], mkeys[0], model, task, num))
                cells.append(cell_tex(pv[1], mkeys[1], model, task, num))
        lines.append(f"{label_m} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              f"\\caption{{{title} Each cell: \\textbf{{paper value}} {{\\tiny[my original-v7 value + variant: "
              r"s=self/own, n=neg/own, sb=self-base, nb=neg-base]}. "
              r"\textcolor{blue}{blue}=$<$1pp (faithful); "
              r"\textcolor{RoyalPurple}{purple}=paper used \emph{own}-typ (a1; base shown as /\#b); "
              r"\textcolor{ForestGreen}{green}$^{\S}$=RankAlign, paper epoch1 vs my epoch2 (a2); "
              r"\textcolor{red}{red}$^{!}$=$>$5pp from own (unexplained); "
              r"\textcolor{orange}{orange}$^{\sim}$=1--5pp from own (unexplained); "
              r"\textcolor{Gray}{grey}$^{\ddagger}$=paper's own-typ variant not on disk "
              r"(can't verify; base shown for reference; FLORA-IFEval also pre-bug-fix per paper).}"
              f"\n\\label{{{label}}}", r"\end{table*}"]
    return "\n".join(lines)


def main():
    t1 = table("ROC$_G$ \\& $\\rho$: paper vs.\\ original v7.", PAPER1,
               ("gen_roc", "rho"), ("ROC$_G$", "$\\rho$"), "tab:cmp-main")
    t2 = table("ROC$_V$ \\& Acc$_V$: paper vs.\\ original v7.", PAPER2,
               ("val_roc", "val_acc"), ("ROC$_V$", "Acc$_V$"), "tab:cmp-val")
    header = (
        "% Colored comparison: original-v7 on-disk vs paper (main.tex).\n"
        "% Auto-generated by scripts/_build_paper_comparison.py.\n"
        "% Needs: \\usepackage[dvipsnames]{xcolor}, \\stdv, booktabs.\n"
    )
    OUT.write_text(header + "\n" + t1 + "\n\n" + t2 + "\n")
    print(f"wrote {OUT}")
    print("codes present:", sorted(flags))
    # text summary of notable cells
    print("\n--- notable (a1 / green / red / nodata) for ROC_G + ROC_V ---")
    for paper, mlist in ((PAPER1, [("gen_roc", 0)]), (PAPER2, [("val_roc", 0)])):
        for (task, model), _ in [(t, None) for t in [("ifeval", "9b-it"), ("ifeval", "qwen"),
                                                      ("rosch", "9b-it"), ("rosch", "qwen")]]:
            for label_m, num in ROWS:
                metric, idx = mlist[0]
                pv = paper.get((task, model, num), (None, None))[idx]
                code, mean, field, base_mean, note = classify(pv, metric, model, task, num)
                if code in ("a1", "green", "red", "nodata", "amber"):
                    mm = f"{mean:.1f}" if mean is not None else "--"
                    print(f"  [{code:6s}] {metric:7s} {label_m:14s} {task}/{model}: "
                          f"paper={pv} mine={mm}({field}) {note}")


if __name__ == "__main__":
    main()
