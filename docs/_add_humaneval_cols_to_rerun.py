"""Append HumanEval (Upper) + HumanEval (Multi) column-groups to rerun_only_tables.tex.

Preserves the existing Hyponymy / IFEval cells byte-for-byte and adds two new 4-wide
groups (G4-31b + Q3.5-9b each) to BOTH the Generator and Validator tables, plus a
provenance bullet. The HumanEval cells are computed with the SAME estimator the existing
columns use (per-problem metric -> mean x100, SE = std(ddof=1)/sqrt(n) x100; see
_build_ifeval_table_v7.py:cell_value) and rendered identically (mean 1dp + \\stdv{se 1dp},
bold = per-column max of 1-dp-rounded mean).

Source: docs/he_harvest_2026-06-17/{qwen,gemma_cu,gemma_cm}_he_perfile_metrics.csv
        (per-file = per-problem rows; humaneval TEST split, 82 problems each).

ROC_G = gen_roc[tc], rho = SPEARMAN spearman[tc] (matches this doc's rho column).
Validator metrics are eval-mode-independent; taken from one mode per (model,ds,setting)
via the assembler's priority self/self -> self/base -> neg/self -> neg/base.

Idempotent guard: aborts if 'HumanEval' already present in the target .tex.
"""
from __future__ import annotations
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
HARVEST = HERE / "he_harvest_2026-06-17"
TEX = HERE / "rerun_only_tables.tex"
FILES = ["qwen_he_perfile_metrics.csv", "gemma_cu_perfile_metrics.csv", "gemma_cm_perfile_metrics.csv"]

# label -> (setting token used in filenames)
METHOD_SETTING = {"Base": "base", "SFT": "s1", "RankAlign": "s2",
                  "Consistency FT": "s13", "FLORA-PMI": "s4", "FLORA-Neg": "s7"}
# table eval-variant -> (typ, base/self)
VARIANT_MODE = {"PMI self": "self/self", "PMI base": "self/base",
                "Neg self": "neg/self", "Neg base": "neg/base"}
VAL_MODE_PRIORITY = ["self/self", "self/base", "neg/self", "neg/base"]
# new column groups: (dataset, model_key) in display order
GROUPS = [("upper", "gemma4"), ("upper", "qwen"), ("multi", "gemma4"), ("multi", "qwen")]


def parse(fname: str):
    m = re.match(r"scores_(basetypneg|basetyp|self|neg)-", fname)
    pref = m.group(1) if m else "?"
    is_base = ("v6-" in fname) and ("-v7-" not in fname)
    if "Qwen3.5-9B" in fname or "Qwen--Qwen3" in fname or "Qwen_Qwen3" in fname:
        mkey = "qwen"
    elif "gemma-4-31" in fname.lower():
        mkey = "gemma4"
    else:
        mkey = "?"
    ds = "upper" if "correct-upper" in fname else ("multi" if "correct-multi" in fname else "?")
    if is_base:
        setting = "base"
    elif ("tcs" in fname) or ("tc-self" in fname):
        setting = "s4"
    elif ("tcn" in fname) or ("tc-neg" in fname):
        setting = "s7"
    elif "cft" in fname:
        setting = "s13"
    elif ("fsx" in fname) or ("force-same-x" in fname):
        setting = "s3"
    elif ("lo0.1" in fname) or ("labelonly" in fname) or re.search(r"-p0-", fname) or ("pref0" in fname):
        setting = "s1"
    else:
        setting = "s2"
    typ = "neg" if pref in ("neg", "basetypneg") else "self"
    bt = "base" if pref in ("basetyp", "basetypneg") else "self"
    return mkey, ds, setting, f"{typ}/{bt}"


def load_all() -> pd.DataFrame:
    rows = []
    for fn in FILES:
        df = pd.read_csv(HARVEST / fn)
        df = df[df["file"].str.contains("humaneval", na=False)].copy()
        meta = df["file"].apply(lambda f: pd.Series(parse(f), index=["mkey", "ds", "setting", "evalmode"]))
        rows.append(pd.concat([meta, df], axis=1))
    return pd.concat(rows, ignore_index=True)


def agg(values: np.ndarray) -> tuple[float, float, int]:
    """mean x100, se = std(ddof=1)/sqrt(n) x100 -- exactly cell_value()."""
    v = values[~np.isnan(values)] * 100.0
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(v.mean())
    se = float(v.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return mean, se, n


def cell_stat(df, setting, evalmode, model, variant_col):
    """Return (mean, se, n) for one (setting, evalmode, model) on a 'tc' metric column."""
    sub = df[(df["setting"] == setting) & (df["evalmode"] == evalmode) &
             (df["mkey"] == model) & (df["variant"] == "tc")]
    if sub.empty:
        return float("nan"), float("nan"), 0
    return agg(sub[variant_col].to_numpy(dtype=float))


def val_stat(df, setting, model, ds, col):
    """Validator metric (eval-mode-independent): pick first available mode by priority."""
    for mode in VAL_MODE_PRIORITY:
        sub = df[(df["setting"] == setting) & (df["evalmode"] == mode) &
                 (df["mkey"] == model) & (df["ds"] == ds) & (df["variant"] == "raw")]
        if not sub.empty:
            return agg(sub[col].to_numpy(dtype=float))
    return float("nan"), float("nan"), 0


def render(mean, se, is_max):
    if math.isnan(mean):
        return "---"
    num = rf"\textbf{{{mean:.1f}}}" if is_max else f"{mean:.1f}"
    return num + (rf"\stdv{{{se:.1f}}}" if not math.isnan(se) else "")


def colmax(values):
    vals = [round(m, 1) for (m, _) in values if not math.isnan(m)]
    return max(vals) if vals else None


def build_gen_cells(df):
    """Return dict[(label,variant)] -> list[str x8], with per-column bolding."""
    # raw (mean,se) per (label,variant) per column
    raw = {}
    for label, setting in METHOD_SETTING.items():
        for variant, evalmode in VARIANT_MODE.items():
            cells = []
            for ds, model in GROUPS:
                em = evalmode  # typ/self|base; but data uses evalmode incl ds via filter on ds
                # filter ds too:
                sub = df[(df["setting"] == setting) & (df["evalmode"] == em) &
                         (df["mkey"] == model) & (df["ds"] == ds) & (df["variant"] == "tc")]
                if sub.empty:
                    roc = (float("nan"), float("nan"))
                    rho = (float("nan"), float("nan"))
                else:
                    m, s, _ = agg(sub["gen_roc"].to_numpy(dtype=float)); roc = (m, s)
                    m, s, _ = agg(sub["spearman"].to_numpy(dtype=float)); rho = (m, s)
                cells.extend([roc, rho])
            raw[(label, variant)] = cells
    # per-column max over all (label,variant)
    ncol = 8
    maxes = [colmax([raw[k][j] for k in raw]) for j in range(ncol)]
    out = {}
    for k, cells in raw.items():
        out[k] = [render(m, s, maxes[j] is not None and not math.isnan(m) and round(m, 1) == maxes[j])
                  for j, (m, s) in enumerate(cells)]
    return out


def build_val_cells(df):
    raw = {}
    for label, setting in METHOD_SETTING.items():
        cells = []
        for ds, model in GROUPS:
            cells.append(val_stat(df, setting, model, ds, "val_roc"))  # (m,se,n)
            cells.append(val_stat(df, setting, model, ds, "val_acc"))
        raw[label] = [(c[0], c[1]) for c in cells]
    ncol = 8
    maxes = [colmax([raw[k][j] for k in raw]) for j in range(ncol)]
    out = {}
    for k, cells in raw.items():
        out[k] = [render(m, s, maxes[j] is not None and not math.isnan(m) and round(m, 1) == maxes[j])
                  for j, (m, s) in enumerate(cells)]
    return out


# ---- header / colspec / resizebox rewriting --------------------------------
GEN_HDR_OLD = r"""\begin{center}\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{ll cccc cccc}
\toprule
& & \multicolumn{4}{c}{\textbf{Hyponymy}} & \multicolumn{4}{c}{\textbf{IFEval (OOD)}} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-10}
& & \multicolumn{2}{c}{G2-9b-it} & \multicolumn{2}{c}{Q3.5-9b} & \multicolumn{2}{c}{G2-9b-it} & \multicolumn{2}{c}{Q3.5-9b} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}
\textbf{Method} & \textbf{eval} & ROC$_G$ & $\rho$ & ROC$_G$ & $\rho$ & ROC$_G$ & $\rho$ & ROC$_G$ & $\rho$ \\"""

GEN_HDR_NEW = r"""\begin{center}\setlength{\tabcolsep}{3pt}\resizebox{\textwidth}{!}{%
\begin{tabular}{ll cccc cccc cccc cccc}
\toprule
& & \multicolumn{4}{c}{\textbf{Hyponymy}} & \multicolumn{4}{c}{\textbf{IFEval (OOD)}} & \multicolumn{4}{c}{\textbf{HumanEval (Upper)}} & \multicolumn{4}{c}{\textbf{HumanEval (Multi)}} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}\cmidrule(lr){15-18}
& & \multicolumn{2}{c}{G2-9b-it} & \multicolumn{2}{c}{Q3.5-9b} & \multicolumn{2}{c}{G2-9b-it} & \multicolumn{2}{c}{Q3.5-9b} & \multicolumn{2}{c}{G4-31b} & \multicolumn{2}{c}{Q3.5-9b} & \multicolumn{2}{c}{G4-31b} & \multicolumn{2}{c}{Q3.5-9b} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}\cmidrule(lr){13-14}\cmidrule(lr){15-16}\cmidrule(lr){17-18}
\textbf{Method} & \textbf{eval} & ROC$_G$ & $\rho$ & ROC$_G$ & $\rho$ & ROC$_G$ & $\rho$ & ROC$_G$ & $\rho$ & ROC$_G$ & $\rho$ & ROC$_G$ & $\rho$ & ROC$_G$ & $\rho$ & ROC$_G$ & $\rho$ \\"""

VAL_HDR_OLD = r"""\begin{center}\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{l cccc cccc}
\toprule
& \multicolumn{4}{c}{\textbf{Hyponymy}} & \multicolumn{4}{c}{\textbf{IFEval (OOD)}} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
& \multicolumn{2}{c}{G2-9b-it} & \multicolumn{2}{c}{Q3.5-9b} & \multicolumn{2}{c}{G2-9b-it} & \multicolumn{2}{c}{Q3.5-9b} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
\textbf{Method} & ROC$_V$ & Acc$_V$ & ROC$_V$ & Acc$_V$ & ROC$_V$ & Acc$_V$ & ROC$_V$ & Acc$_V$ \\"""

VAL_HDR_NEW = r"""\begin{center}\setlength{\tabcolsep}{3pt}\resizebox{\textwidth}{!}{%
\begin{tabular}{l cccc cccc cccc cccc}
\toprule
& \multicolumn{4}{c}{\textbf{Hyponymy}} & \multicolumn{4}{c}{\textbf{IFEval (OOD)}} & \multicolumn{4}{c}{\textbf{HumanEval (Upper)}} & \multicolumn{4}{c}{\textbf{HumanEval (Multi)}} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}\cmidrule(lr){14-17}
& \multicolumn{2}{c}{G2-9b-it} & \multicolumn{2}{c}{Q3.5-9b} & \multicolumn{2}{c}{G2-9b-it} & \multicolumn{2}{c}{Q3.5-9b} & \multicolumn{2}{c}{G4-31b} & \multicolumn{2}{c}{Q3.5-9b} & \multicolumn{2}{c}{G4-31b} & \multicolumn{2}{c}{Q3.5-9b} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}\cmidrule(lr){14-15}\cmidrule(lr){16-17}
\textbf{Method} & ROC$_V$ & Acc$_V$ & ROC$_V$ & Acc$_V$ & ROC$_V$ & Acc$_V$ & ROC$_V$ & Acc$_V$ & ROC$_V$ & Acc$_V$ & ROC$_V$ & Acc$_V$ & ROC$_V$ & Acc$_V$ & ROC$_V$ & Acc$_V$ \\"""

PROV_EXTRA = r"""
\item \textbf{HumanEval (Upper) \& (Multi) / G4-31b \& Q3.5-9b}: TEST split, mean $\pm$ SE over
82 problems (per-problem metric, then $\bar{x}\pm \mathrm{sd}/\sqrt{n}$, same estimator as the other
columns). ROC$_G$/$\rho$ are the \emph{tc} (typicality-corrected) variant; $\rho$ is Spearman.
gemma-4-31b has no base eval, so its Base cells are \texttt{---}. Source score CSVs:
\texttt{outputs\_gemma4\_mll\_tmp[-multi]/} (gemma-4) and \texttt{outputs-rerun-wandb/} (qwen);
harvested into \texttt{docs/he\_harvest\_2026-06-17/}. Exact checkpoint dir names per setting:
see \texttt{docs/he\_trainset\_eval\_model\_origins.md}. Note: HumanEval s13 (Consistency FT) also
has self-typ evals, not shown here because this shared table only has CFT base-typ rows."""


def splice_rows(block: str, lookup: dict, key_is_variant: bool) -> str:
    """Append 8 cells to each data row in a table block. Rows are matched by their
    leading 'Method[ & eval]' fields; \\midrule / header lines pass through."""
    out_lines = []
    for line in block.split("\n"):
        st = line.strip()
        m = re.match(r"^(Base|SFT|RankAlign|Consistency FT|FLORA-PMI|FLORA-Neg)\b(.*)\\\\$", st)
        if not m or st.startswith(r"\textbf{Method}"):
            out_lines.append(line)
            continue
        label = m.group(1)
        if key_is_variant:
            vm = re.match(r"^[^&]+&\s*(PMI self|PMI base|Neg self|Neg base)\s*&", st)
            if not vm:
                out_lines.append(line); continue
            key = (label, vm.group(1).strip())
        else:
            key = label
        cells = lookup.get(key)
        if cells is None:
            cells = ["---"] * 8
        new = st[:-2].rstrip()  # drop trailing '\\'
        if not new.endswith("&"):
            new += " "
        new += "& " + " & ".join(cells) + r" \\"
        out_lines.append(new)
    return "\n".join(out_lines)


def main():
    text = TEX.read_text()
    if "HumanEval" in text:
        raise SystemExit("ABORT: 'HumanEval' already present in rerun_only_tables.tex "
                         "(git checkout it first to re-run).")
    df = load_all()
    gen = build_gen_cells(df)
    val = build_val_cells(df)

    # split into generator / validator regions by the subsection markers
    gen_start = text.index(r"\subsection*{Generator")
    val_start = text.index(r"\subsection*{Validator")
    prov_start = text.index(r"\subsection*{Provenance")

    head = text[:gen_start]
    # \resizebox needs graphicx
    if "graphicx" not in head:
        head = head.replace(r"\usepackage{booktabs,amsmath,amssymb,xcolor}",
                            r"\usepackage{booktabs,amsmath,amssymb,xcolor,graphicx}", 1)
    gen_block = text[gen_start:val_start]
    val_block = text[val_start:prov_start]
    prov_block = text[prov_start:]

    # headers + resizebox + colspec
    assert GEN_HDR_OLD in gen_block, "generator header not found verbatim"
    assert VAL_HDR_OLD in val_block, "validator header not found verbatim"
    gen_block = gen_block.replace(GEN_HDR_OLD, GEN_HDR_NEW)
    val_block = val_block.replace(VAL_HDR_OLD, VAL_HDR_NEW)
    # close the resizebox: \end{tabular} -> \end{tabular}}
    gen_block = gen_block.replace(r"\end{tabular}", r"\end{tabular}}", 1)
    val_block = val_block.replace(r"\end{tabular}", r"\end{tabular}}", 1)

    # append cells to data rows
    gen_block = splice_rows(gen_block, gen, key_is_variant=True)
    val_block = splice_rows(val_block, val, key_is_variant=False)

    # provenance: insert PROV_EXTRA before the closing \end{itemize} of the provenance list
    last_end_itemize = prov_block.rstrip().rfind(r"\end{itemize}")
    prov_block = prov_block[:last_end_itemize] + PROV_EXTRA + "\n" + prov_block[last_end_itemize:]

    TEX.write_text(head + gen_block + val_block + prov_block)
    print("spliced HumanEval columns into", TEX)
    # quick sanity print
    for k in [("Base", "PMI self"), ("RankAlign", "Neg base"), ("FLORA-PMI", "PMI self")]:
        print(" GEN", k, "->", gen.get(k))
    print(" VAL RankAlign ->", val.get("RankAlign"))


if __name__ == "__main__":
    main()
