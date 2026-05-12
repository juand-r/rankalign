"""Shared helper for the 4-table layout described in
``docs/results_table_format.md``.

Each metric (gen_roc, val_roc, val_acc, pearson) is rendered as 4 tables:

    T1 — Baselines, self eval     (Base HF, SFT)
    T2 — TC × pairs grid, self    (3×2: {offline TC, online TC, no TC} × {offline pairs, online pairs})
    T3 — Baselines, neg eval
    T4 — TC × pairs grid, neg

The mapping from grid cell → (variant id, eval ref) is fixed by the launcher
(``scripts/run_train_membership_quickiter.sh`` /
``scripts/run_train_rosch_online_quickiter.sh``):

    self side
    ----------
    (no TC,      offline pairs) = variant 1 (RankAlign baseline)   eval_ref=self
    (no TC,      online pairs)  = variant 4 (online pairs only)    eval_ref=self
    (offline TC, offline pairs) = variant 2 (offline self-TC)      eval_ref=basetyp
    (offline TC, online pairs)  = NOT TRAINED
    (online TC,  offline pairs) = variant 3 (online self-TC)       eval_ref=self
    (online TC,  online pairs)  = variant 5 (both online self)     eval_ref=self

    neg side
    --------
    (no TC,      offline pairs) = variant 1 (RankAlign baseline)   eval_ref=neg
    (no TC,      online pairs)  = variant 4 (online pairs only)    eval_ref=neg
    (offline TC, offline pairs) = variant 7 (offline neg-TC)       eval_ref=basetypneg
    (offline TC, online pairs)  = NOT TRAINED
    (online TC,  offline pairs) = variant 8 (online neg-TC)        eval_ref=neg
    (online TC,  online pairs)  = variant 9 (both online neg)      eval_ref=neg

The eval-ref mixing (offline TC → basetyp[neg]; everything else → self/neg) is
the "best per row" convention from the doc. Each grid cell carries a small
``[basetyp]`` / ``[basetypneg]`` annotation when its eval ref differs from
the table's nominal one, so the reader can always tell.
"""
from __future__ import annotations

from typing import Callable, Optional


# (variant id, eval ref) lookup per side.
GRID_SELF = {
    ("no TC",      "offline pairs"): ("1", "self"),
    ("no TC",      "online pairs"):  ("4", "self"),
    ("offline TC", "offline pairs"): ("2", "basetyp"),
    ("offline TC", "online pairs"):  None,
    ("online TC",  "offline pairs"): ("3", "self"),
    ("online TC",  "online pairs"):  ("5", "self"),
}
GRID_NEG = {
    ("no TC",      "offline pairs"): ("1", "neg"),
    ("no TC",      "online pairs"):  ("4", "neg"),
    ("offline TC", "offline pairs"): ("7", "basetypneg"),
    ("offline TC", "online pairs"):  None,
    ("online TC",  "offline pairs"): ("8", "neg"),
    ("online TC",  "online pairs"):  ("9", "neg"),
}

GRID_ROW_ORDER = ["offline TC", "online TC", "no TC"]
GRID_COL_ORDER = ["offline pairs", "online pairs"]

# Baselines: id, label, eval_ref
BASELINES_SELF = [
    ("0", "Base HF", "self"),
    ("6", "SFT (NLL all)", "self"),
]
BASELINES_NEG = [
    ("0", "Base HF", "neg"),
    ("6", "SFT (NLL all)", "neg"),
]

NICE_LABELS = {
    "1": "RankAlign baseline",
    "2": "+ offline self-TC",
    "3": "+ online self-TC",
    "4": "+ online pairs",
    "5": "+ both online (self)",
    "6": "SFT (NLL all)",
    "7": "+ offline neg-TC",
    "8": "+ online neg-TC",
    "9": "+ both online (neg)",
}


# -- API ---------------------------------------------------------------------

# A "value getter" returns a *display string* for (variant_id, eval_ref) or
# None if no data. Display string is whatever the caller wants — a bare
# "82.78" for single-task, or "82.78 (8.12)" for multi-task mean(std).
ValueGetter = Callable[[str, str], Optional[str]]


def emit_4_tables(get_value: ValueGetter, metric_title: str) -> str:
    """Return markdown for the 4 tables under one metric.

    metric_title is the section header (e.g. "Generator ROC-AUC — × 100").
    """
    parts = [f"### {metric_title}", ""]
    parts.append(_baseline_table(get_value, BASELINES_SELF,
                                 "Table 1 — baselines, self eval"))
    parts.append(_grid_table(get_value, GRID_SELF,
                             "Table 2 — self eval, TC × pairs"))
    parts.append(_baseline_table(get_value, BASELINES_NEG,
                                 "Table 3 — baselines, neg eval"))
    parts.append(_grid_table(get_value, GRID_NEG,
                             "Table 4 — neg eval, TC × pairs"))
    return "\n".join(parts)


def _baseline_table(get_value: ValueGetter, rows, title: str) -> str:
    out = [f"#### {title}", "",
           "| variant | value |",
           "| --- | --- |"]
    any_row = False
    for vid, label, eval_ref in rows:
        v = get_value(vid, eval_ref)
        if v is None:
            v = "—"
        else:
            any_row = True
        out.append(f"| {label} | {v} |")
    out.append("")
    if not any_row:
        return "\n".join([f"#### {title}", "", "_(no data)_", ""])
    return "\n".join(out)


def _grid_table(get_value: ValueGetter, grid, title: str) -> str:
    out = [f"#### {title}", "",
           "|  | " + " | ".join(GRID_COL_ORDER) + " |",
           "| --- | " + " | ".join(["---"] * len(GRID_COL_ORDER)) + " |"]
    nominal_eval_ref = "self" if "self" in title else "neg"
    for row_label in GRID_ROW_ORDER:
        cells = [row_label]
        for col_label in GRID_COL_ORDER:
            mapping = grid[(row_label, col_label)]
            if mapping is None:
                cells.append("—")
                continue
            vid, eval_ref = mapping
            v = get_value(vid, eval_ref)
            if v is None:
                cells.append("—")
                continue
            tag = ""
            if eval_ref != nominal_eval_ref and eval_ref in {"basetyp", "basetypneg"}:
                tag = f" `[{eval_ref}]`"
            note = ""
            if vid == "1":
                note = " (RankAlign)"
            cells.append(f"{v}{tag}{note}")
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out)


# -- Convenience: format raw numbers as cell strings -------------------------

def fmt_single(value: float, scale: float = 100.0, decimals: int = 2) -> str:
    """Format a single scalar as a markdown cell, scaled."""
    return f"{value * scale:.{decimals}f}"


def fmt_mean_std(mean: float, std: float, scale: float = 100.0,
                 decimals: int = 2) -> str:
    """Format mean (std) as a markdown cell, scaled. If std is NaN/None,
    returns just the mean string."""
    if std is None or (std != std):  # NaN check (NaN != NaN)
        return fmt_single(mean, scale=scale, decimals=decimals)
    return (f"{mean * scale:.{decimals}f} "
            f"({std * scale:.{decimals}f})")
