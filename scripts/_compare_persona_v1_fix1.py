#!/usr/bin/env python3
"""Compare parent (v6) vs fix1 (v7) persona-v1 metrics for variants #3/#4/#7.

Why a separate script:
  - User explicitly asked NOT to extend the existing metrics tables. Instead,
    recompute the relevant cells on the fly and present a side-by-side
    parent-vs-fix1 comparison.

What this script does:
  - For each base in {9b-it, 2b-it, 2b}:
    For variants {#3 New, #4 New+selfTC, #7 New+negTC}:
      - Compute the parent (v6, --force-same-x) cell from existing eval CSVs.
      - Compute the fix1 (v7, --no-force-same-x, --fix1 suffix) cell from
        the new overnight eval CSVs.
      - Show parent | fix1 | Δ side-by-side.
  - Columns shown depend on the variant's TC policy:
      #3 (no TC)     -> Raw, PMI base, Neg base
      #4 (selfTC)    -> Raw, PMI base
      #7 (negTC)     -> Raw, Neg base
  - Cells: mean ± SE × 100 across N=6 personas (or 3 if SPLIT=id/ood).

Assumptions:
  - Parent eval CSVs were produced with disc-shots zero for 9b-it/2b-it and
    disc-shots few for 2b. Fix1 evals use the same (verified by
    run_train_persona_v1_fix1.sh + run_eval_persona_v1_trained_fix1.sh).
  - Both use --base-typcorr at eval time (so PMI/Neg cols are the
    `basetyp-`/`basetypneg-` flavors, NOT bare `self-`/`neg-`).
  - Metric default = gen_roc (set PERSONA_METRIC to override).

Env vars:
  PERSONA_SPLIT   ∈ {id, ood, all}   (default all)
  PERSONA_METRIC  ∈ {gen_roc, pearson, spearman, val_roc, val_acc}
  PERSONA_DISC    ∈ {auto, zero, few}
                    'auto' (default) means: use zero for 9b-it/2b-it,
                    few for 2b. Set explicitly to override.

Output: stdout markdown tables only. No CSVs written.
"""

from __future__ import annotations
import os
import sys
import math
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from summarize_scores_file import load_scores, compute_all_metrics  # noqa: E402

OUT_DIR = REPO / "outputs"

# ============================================================================
# Config
# ============================================================================

PERSONA_V1_ID  = ["psychopathy", "machiavellianism", "narcissism"]
PERSONA_V1_OOD = ["desire-to-create-allies", "interest-in-music", "interest-in-science"]
PERSONA_V1_ALL = PERSONA_V1_ID + PERSONA_V1_OOD

SPLIT = (os.environ.get("PERSONA_SPLIT", "all")).lower()
if SPLIT == "id":
    PERSONAS = PERSONA_V1_ID
elif SPLIT == "ood":
    PERSONAS = PERSONA_V1_OOD
elif SPLIT == "all":
    PERSONAS = PERSONA_V1_ALL
else:
    raise SystemExit(f"PERSONA_SPLIT must be 'id'/'ood'/'all', got {SPLIT!r}")
EVAL_TASKS = [f"persona-v1-{p}" for p in PERSONAS]
N_EXPECTED = len(EVAL_TASKS)

METRIC = (os.environ.get("PERSONA_METRIC", "gen_roc")).lower()
SUPPORTED_METRICS = {"gen_roc", "pearson", "spearman", "val_roc", "val_acc"}
if METRIC not in SUPPORTED_METRICS:
    raise SystemExit(f"PERSONA_METRIC must be one of {SUPPORTED_METRICS}")
METRIC_LABEL = {
    "gen_roc": "GenROC", "pearson": "Pearson(gen, val)",
    "spearman": "Spearman(gen, val)", "val_roc": "ValROC", "val_acc": "ValAcc",
}[METRIC]

DISC_OVERRIDE = (os.environ.get("PERSONA_DISC", "auto")).lower()


# Mirror src/tasks/common.py:build_model_short
_MODEL_SHORT_MAX_LEN = 160


def _maybe_truncate_for_match(raw: str) -> str:
    if len(raw) > _MODEL_SHORT_MAX_LEN:
        h = hashlib.md5(raw.encode()).hexdigest()[:8]
        return raw[:_MODEL_SHORT_MAX_LEN - 9] + '_' + h
    return raw


def _make_match(canonical: str):
    """Match either the literal canonical form or its hashed truncation."""
    truncated = _maybe_truncate_for_match(canonical)
    if canonical == truncated:
        def m(s, _c=canonical):
            return s == _c
    else:
        def m(s, _c=canonical, _t=truncated):
            return s == _c or s == _t
    m.__doc__ = canonical
    return m


# ============================================================================
# Variant suffix definitions -- parent (with fsx) vs fix1 (no fsx, --fix1).
# ============================================================================

# Each entry: variant_num -> dict(label, parent_suffix, fix1_suffix, columns).
# `columns` is the list of (col_label, eval_prefix, variant) tuples we want
# to show -- restricted to the cols meaningful for that variant's TC training.
COLUMN_SPECS = {
    "Raw":      ("Raw",      ["self-", "neg-", "basetyp-", "basetypneg-", ""], "raw"),
    "PMI base": ("PMI base", "basetyp-",    "tc"),
    "Neg base": ("Neg base", "basetypneg-", "tc"),
}

VARIANTS = {
    3: dict(
        label="3.New",
        parent_suffix="--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1",
        fix1_suffix=  "--full-completion--nllv1.0--nllg1.0--vallogodds--semi0.1--fix1",
        columns=["Raw", "PMI base", "Neg base"],
    ),
    4: dict(
        label="4.New+selfTC",
        parent_suffix="--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1",
        fix1_suffix=  "--tc-self--full-completion--nllv1.0--nllg1.0--vallogodds--semi0.1--fix1",
        columns=["Raw", "PMI base"],
    ),
    7: dict(
        label="7.New+negTC",
        parent_suffix="--tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1",
        fix1_suffix=  "--tc-neg--full-completion--nllv1.0--nllg1.0--vallogodds--semi0.1--fix1",
        columns=["Raw", "Neg base"],
    ),
}


def build_match(base_key: str, kind: str, variant_num: int):
    """Return a model_short matcher for (base_key, kind, variant_num).
    kind in {'parent', 'fix1'}.
    """
    spec = VARIANTS[variant_num]
    if kind == "parent":
        prefix = "v6-"
        suffix = spec["parent_suffix"]
    else:
        prefix = "v7-"
        suffix = spec["fix1_suffix"]
    merged = "_merged" if base_key == "9b-it" else ""
    cp = (f"{prefix}google--gemma-2-{base_key}-delta0.15-epoch2"
          f"--persona-v1-all--d2g--random--alpha1.0")
    canonical = f"{cp}{suffix}{merged}"
    return _make_match(canonical)


# ============================================================================
# File index + cell computation (adapted from _build_persona_v1_table.py)
# ============================================================================

def _build_index() -> list[tuple[str, str, str, Path]]:
    KNOWN_PREFIXES = ("self-", "neg-", "basetyp-", "basetypneg-")
    index: list[tuple[str, str, str, Path]] = []
    for p in OUT_DIR.glob("scores_*persona-v1-*_test_log-odds*.csv"):
        name = p.name
        after_prefix = name[len("scores_"):]
        pfx = ""
        for kp in KNOWN_PREFIXES:
            if after_prefix.startswith(kp):
                pfx = kp
                break
        rest = after_prefix[len(pfx):]
        chosen_task = None
        for t in EVAL_TASKS:
            if f"_{t}_test_log-odds" in rest:
                chosen_task = t
                break
        if chosen_task is None:
            continue
        model_short = rest.split(f"_{chosen_task}_test_log-odds", 1)[0]
        index.append((pfx, model_short, chosen_task, p))
    return index


_FILE_INDEX: list[tuple[str, str, str, Path]] | None = None


def _disc_for_base(base_key: str) -> str:
    if DISC_OVERRIDE != "auto":
        return DISC_OVERRIDE
    return "few" if base_key == "2b" else "zero"


def _check_disc_match(df: pd.DataFrame, base_key: str) -> bool:
    want = _disc_for_base(base_key)
    if "strategy" not in df.columns or df["strategy"].empty:
        return False
    return f"disc:{want}" in str(df["strategy"].iloc[0])


def find_score_files(matcher, eval_prefix) -> dict[str, list[Path]]:
    global _FILE_INDEX
    if _FILE_INDEX is None:
        _FILE_INDEX = _build_index()

    if isinstance(eval_prefix, str):
        prefixes = {eval_prefix}
    else:
        prefixes = set(eval_prefix)

    matches: dict[str, list[Path]] = {t: [] for t in EVAL_TASKS}
    for pfx, model_short, task, p in _FILE_INDEX:
        if pfx not in prefixes:
            continue
        if not matcher(model_short):
            continue
        matches[task].append(p)
    return matches


def cell_value(matcher, eval_prefix, variant: str, base_key: str):
    files_by_task = find_score_files(matcher, eval_prefix)
    vals = []
    for task in EVAL_TASKS:
        candidates = files_by_task.get(task, [])
        if not candidates:
            continue
        # Pick the latest-by-name candidate that passes disc filter.
        chosen_df = None
        for path in sorted(candidates, reverse=True):
            try:
                df = load_scores(path)
            except Exception:
                continue
            if not _check_disc_match(df, base_key):
                continue
            chosen_df = df
            break
        if chosen_df is None:
            continue
        try:
            metrics = compute_all_metrics(chosen_df)
        except Exception:
            continue
        if variant not in metrics:
            continue
        v = metrics[variant].get(METRIC)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        vals.append(float(v))
    n = len(vals)
    if n == 0:
        return dict(mean=None, se=None, n=0)
    arr = np.array(vals, dtype=float) * 100.0
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    return dict(mean=mean, se=se, n=n)


def fmt_cell(c: dict) -> str:
    if c["n"] == 0:
        return "—"
    if c["se"] is None or math.isnan(c["se"]):
        s = f"{c['mean']:.2f}"
    else:
        s = f"{c['mean']:.2f} ± {c['se']:.2f}"
    if c["n"] != N_EXPECTED:
        s += f" (n={c['n']})"
    return s


def fmt_delta(parent: dict, fix1: dict) -> str:
    if parent["n"] == 0 or fix1["n"] == 0:
        return "—"
    delta = fix1["mean"] - parent["mean"]
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.2f}"


# ============================================================================
# Main
# ============================================================================

def render_table(base_key: str):
    disc_used = _disc_for_base(base_key)
    print()
    print(f"### gemma-2-{base_key}  (disc-shots: {disc_used})")
    print()

    # We always render rows for #3, #4, #7. Each row has 3 numeric "groups":
    # (Raw_parent, Raw_fix1, Δ_Raw), (PMI_parent, PMI_fix1, Δ_PMI),
    # (Neg_parent, Neg_fix1, Δ_Neg). For variants where a column is N/A by
    # TC policy, we render "---".
    cols_in_table = ["Raw", "PMI base", "Neg base"]
    header = ["Variant"]
    for c in cols_in_table:
        header += [f"{c} (parent)", f"{c} (fix1)", f"Δ {c}"]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")

    for vnum in [3, 4, 7]:
        spec = VARIANTS[vnum]
        parent_match = build_match(base_key, "parent", vnum)
        fix1_match = build_match(base_key, "fix1", vnum)

        row_cells = [f"{vnum}.{spec['label'].split('.', 1)[1]}"]
        for col_label in cols_in_table:
            if col_label not in spec["columns"]:
                row_cells += ["---", "---", "---"]
                continue
            col_meta = COLUMN_SPECS[col_label]
            _, eval_prefix, variant_str = col_meta
            p_cell = cell_value(parent_match, eval_prefix, variant_str, base_key)
            f_cell = cell_value(fix1_match,   eval_prefix, variant_str, base_key)
            row_cells += [fmt_cell(p_cell), fmt_cell(f_cell), fmt_delta(p_cell, f_cell)]
        print("| " + " | ".join(row_cells) + " |")


def main():
    split_human = {"id": "in-domain (3)", "ood": "held-out OOD (3)", "all": "all 6"}[SPLIT]
    print(f"# Persona-v1 {METRIC_LABEL} × 100 — fix1 vs parent")
    print()
    print(f"- Split: **{split_human}** ({', '.join(PERSONAS)})")
    print(f"- Metric: **{METRIC_LABEL}**")
    print(f"- Each cell: mean ± SE across personas. Δ = fix1 − parent.")
    print(f"- 'parent' = v6 (--force-same-x). 'fix1' = v7 (--no-force-same-x, --fix1 suffix).")
    print(f"- Eval flags: --base-typcorr always; --self-typcorr or --neg-typcorr per-variant TC policy.")

    for base_key in ["9b-it", "2b-it", "2b"]:
        render_table(base_key)


if __name__ == "__main__":
    main()
