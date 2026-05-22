#!/usr/bin/env python3
"""Build the rosch OOD table per IMPORTANT-RESEARCH-PLAN.md.

Rosch is all-OOD: the fine-tuned models were trained on
`membership-sans-rosch-v0-all`, i.e. membership data with all rosch
categories held out. So every rosch category is unseen at training time.

Rows: 12 methods (0=Base, 1..9, 11, 12), same numbering & ordering as
the IFEval driver (`scripts/_build_ifeval_ood_table.py`).
Cols: Raw, PMI self, PMI base, Neg self, Neg base.

Env vars:
- ROSCH_MODEL  ∈ {2b, 2b-it, 9b-it}   default 9b-it
- ROSCH_METRIC ∈ {gen_roc, pearson, spearman, val_roc, val_acc}
                                       default gen_roc

Writes:
- metrics-from-scores/rosch_{model}_{metric}_table_long.csv
- metrics-from-scores/rosch_{model}_{metric}_table_cells.csv
- prints the markdown table to stdout
"""

from __future__ import annotations
import os
import re
import sys
import math
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from summarize_scores_file import load_scores, compute_all_metrics  # noqa: E402

OUT_DIR = REPO / "outputs"
METRICS_DIR = REPO / "metrics-from-scores"
METRICS_DIR.mkdir(exist_ok=True)

# End-of-run provenance: every (file consumed) and every (collapsed dup
# group, kept newest) is recorded here. Summarized at the end of main().
PROVENANCE_USED: list[dict] = []
PROVENANCE_DUPS: list[dict] = []

# Rosch eval tasks (all OOD).
EVAL_TASKS = [
    "rosch-bird", "rosch-carpenters-tool", "rosch-clothing", "rosch-fruit",
    "rosch-furniture", "rosch-sport", "rosch-toy", "rosch-vegetable",
    "rosch-vehicle", "rosch-weapon",
]
N_EXPECTED = len(EVAL_TASKS)

MODEL = os.environ.get("ROSCH_MODEL", "9b-it").lower()
VALID_MODELS = {"2b", "2b-it", "9b-it"}
if MODEL not in VALID_MODELS:
    raise SystemExit(f"ROSCH_MODEL must be one of {VALID_MODELS}, got {MODEL!r}")

METRIC = os.environ.get("ROSCH_METRIC", "gen_roc").lower()
SUPPORTED_METRICS = {"gen_roc", "pearson", "spearman", "val_roc", "val_acc"}
if METRIC not in SUPPORTED_METRICS:
    raise SystemExit(f"ROSCH_METRIC must be one of {SUPPORTED_METRICS}, got {METRIC!r}")
METRIC_LABEL = {
    "gen_roc": "GenROC", "pearson": "Pearson(gen, val)",
    "spearman": "Spearman(gen, val)", "val_roc": "ValROC", "val_acc": "ValAcc",
}[METRIC]

# Base HF model string (no delta/epoch suffix).
BASE_HF = f"v6-google_gemma-2-{MODEL}"
# 9b-it ckpts have `_merged` suffix; 2b/2b-it do not.
MERGED_OPT = r"(_merged)?"

# Common prefix for trained-model regex (model + training task).
CP = rf"v6-google_gemma-2-{MODEL}-delta0\.15-epoch2_membership-sans-rosch-v0-all_d2g_random_alpha1\.0"

COLUMNS = [
    ("Raw",      ["self-", "neg-", "basetyp-", "basetypneg-"], "raw"),
    ("PMI self", "self-",       "tc"),
    ("PMI base", "basetyp-",    "tc"),
    ("Neg self", "neg-",        "tc"),
    ("Neg base", "basetypneg-", "tc"),
]

# Methods follow the same numbering as the IFEval driver.
METHODS: list[dict] = [
    dict(
        num=0, label="Base",
        match=lambda s: s == BASE_HF,
    ),
    # 1 SFT-lo: sft + labelonly, allow fsx variant (flagged in label).
    dict(
        num=1, label="SFT labelonly 10%",
        match=lambda s: bool(re.fullmatch(
            rf"{CP}_full-completion_pref0\.0_nllv1\.0_nllg1\.0(_force-same-x)?_labelonly0\.1{MERGED_OPT}",
            s,
        )),
        fsx_label_match=lambda s: "_force-same-x_" in s,
    ),
    # 2 RankAlign: pref-only baseline, semi optional (no-op in g-mode pref-only).
    dict(
        num=2, label="RankAlign",
        match=lambda s: bool(re.fullmatch(
            rf"{CP}_full-completion(_semi0\.1)?{MERGED_OPT}",
            s,
        )),
    ),
    # 3 New + fsx [-TC]
    dict(
        num=3, label="New + fsx [-TC]",
        match=lambda s: bool(re.fullmatch(
            rf"{CP}_full-completion_nllv1\.0_nllg1\.0_force-same-x_vallogodds_semi0\.1{MERGED_OPT}",
            s,
        )),
    ),
    # 4 New + PMI + fsx
    dict(
        num=4, label="New + PMI + fsx",
        match=lambda s: bool(re.fullmatch(
            rf"{CP}_tc-self_full-completion_nllv1\.0_nllg1\.0_force-same-x_vallogodds_semi0\.1{MERGED_OPT}",
            s,
        )),
    ),
    # 5 RankAlign + PMI + fsx [-NLL]: pref-only + fsx + tc-self + semi. vlo optional.
    dict(
        num=5, label="RA + PMI + fsx [-NLL]",
        match=lambda s: bool(re.fullmatch(
            rf"{CP}_tc-self_full-completion_force-same-x(_vallogodds)?_semi0\.1{MERGED_OPT}",
            s,
        )),
    ),
    # 6 RA + PMI [+TC]: pref-only + tc-self + semi, no fsx, no NLL.
    dict(
        num=6, label="RA + PMI [+TC]",
        match=lambda s: bool(re.fullmatch(
            rf"{CP}_tc-self_full-completion(_vallogodds)?_semi0\.1{MERGED_OPT}",
            s,
        )),
    ),
    # 7 New + NegTC + fsx
    dict(
        num=7, label="New + NegTC + fsx",
        match=lambda s: bool(re.fullmatch(
            rf"{CP}_tc-neg_full-completion_nllv1\.0_nllg1\.0_force-same-x_vallogodds_semi0\.1{MERGED_OPT}",
            s,
        )),
    ),
    # 8 RA + NegTC + fsx [-NLL]
    dict(
        num=8, label="RA + NegTC + fsx [-NLL]",
        match=lambda s: bool(re.fullmatch(
            rf"{CP}_tc-neg_full-completion_force-same-x(_vallogodds)?_semi0\.1{MERGED_OPT}",
            s,
        )),
    ),
    # 9 RA + NegTC [+TC]
    dict(
        num=9, label="RA + NegTC [+TC]",
        match=lambda s: bool(re.fullmatch(
            rf"{CP}_tc-neg_full-completion(_vallogodds)?_semi0\.1{MERGED_OPT}",
            s,
        )),
    ),
    # 11 New + PMI [-fsx]
    dict(
        num=11, label="New + PMI [-fsx]",
        match=lambda s: bool(re.fullmatch(
            rf"{CP}_tc-self_full-completion_nllv1\.0_nllg1\.0_vallogodds_semi0\.1{MERGED_OPT}",
            s,
        )),
    ),
    # 12 New + NegTC [-fsx]
    dict(
        num=12, label="New + NegTC [-fsx]",
        match=lambda s: bool(re.fullmatch(
            rf"{CP}_tc-neg_full-completion_nllv1\.0_nllg1\.0_vallogodds_semi0\.1{MERGED_OPT}",
            s,
        )),
    ),
]

NA_COLS = {
    4: {"Neg self", "Neg base"},
    5: {"Neg self", "Neg base"},
    6: {"Neg self", "Neg base"},
    11: {"Neg self", "Neg base"},
    7: {"PMI self", "PMI base"},
    8: {"PMI self", "PMI base"},
    9: {"PMI self", "PMI base"},
    12: {"PMI self", "PMI base"},
}


# Most specific first so `basetyp-` doesn't match `basetypneg-`.
_EVAL_PREFIXES_ORDERED = ("basetypneg-", "basetyp-", "self-", "neg-")


def _extract_eval_prefix(filename: str) -> str:
    after = filename[len("scores_"):] if filename.startswith("scores_") else filename
    for p in _EVAL_PREFIXES_ORDERED:
        if after.startswith(p):
            return p
    return ""


def find_score_files(method: dict, eval_prefix: str | list[str]) -> dict[str, list[Path]]:
    if isinstance(eval_prefix, str):
        prefixes = [eval_prefix]
    else:
        prefixes = list(eval_prefix)
    matches: dict[str, list[Path]] = {t: [] for t in EVAL_TASKS}
    for pfx in prefixes:
        pattern = f"scores_{pfx}*_test_log-odds*.csv"
        for p in OUT_DIR.glob(pattern):
            name = p.name
            after_prefix = name[len("scores_"):]
            if pfx and not after_prefix.startswith(pfx):
                continue
            if not pfx:
                if any(after_prefix.startswith(x) for x in ("self-", "neg-", "basetyp-", "basetypneg-")):
                    continue
            rest = after_prefix[len(pfx):] if pfx else after_prefix
            chosen_task = None
            for t in EVAL_TASKS:
                if f"_{t}_test_log-odds" in rest:
                    chosen_task = t
                    break
            if chosen_task is None:
                continue
            model_short = rest.split(f"_{chosen_task}_test_log-odds", 1)[0]
            if method["match"](model_short):
                matches[chosen_task].append(p)
    return matches


def cell_value(method: dict, eval_prefix, variant: str):
    files_by_task = find_score_files(method, eval_prefix)
    vals = []
    used_files = []
    for task in EVAL_TASKS:
        candidates = files_by_task.get(task, [])
        if not candidates:
            continue
        from collections import defaultdict
        buckets: dict[str, list[Path]] = defaultdict(list)
        for c in candidates:
            buckets[_extract_eval_prefix(c.name)].append(c)
        for pfx, fs in buckets.items():
            if len(fs) > 1:
                fs_sorted = sorted(fs)
                kept = fs_sorted[-1]
                dropped = fs_sorted[:-1]
                PROVENANCE_DUPS.append(dict(
                    method_num=method.get("num"),
                    method=method.get("label"),
                    task=task, prefix=pfx,
                    kept=str(kept),
                    dropped=";".join(str(d) for d in dropped),
                ))
                buckets[pfx] = [kept]
        single_candidates = [fs[0] for fs in buckets.values()]
        path = sorted(single_candidates)[0]
        PROVENANCE_USED.append(dict(
            method_num=method.get("num"),
            method=method.get("label"),
            task=task, file=path.name,
        ))
        try:
            df = load_scores(path)
            metrics = compute_all_metrics(df)
        except Exception:
            continue
        if variant not in metrics:
            continue
        v = metrics[variant].get(METRIC)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        vals.append((task, float(v), path.name))
        used_files.append((task, path.name, float(v)))
    n = len(vals)
    if n == 0:
        return dict(mean=None, se=None, n=0, vals=[], files=[])
    arr = np.array([v for _, v, _ in vals], dtype=float) * 100.0
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    return dict(mean=mean, se=se, n=n, vals=vals, files=used_files)


def fmt_cell(c: dict, expected_n: int) -> str:
    if c["n"] == 0:
        return "—"
    if c["se"] is None or math.isnan(c["se"]):
        s = f"{c['mean']:.2f}"
    else:
        s = f"{c['mean']:.2f} ± {c['se']:.2f}"
    if c["n"] != expected_n:
        s += f" (n={c['n']})"
    return s


def main():
    long_rows = []
    cell_rows = []
    table_rows = []

    ROW_ORDER = [0, 1, 2, 3, 4, 5, 6, 11, 7, 8, 9, 12]
    methods_by_num = {m["num"]: m for m in METHODS}
    ordered = [methods_by_num[n] for n in ROW_ORDER]

    fsx_label_suffix: dict[int, str] = {}

    for m in ordered:
        cells = {}
        if "fsx_label_match" in m:
            seen_fsx = False
            for _col, pfx, _v in COLUMNS:
                for f in find_score_files(m, pfx).values():
                    for p in f:
                        if m["fsx_label_match"](p.name):
                            seen_fsx = True
                            break
                    if seen_fsx:
                        break
                if seen_fsx:
                    break
            fsx_label_suffix[m["num"]] = " +fsx" if seen_fsx else ""
        for col_label, eval_prefix, variant in COLUMNS:
            if col_label in NA_COLS.get(m["num"], set()):
                cells[col_label] = "---"
                cell_rows.append(dict(
                    method_num=m["num"], method=m["label"],
                    column=col_label, mean=None, se=None, n=0, note="NA per template",
                ))
                continue
            c = cell_value(m, eval_prefix, variant)
            cells[col_label] = fmt_cell(c, N_EXPECTED)
            cell_rows.append(dict(
                method_num=m["num"], method=m["label"],
                column=col_label, mean=c["mean"], se=c["se"], n=c["n"],
                note=("OK" if c["n"] == N_EXPECTED else f"missing {N_EXPECTED - c['n']}/{N_EXPECTED}"),
            ))
            for task, val, fn in c["vals"]:
                long_rows.append(dict(
                    method_num=m["num"], method=m["label"],
                    column=col_label, eval_prefix=eval_prefix, variant=variant,
                    task=task, value=val, file=fn,
                ))
        table_rows.append(dict(num=m["num"], label=m["label"], **cells))

    long_csv = METRICS_DIR / f"rosch_{MODEL}_{METRIC}_table_long.csv"
    cells_csv = METRICS_DIR / f"rosch_{MODEL}_{METRIC}_table_cells.csv"
    pd.DataFrame(long_rows).to_csv(long_csv, index=False)
    pd.DataFrame(cell_rows).to_csv(cells_csv, index=False)

    print(f"\nRosch (all OOD) {METRIC_LABEL} × 100 — mean ± SE across {N_EXPECTED} categories")
    print(f"Model: gemma-2-{MODEL}, trained on membership-sans-rosch-v0-all, epoch2\n")
    header = ["Method"] + [c[0] for c in COLUMNS]
    sep = ["---"] * len(header)
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(sep) + " |")
    for r in table_rows:
        cells_display = [str(r[c[0]]) for c in COLUMNS]
        suf = fsx_label_suffix.get(r["num"], "")
        print(f"| {r['num']} {r['label']}{suf} | " + " | ".join(cells_display) + " |")

    print(f"\nCSVs:\n- [{long_csv.relative_to(REPO)}]({long_csv.relative_to(REPO)})\n- [{cells_csv.relative_to(REPO)}]({cells_csv.relative_to(REPO)})")

    _print_provenance(f"rosch_{MODEL}_{METRIC}")


def _print_provenance(tag: str):
    """Show what was actually consumed and which dups were collapsed."""
    n_used = len(PROVENANCE_USED)
    n_dups = len(PROVENANCE_DUPS)

    consumed_models = set()
    for r in PROVENANCE_USED:
        name = r["file"]
        after = name[len("scores_"):] if name.startswith("scores_") else name
        pfx = _extract_eval_prefix(after) or ""
        rest = after[len(pfx):]
        m = re.search(r"^(.+?)_rosch-[\w\-]+_test_log-odds", rest)
        ms = m.group(1) if m else rest
        ep_m = re.search(r"epoch(\d+)", ms)
        ep = int(ep_m.group(1)) if ep_m else None
        consumed_models.add((ep, ms))

    print(f"\nProvenance ({n_used} files consumed; "
          f"{n_dups} within-prefix dup groups collapsed by keeping newest):")
    print(f"  unique (epoch, model_short) pairs consumed: {len(consumed_models)}")
    epochs = sorted({e for e, _ in consumed_models if e is not None})
    if epochs:
        print(f"  epochs observed: {epochs}")
        for e in epochs:
            for _, ms in sorted([(ee, mm) for ee, mm in consumed_models if ee == e]):
                print(f"    epoch={e}  {ms}")
    none_models = sorted([mm for ee, mm in consumed_models if ee is None])
    if none_models:
        print("  (no-epoch models — likely base HF ckpts)")
        for ms in none_models:
            print(f"    {ms}")
    if n_dups > 0:
        used_csv = METRICS_DIR / f"{tag}_files_used.csv"
        dup_csv = METRICS_DIR / f"{tag}_dups_collapsed.csv"
        pd.DataFrame(PROVENANCE_USED).to_csv(used_csv, index=False)
        pd.DataFrame(PROVENANCE_DUPS).to_csv(dup_csv, index=False)
        print(f"  files-used CSV: [{used_csv.relative_to(REPO)}]({used_csv.relative_to(REPO)})")
        print(f"  dups-collapsed CSV: [{dup_csv.relative_to(REPO)}]({dup_csv.relative_to(REPO)})")


if __name__ == "__main__":
    main()
