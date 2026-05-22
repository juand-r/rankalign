#!/usr/bin/env python3
"""Build the IFEval OOD GenROC table per IMPORTANT-RESEARCH-PLAN.md.

Rows: 13 methods (0=Base, 1..9, 11, 12).
Cols: Raw, PMI self, PMI base, Neg self, Neg base.

Each cell = mean ± SE of GenROC × 100 across 20 OOD ifeval-prompts
(prompt_1..13, 15..21).

Sources:
- model = gemma-2-9b-it, epoch2, trained on ifeval-concat-all
  (the base HF model for row 0 has no '-delta...' suffix)
- score files in outputs/

Per the user's note: vlo-on-pref-only is a no-op, so settings #5 and #8
match either with or without the vallogodds flag.

Writes:
- metrics-from-scores/ifeval_ood_table_long.csv   (one row per (method, col, task))
- metrics-from-scores/ifeval_ood_table_cells.csv   (one row per (method, col))
- prints the markdown table to stdout
"""

from __future__ import annotations
import os
import re
import sys
import math
from pathlib import Path
from glob import glob
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

# OOD vs ID split per src/tasks/ifeval_concat.py:is_test_only_prompt
# - OOD = prompts 1..21 (prompt itself never appeared in training)
# - ID  = prompts 22+ (prompt appeared in training; completions are split
#         50/50 train/test, we eval on the held-out 50%)
DATA_DIR = REPO / "data" / "fixed-prompts-ifeval"


def _discover_prompts() -> list[int]:
    nums = []
    for f in DATA_DIR.iterdir():
        if f.name.startswith("gpt_ifeval_results_prompt_") and f.name.endswith(".jsonl"):
            s = f.name[len("gpt_ifeval_results_prompt_"):-len(".jsonl")]
            if s.isdigit():
                nums.append(int(s))
    return sorted(nums)


_ALL_PROMPTS = _discover_prompts()
SPLIT = (os.environ.get("IFEVAL_SPLIT", "ood")).lower()
if SPLIT == "ood":
    PROMPTS = [n for n in _ALL_PROMPTS if 1 <= n <= 21]
elif SPLIT == "id":
    PROMPTS = [n for n in _ALL_PROMPTS if n >= 22]
else:
    raise SystemExit(f"IFEVAL_SPLIT must be 'ood' or 'id', got {SPLIT!r}")
EVAL_TASKS = [f"ifeval-prompt_{n}" for n in PROMPTS]
N_EXPECTED = len(EVAL_TASKS)

# Which metric column from summarize_scores_file to read into the table.
# Supported: gen_roc (default), pearson, spearman, val_roc, val_acc
METRIC = os.environ.get("IFEVAL_METRIC", "gen_roc").lower()
SUPPORTED_METRICS = {"gen_roc", "pearson", "spearman", "val_roc", "val_acc"}
if METRIC not in SUPPORTED_METRICS:
    raise SystemExit(f"IFEVAL_METRIC must be one of {SUPPORTED_METRICS}, got {METRIC!r}")
# Pearson/spearman/val_acc/val_roc all live in [-1,1] or [0,1] — multiplier
# is still 100 to match the project convention.
METRIC_LABEL = {
    "gen_roc": "GenROC", "pearson": "Pearson(gen, val)",
    "spearman": "Spearman(gen, val)", "val_roc": "ValROC", "val_acc": "ValAcc",
}[METRIC]

# (column_label, eval_prefix_or_list, variant)
# For "Raw" the raw gen_score is the same across all eval prefixes (it is
# just log P_theta(y|x)); search every typicality-prefixed file as fallback.
COLUMNS = [
    ("Raw",      ["self-", "neg-", "basetyp-", "basetypneg-"], "raw"),
    ("PMI self", "self-",       "tc"),
    ("PMI base", "basetyp-",    "tc"),
    ("Neg self", "neg-",        "tc"),
    ("Neg base", "basetypneg-", "tc"),
]

# Patterns matching model_short for each method.
# model_short comes from basename of model dir with '--' -> '_'.
# Convention: 9b-it, delta0.15, epoch2, ifeval-concat-all, d2g, random, alpha1.0
# COMMON_PREFIX = v6-google_gemma-2-9b-it-delta0.15-epoch2_ifeval-concat-all_d2g_random_alpha1.0
# For row 0 (base HF), there is no -delta/epoch/task suffix.

CP = r"v6-google_gemma-2-9b-it-delta0\.15-epoch2_ifeval-concat-all_d2g_random_alpha1\.0"

# A method spec is dict with:
#   match: callable(model_short_without_merged) -> bool
#   label: pretty label
METHODS: list[dict] = [
    dict(
        num=0, label="Base",
        match=lambda s: s == "v6-google_gemma-2-9b-it",
    ),
    # 1 SFT-lo: sft, labelonly, no TC. fsx variant allowed (and flagged in label).
    # For 9b-it ifeval the only available ckpt is the fsx variant.
    dict(
        num=1, label="SFT labelonly 10%",
        # Accept both no-fsx and with-fsx; label below tracks fsx presence.
        match=lambda s: bool(re.fullmatch(
            rf"{CP}_full-completion_pref0\.0_nllv1\.0_nllg1\.0(_force-same-x)?_labelonly0\.1(_merged)?",
            s,
        )),
        fsx_label_match=lambda s: "_force-same-x_" in s,  # for table label
    ),
    # 2 RankAlign: pref-only, semi 0.1, no fsx, no TC.
    # Per inventory: in g-mode pref-only, semi is a no-op, so V2G baseline
    # (`full-completion` alone) is the same checkpoint.
    dict(
        num=2, label="RankAlign",
        match=lambda s: bool(re.fullmatch(
            rf"{CP}_full-completion(_semi0\.1)?(_merged)?",
            s,
        )),
    ),
    # 3 New + fsx [-TC]: comb, semi, vlo, fsx, no TC
    dict(
        num=3, label="New + fsx [-TC]",
        match=lambda s: bool(re.fullmatch(
            rf"{CP}_full-completion_nllv1\.0_nllg1\.0_force-same-x_vallogodds_semi0\.1(_merged)?",
            s,
        )),
    ),
    # 4 New + PMI + fsx: comb + vlo + fsx + tc-self + semi
    dict(
        num=4, label="New + PMI + fsx",
        match=lambda s: bool(re.fullmatch(
            rf"{CP}_tc-self_full-completion_nllv1\.0_nllg1\.0_force-same-x_vallogodds_semi0\.1(_merged)?",
            s,
        )),
    ),
    # 5 RankAlign + PMI + fsx [-NLL]: pref-only + fsx + tc-self + semi. vlo OK either way.
    dict(
        num=5, label="RA + PMI + fsx [-NLL]",
        match=lambda s: bool(re.fullmatch(
            rf"{CP}_tc-self_full-completion_force-same-x(_vallogodds)?_semi0\.1(_merged)?",
            s,
        )),
    ),
    # 6 RankAlign + PMI [+TC]: pref-only + tc-self + semi, no fsx, no vlo, no NLL.
    dict(
        num=6, label="RA + PMI [+TC]",
        match=lambda s: bool(re.fullmatch(
            rf"{CP}_tc-self_full-completion(_vallogodds)?_semi0\.1(_merged)?",
            s,
        )),
    ),
    # 7 New + NegTC + fsx: comb + vlo + fsx + tc-neg + semi
    dict(
        num=7, label="New + NegTC + fsx",
        match=lambda s: bool(re.fullmatch(
            rf"{CP}_tc-neg_full-completion_nllv1\.0_nllg1\.0_force-same-x_vallogodds_semi0\.1(_merged)?",
            s,
        )),
    ),
    # 8 RankAlign + NegTC + fsx [-NLL]: pref-only + fsx + tc-neg + semi (vlo OK)
    dict(
        num=8, label="RA + NegTC + fsx [-NLL]",
        match=lambda s: bool(re.fullmatch(
            rf"{CP}_tc-neg_full-completion_force-same-x(_vallogodds)?_semi0\.1(_merged)?",
            s,
        )),
    ),
    # 9 RankAlign + NegTC [+TC]: pref-only + tc-neg + semi, no fsx
    dict(
        num=9, label="RA + NegTC [+TC]",
        match=lambda s: bool(re.fullmatch(
            rf"{CP}_tc-neg_full-completion(_vallogodds)?_semi0\.1(_merged)?",
            s,
        )),
    ),
    # 11 New + PMI [-fsx]: comb + vlo + tc-self + semi, NO fsx
    dict(
        num=11, label="New + PMI [-fsx]",
        match=lambda s: bool(re.fullmatch(
            rf"{CP}_tc-self_full-completion_nllv1\.0_nllg1\.0_vallogodds_semi0\.1(_merged)?",
            s,
        )),
    ),
    # 12 New + NegTC [-fsx]
    dict(
        num=12, label="New + NegTC [-fsx]",
        match=lambda s: bool(re.fullmatch(
            rf"{CP}_tc-neg_full-completion_nllv1\.0_nllg1\.0_vallogodds_semi0\.1(_merged)?",
            s,
        )),
    ),
]

# Columns that are N/A for each row (---). From the user's table:
NA_COLS = {
    # method_num -> set of column labels that are N/A
    4: {"Neg self", "Neg base"},
    5: {"Neg self", "Neg base"},
    6: {"Neg self", "Neg base"},
    11: {"Neg self", "Neg base"},
    7: {"PMI self", "PMI base"},
    8: {"PMI self", "PMI base"},
    9: {"PMI self", "PMI base"},
    12: {"PMI self", "PMI base"},
}


PREFIXES_ALL = ["self-", "basetyp-", "neg-", "basetypneg-", ""]
# Most specific first so `basetyp-` doesn't match `basetypneg-`.
_EVAL_PREFIXES_ORDERED = ("basetypneg-", "basetyp-", "self-", "neg-")


def _extract_eval_prefix(filename: str) -> str:
    after = filename[len("scores_"):] if filename.startswith("scores_") else filename
    for p in _EVAL_PREFIXES_ORDERED:
        if after.startswith(p):
            return p
    return ""


def find_score_files(method: dict, eval_prefix: str | list[str]) -> dict[str, list[Path]]:
    """For method × eval_prefix, find score files keyed by task.

    eval_prefix can be a single prefix string (e.g. 'self-') or a list of
    prefixes to try (used for the Raw column, where any prefix's file
    contains the same raw gen_score).

    Filename pattern:
        scores_{eval_prefix}{model_short}_{task}_test_log-odds{_tc?}_<date>.csv
    """
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
                # Empty prefix: skip any file whose first token after "scores_"
                # matches one of the known typicality prefixes.
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


def cell_value(method: dict, eval_prefix: str, variant: str):
    """Returns dict with: mean, se, n, per_task_values, files."""
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

    # Row order the user wants (explicit).
    ROW_ORDER = [0, 1, 2, 3, 4, 5, 6, 11, 7, 8, 9, 12]
    methods_by_num = {m["num"]: m for m in METHODS}
    ordered = [methods_by_num[n] for n in ROW_ORDER]

    # Track whether method #1 ended up matching fsx variants (for label).
    fsx_label_suffix: dict[int, str] = {}

    for m in ordered:
        cells = {}
        # For label-mutating methods, detect whether the matched files are fsx.
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
            if seen_fsx:
                fsx_label_suffix[m["num"]] = " +fsx"
            else:
                fsx_label_suffix[m["num"]] = ""
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
                    task=task, gen_roc=val, file=fn,
                ))
        table_rows.append(dict(num=m["num"], label=m["label"], **cells))

    # Write CSVs
    suffix = "ood" if SPLIT == "ood" else "id"
    long_csv = METRICS_DIR / f"ifeval_{suffix}_{METRIC}_table_long.csv"
    cells_csv = METRICS_DIR / f"ifeval_{suffix}_{METRIC}_table_cells.csv"
    pd.DataFrame(long_rows).to_csv(long_csv, index=False)
    pd.DataFrame(cell_rows).to_csv(cells_csv, index=False)

    # Print markdown table
    split_human = "OOD" if SPLIT == "ood" else "in-domain (ID)"
    rng_desc = "ifeval-prompt_1..13, 15..21" if SPLIT == "ood" else f"ifeval-prompt_{PROMPTS[0]}..{PROMPTS[-1]} (held-out 50% of completions)"
    print(f"\nIFEval {split_human} {METRIC_LABEL} × 100 — mean ± SE across {N_EXPECTED} prompts ({rng_desc})")
    print(f"Model: gemma-2-9b-it, trained on ifeval-concat-all, epoch2\n")
    header = ["Method"] + [c[0] for c in COLUMNS]
    sep = ["---"] * len(header)
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(sep) + " |")
    for r in table_rows:
        cells_display = [str(r[c[0]]) for c in COLUMNS]
        suf = fsx_label_suffix.get(r["num"], "")
        print(f"| {r['num']} {r['label']}{suf} | " + " | ".join(cells_display) + " |")

    print(f"\nCSVs:\n- [{long_csv.relative_to(REPO)}]({long_csv.relative_to(REPO)})\n- [{cells_csv.relative_to(REPO)}]({cells_csv.relative_to(REPO)})")

    _print_provenance(f"ifeval_{SPLIT}_{METRIC}")


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
        m = re.search(r"^(.+?)_ifeval-prompt_\d+_test_log-odds", rest)
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
