#!/usr/bin/env python3
"""Build the rosch OOD table for v7 / fix1 trained models.

Rosch is all-OOD: the fine-tuned models were trained on
`membership-sans-rosch-v0-all`, i.e. membership data with all rosch
categories held out. So every rosch category is unseen at training time.

This is the **v7** sibling of `_build_rosch_table.py`. It consumes
score CSVs produced by evaluations of models trained with
`scripts/ranking_loss_ref_fix.py` (which prefixes model dirs with `v7-`,
appends `--fix1`, allows variable `--delta-bins`-derived deltas, and
introduces the `--ppd` token for `--per-prompt-delta` runs and
`--shape-budget-mode global` semantics).

Rows: 12 methods (0=Base, 1..9, 11, 12), same numbering as IRP §1.
Cols: Raw, PMI self, PMI base, Neg self, Neg base.

NA structure follows the v7 dispatcher (`scripts/_overnight_launch.sh`)
which only runs ONE TC variant per setting at eval time:
  s1, s2, s3, s4, s5, s6, s11 -> self-tc eval only -> Neg* are NA
  s7, s8, s9, s12             -> neg-tc eval only  -> PMI* are NA

Env vars:
- ROSCH_MODEL  ∈ {2b, 2b-it, 9b-it}   default 9b-it
- ROSCH_METRIC ∈ {gen_roc, pearson, spearman, val_roc, val_acc}
                                       default gen_roc

Writes:
- metrics-from-scores/rosch_v7_{model}_{metric}_table_long.csv
- metrics-from-scores/rosch_v7_{model}_{metric}_table_cells.csv
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
sys.path.insert(0, str(REPO / "src"))
from summarize_scores_file import load_scores, compute_all_metrics  # noqa: E402
from checkpoint_name_parser import matches_v7_setting  # noqa: E402
import hashlib  # noqa: E402

OUT_DIR = REPO / "outputs"
SEARCH_DIRS = [
    OUT_DIR,
    Path("/datastor2/jdr/rankalign/outputs"),
    Path("/datastor2/jdr/rankalign/outputs-rerun-wandb"),
]
METRICS_DIR = REPO / "metrics-from-scores"
METRICS_DIR.mkdir(exist_ok=True)

# Mirror the OLD md5-hash truncation logic from src/tasks/common.py
# (commit 8b4c4fea, 2026-05-24). That branch wrote
# `raw[:151] + '_' + md5(raw)[:8]` for over-160-char basenames. Some
# CSVs from the brief window between 8b4c4fea and a42558a9 (when the
# abbreviation fallback replaced hash truncation as the primary
# overflow strategy) are still on disk in this form. We resolve them
# back to their full basename via a model-dir index, then run the
# structural matcher against the FULL form.
_MODEL_SHORT_MAX_LEN = 160
MODEL_DIRS = [
    Path("/datastor2/jdr/rankalign/models2"),
    Path("/datastor1/jdr/gv-gap/rankalign/models"),
    Path("/datastor1/jdr/gv-gap/rankalign/models2"),
]


def _md5_truncate(raw: str) -> str:
    """Reproduce the md5-hash truncation form for a too-long raw basename."""
    if len(raw) <= _MODEL_SHORT_MAX_LEN:
        return raw
    h = hashlib.md5(raw.encode()).hexdigest()[:8]
    return raw[:_MODEL_SHORT_MAX_LEN - 9] + '_' + h


_DIR_INDEX: dict[str, str] | None = None


def _build_dir_index() -> dict[str, str]:
    """Map the two known long-name truncation forms back to the full
    on-disk basename for every v7-* dir. Used to resolve a CSV's
    model_short (which may be md5-hash-truncated, abbreviated, or full)
    to the canonical full form for structural matching.
    """
    idx: dict[str, str] = {}
    for d in MODEL_DIRS:
        if not d.is_dir():
            continue
        for sub in d.glob("v7-*"):
            full = sub.name
            # Direct (no truncation needed)
            idx[full] = full
            # Md5 truncation form (legacy 8b4c4fea behavior)
            idx[_md5_truncate(full)] = full
    return idx


def _resolve_full_basename(model_short: str) -> str:
    """Return FULL on-disk basename for a model_short (handles legacy
    md5-hash-truncated forms). Falls back to the input on cache miss —
    structural matcher handles abs-path-embedded form (A) and
    abbreviated form (C) directly."""
    global _DIR_INDEX
    if _DIR_INDEX is None:
        _DIR_INDEX = _build_dir_index()
    return _DIR_INDEX.get(model_short, model_short)

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

# Model registry: friendly key -> (model_short used in v7-<...> trained-model filenames,
# base-model token used in v6-<...> base filenames). To support a new model, add ONE line
# here — nothing else below is model-specific.
MODEL_REGISTRY = {
    "2b":         ("gemma-2-2b",    "v6-google_gemma-2-2b"),
    "2b-it":      ("gemma-2-2b-it", "v6-google_gemma-2-2b-it"),
    "9b-it":      ("gemma-2-9b-it", "v6-google_gemma-2-9b-it"),
    "qwen3.5-9b": ("Qwen3.5-9B",    "v6-Qwen_Qwen3.5-9B"),
}
MODEL = os.environ.get("ROSCH_MODEL", "9b-it").lower()
if MODEL not in MODEL_REGISTRY:
    raise SystemExit(f"ROSCH_MODEL must be one of {sorted(MODEL_REGISTRY)}, got {MODEL!r}")
MODEL_SHORT, BASE_TOKEN = MODEL_REGISTRY[MODEL]

METRIC = os.environ.get("ROSCH_METRIC", "gen_roc").lower()
SUPPORTED_METRICS = {"gen_roc", "pearson", "spearman", "val_roc", "val_acc"}
if METRIC not in SUPPORTED_METRICS:
    raise SystemExit(f"ROSCH_METRIC must be one of {SUPPORTED_METRICS}, got {METRIC!r}")
METRIC_LABEL = {
    "gen_roc": "GenROC", "pearson": "Pearson(gen, val)",
    "spearman": "Spearman(gen, val)", "val_roc": "ValROC", "val_acc": "ValAcc",
}[METRIC]

# Base HF model string. Note: BASE models still get the legacy "v6-" prefix
# in CSV filenames (eval_by_claude.py line ~1417 / src/tasks/common.py
# `build_model_short`) regardless of fix1, because that prefix is hard-coded
# for HF-style "org/name" inputs.
BASE_HF = BASE_TOKEN

# Trained-model identification is now structural (parse → field-compare),
# handled by `matches_v7_setting()` from checkpoint_name_parser. This works
# uniformly across the three on-disk filename formats:
#   (A) legacy abs-path-embedded: `v6-_datastor2_..._v7-google--gemma-...-fix1`
#   (B) un-abbreviated:           `v7-google--gemma-2-9b-it-delta2.69-...-fix1[_merged]`
#   (C) abbreviated (>160ch):     `v7-gemma-2-9b-it-d2.69-e2-...-sm0.1-fix1`
TASK_SEG = "membership-sans-rosch-v0-all"
GEMMA_MODEL = MODEL_SHORT  # model_short from MODEL_REGISTRY (what parse_checkpoint_name returns)

COLUMNS = [
    ("Raw",      ["self-", "neg-", "basetyp-", "basetypneg-"], "raw"),
    ("PMI self", "self-",       "tc"),
    ("PMI base", "basetyp-",    "tc"),
    ("Neg self", "neg-",        "tc"),
    ("Neg base", "basetypneg-", "tc"),
]

# Methods follow the same numbering as IRP §1. v7-specific tweaks:
#   - All trained methods end in `_fix1[_merged]` (the FIX_TAIL).
#   - Methods with fsx (s3, s4, s5, s7, s8) include `_ppd` after
#     `_force-same-x` because `_overnight_launch.sh` always passes
#     `--per-prompt-delta` whenever `--force-same-x` is set.
#   - s5 and s8 (RA + fsx + TC) drop the `(_vallogodds)?` optionality:
#     v7 explicitly does NOT pass `--validator-log-odds` for these
#     (was a v6-era bug that's been fixed; see _overnight_launch.sh
#     case "s5"/case "s8").
#   - 2b-it cells run for ~5h walltime; some may save only epoch 0/1
#     before the walltime kills them. CP allows any of {0,1,2}.
def _setting_match(**expected):
    """Build a match-lambda for a v7 setting. Resolves md5-hash-truncated
    model_shorts back to their full basename via DIR_INDEX before
    structural matching (handles the brief 8b4c4fea-era CSVs)."""
    def m(s):
        full = _resolve_full_basename(s)
        return matches_v7_setting(
            full, model=GEMMA_MODEL, task_segment=TASK_SEG, **expected,
        )
    return m


# Each METHODS entry declares ONLY the flags that distinguish that setting.
# Defaults in matches_v7_setting (pref=1.0, nll_v=0.0, nll_g=0.0, vlo/fsx/
# ppd/cft=False, tc/semi/labelonly=UNSET, fix1=True) cover the rest.
METHODS: list[dict] = [
    dict(
        num=0, label="Base",
        match=lambda s: s == BASE_HF,
    ),
    # 1 SFT-lo: sft + labelonly. No fsx, no tc, no vlo.
    dict(num=1, label="SFT labelonly 10%",
         match=_setting_match(pref=0.0, nll_v=1.0, nll_g=1.0, labelonly=0.1)),
    # 2 RankAlign: pref-only + semi, no fsx, no TC, no vlo.
    dict(num=2, label="RankAlign",
         match=_setting_match(semi=0.1)),
    # 3 New + fsx [-TC]: comb + fsx (+ ppd) + vlo, no TC.
    dict(num=3, label="New + fsx [-TC]",
         match=_setting_match(nll_v=1.0, nll_g=1.0, force_same_x=True,
                              ppd=True, vallogodds=True, semi=0.1)),
    # 4 New + PMI + fsx: comb + fsx (+ ppd) + tc-self + vlo.
    dict(num=4, label="New + PMI + fsx",
         match=_setting_match(tc='self', nll_v=1.0, nll_g=1.0,
                              force_same_x=True, ppd=True, vallogodds=True,
                              semi=0.1)),
    # 5 RA + PMI + fsx [-NLL]: pref-only + fsx (+ ppd) + tc-self, NO vlo.
    dict(num=5, label="RA + PMI + fsx [-NLL]",
         match=_setting_match(tc='self', force_same_x=True, ppd=True,
                              semi=0.1)),
    # 6 RA + PMI [+TC]: pref-only + tc-self, no fsx, no NLL, no vlo.
    dict(num=6, label="RA + PMI [+TC]",
         match=_setting_match(tc='self', semi=0.1)),
    # 7 New + NegTC + fsx: comb + fsx (+ ppd) + tc-neg + vlo.
    dict(num=7, label="New + NegTC + fsx",
         match=_setting_match(tc='neg', nll_v=1.0, nll_g=1.0,
                              force_same_x=True, ppd=True, vallogodds=True,
                              semi=0.1)),
    # 8 RA + NegTC + fsx [-NLL]: pref-only + fsx (+ ppd) + tc-neg, NO vlo.
    dict(num=8, label="RA + NegTC + fsx [-NLL]",
         match=_setting_match(tc='neg', force_same_x=True, ppd=True,
                              semi=0.1)),
    # 9 RA + NegTC [+TC]: pref-only + tc-neg, no fsx, no NLL, no vlo.
    dict(num=9, label="RA + NegTC [+TC]",
         match=_setting_match(tc='neg', semi=0.1)),
    # 11 New + PMI [-fsx]: comb + tc-self + vlo, no fsx, no ppd.
    dict(num=11, label="New + PMI [-fsx]",
         match=_setting_match(tc='self', nll_v=1.0, nll_g=1.0,
                              vallogodds=True, semi=0.1)),
    # 12 New + NegTC [-fsx]: comb + tc-neg + vlo, no fsx, no ppd.
    dict(num=12, label="New + NegTC [-fsx]",
         match=_setting_match(tc='neg', nll_v=1.0, nll_g=1.0,
                              vallogodds=True, semi=0.1)),
    # 13 SFT + CFT: same flags as s1 plus --cft (consistency-ft).
    dict(num=13, label="SFT + CFT",
         match=_setting_match(pref=0.0, nll_v=1.0, nll_g=1.0,
                              cft=True, labelonly=0.1)),
]

# v7 NA structure: settings trained WITH a specific TC objective only
# produce that one prefix's CSVs. Settings trained without TC (s1/s2/s3)
# get BOTH self-tc and neg-tc evals (matching v6 launcher policy), so
# they have no NA columns.
#
# Updated 2026-05-24 19:55: dropped rows 1/2/3 from NA_COLS to match the
# fixed _overnight_launch.sh dispatcher (TC_EVAL_LIST="self neg" for
# s1/s2/s3). Builder will now read both basetyp-self and basetypneg
# CSVs for those rows.
NA_COLS = {
    4:  {"Neg self", "Neg base"},
    5:  {"Neg self", "Neg base"},
    6:  {"Neg self", "Neg base"},
    11: {"Neg self", "Neg base"},
    7:  {"PMI self", "PMI base"},
    8:  {"PMI self", "PMI base"},
    9:  {"PMI self", "PMI base"},
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
    seen_basenames: set[str] = set()
    for pfx in prefixes:
        pattern = f"scores_{pfx}*_test_log-odds*.csv"
        for d in SEARCH_DIRS:
            if not d.is_dir():
                continue
            for p in d.glob(pattern):
                name = p.name
                if name in seen_basenames:
                    continue
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
                    seen_basenames.add(name)
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

    ROW_ORDER = [0, 1, 2, 3, 4, 5, 6, 11, 7, 8, 9, 12, 13]
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

    long_csv = METRICS_DIR / f"rosch_v7_{MODEL}_{METRIC}_table_long.csv"
    cells_csv = METRICS_DIR / f"rosch_v7_{MODEL}_{METRIC}_table_cells.csv"
    pd.DataFrame(long_rows).to_csv(long_csv, index=False)
    pd.DataFrame(cell_rows).to_csv(cells_csv, index=False)

    print(f"\nRosch (all OOD) [v7/fix1] {METRIC_LABEL} × 100 — mean ± SE across {N_EXPECTED} categories")
    print(f"Model: {MODEL_SHORT}, trained on membership-sans-rosch-v0-all,")
    print(f"epoch picked by `ls -dt | head -1` from epoch[012] glob.\n")
    header = ["Method"] + [c[0] for c in COLUMNS]
    sep = ["---"] * len(header)
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(sep) + " |")
    for r in table_rows:
        cells_display = [str(r[c[0]]) for c in COLUMNS]
        suf = fsx_label_suffix.get(r["num"], "")
        print(f"| {r['num']} {r['label']}{suf} | " + " | ".join(cells_display) + " |")

    print(f"\nCSVs:\n- [{long_csv.relative_to(REPO)}]({long_csv.relative_to(REPO)})\n- [{cells_csv.relative_to(REPO)}]({cells_csv.relative_to(REPO)})")

    _print_provenance(f"rosch_v7_{MODEL}_{METRIC}")


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
