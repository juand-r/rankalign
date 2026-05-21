#!/usr/bin/env python3
"""Build the humaneval table for gemma-4-31B-it on humaneval-v2.1correct-multi.

The gemma-4 humaneval ckpts live across several `outputs_gemma4_*` dirs
and have a different model_short convention than the gemma-2 tasks:

  - Separators between dataset/loss flags are `--` (double hyphen) rather
    than `_`.
  - Some model_shorts are truncated by the eval pipeline at a fixed
    length and end with an `_<8hexhash>` suffix.
  - A few have a `_workspace_models_g4it_v6-google--...` path-prefix
    artifact baked in.

To handle the truncation, method matching is done via a *prefix-on-
flag-fragments* approach: we normalize the model_short (drop the optional
path artifact, optionally strip the hash suffix) and check that the
expected set of flag tokens are all present and the disallowed ones are
absent. This is more permissive than the fullmatch-regex approach used
for the gemma-2 tables, but it is required because the hashed tail hides
the trailing flags.

Eval tasks are inferred from disk: every `humaneval-v2.1correct-multi-
humaneval_<id>` task encountered across the search dirs.

Rows: 12 methods (0=Base, 1..9, 11, 12), same numbering as IFEval.
Cols: Raw, PMI self, PMI base, Neg self, Neg base.

Env vars:
- HUMANEVAL_METRIC ∈ {gen_roc, pearson, spearman, val_roc, val_acc}
                                          default gen_roc

Writes:
- metrics-from-scores/humaneval_v2.1correct-multi_g4-31B-it_{metric}_table_long.csv
- metrics-from-scores/humaneval_v2.1correct-multi_g4-31B-it_{metric}_table_cells.csv
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

SEARCH_DIRS = [
    REPO / "outputs_gemma4_from_pod",
    REPO / "outputs_gemma4_3epoch_e2",
    REPO / "outputs_gemma4_e0_firstrun",
    REPO / "outputs_gemma4_from_pod_epoch0",
    REPO / "outputs",
]
METRICS_DIR = REPO / "metrics-from-scores"
METRICS_DIR.mkdir(exist_ok=True)

TRAIN_TASK = "humaneval-v2.1correct-multi"          # appears in eval-task name
TRAIN_DATA_TAG = "humaneval-v2.1correct-multi-all"   # appears in model_short

METRIC = os.environ.get("HUMANEVAL_METRIC", "gen_roc").lower()
SUPPORTED_METRICS = {"gen_roc", "pearson", "spearman", "val_roc", "val_acc"}
if METRIC not in SUPPORTED_METRICS:
    raise SystemExit(f"HUMANEVAL_METRIC must be one of {SUPPORTED_METRICS}, got {METRIC!r}")
METRIC_LABEL = {
    "gen_roc": "GenROC", "pearson": "Pearson(gen, val)",
    "spearman": "Spearman(gen, val)", "val_roc": "ValROC", "val_acc": "ValAcc",
}[METRIC]


def _discover_eval_tasks() -> list[str]:
    seen: set[str] = set()
    for d in SEARCH_DIRS:
        if not d.is_dir():
            continue
        for p in d.glob(f"scores_*_{TRAIN_TASK}-humaneval_*_test_log-odds*.csv"):
            m = re.search(rf"_({re.escape(TRAIN_TASK)}-humaneval_\d+)_test_log-odds", p.name)
            if m:
                seen.add(m.group(1))
    return sorted(seen, key=lambda s: int(s.rsplit("_", 1)[-1]))


EVAL_TASKS = _discover_eval_tasks()
N_EXPECTED = len(EVAL_TASKS)
if N_EXPECTED == 0:
    raise SystemExit(f"No eval tasks discovered for {TRAIN_TASK!r}.")


COLUMNS = [
    ("Raw",      ["self-", "neg-", "basetyp-", "basetypneg-"], "raw"),
    ("PMI self", "self-",       "tc"),
    ("PMI base", "basetyp-",    "tc"),
    ("Neg self", "neg-",        "tc"),
    ("Neg base", "basetypneg-", "tc"),
]

# Base model_short — the un-trained gemma-4-31B-it (no delta/epoch suffix).
BASE_PATTERN = re.compile(r"^v6-google[_\-]+gemma-4-31B-it$")


# Helpers for token-based method matching.
# Every flag token any of the 12 methods cares about. Used to recover from
# mid-token hash truncation: if the last observed token is a strict prefix
# of one (and only one) known flag, we promote it to the full flag.
_KNOWN_FLAGS = [
    "full-completion", "pref0.0", "nllv1.0", "nllg1.0",
    "tc-self", "tc-neg", "force-same-x", "vallogodds",
    "semi0.1", "labelonly0.1",
]
# Flags whose canonical position is BEFORE `full-completion` in the
# model_short. If `full-completion` is observed and one of these flags is
# missing from the observed flag list, the flag is *definitively* absent
# (it cannot be hidden by the trailing hash truncation).
_EARLY_FLAGS = {"tc-self", "tc-neg"}


def _norm_tokens(s: str) -> tuple[str, list[str], bool]:
    """Return (model_id, tokens, was_truncated).

    model_id   = "gemma-4-31B-it-delta0.15-epoch2" or similar.
    tokens     = list of flag tokens between the dataset tag and the trailing
                 hash (if any).  E.g. ["full-completion", "tc-self",
                 "nllv1.0", "nllg1.0", "force-same-x", "vallogodds",
                 "semi0.1"]
    was_truncated = True if the model_short ends with the 8-hex-hash suffix
                 (which means trailing tokens may be missing).
    """
    # Drop optional path artifact prefix.
    s2 = s
    if s2.startswith("v6-_workspace_models_g4it_"):
        s2 = s2[len("v6-_workspace_models_g4it_"):]
    # Normalize `--` and `_` as token separators.
    parts = re.split(r"-{2}|_", s2)
    # Strip leading "v6-google" if present (parts[0]=='v6', parts[1]=='google')
    while parts and parts[0] in ("v6", "google", ""):
        parts.pop(0)
    if not parts:
        return ("", [], False)
    # First parts compose the model_id up to the training-task tag.
    # Find the training-data tag token index.
    try:
        tag_idx = parts.index("d2g")
    except ValueError:
        # No d2g token => probably the un-trained base. Return as base.
        return ("-".join(parts), [], False)
    # Everything before "d2g" is model/dataset; everything after "d2g random
    # alpha1.0" is flags.
    flag_start = tag_idx + 3 if tag_idx + 3 <= len(parts) else tag_idx + 1
    flags = parts[flag_start:]
    # Hash truncation: detect last token == 8 hex chars (then the previous
    # token is the partial mid-word stem before truncation).
    was_trunc = bool(flags) and bool(re.fullmatch(r"[0-9a-f]{8}", flags[-1]))
    if was_trunc:
        flags = flags[:-1]
        # If the new last token is a strict prefix of exactly one known flag,
        # promote it to the full flag (e.g. "force-sa" -> "force-same-x").
        if flags:
            stem = flags[-1]
            promotions = [f for f in _KNOWN_FLAGS if f != stem and f.startswith(stem)]
            if len(promotions) == 1:
                flags = flags[:-1] + [promotions[0]]
    return ("", flags, was_trunc)


def _match_method(model_short: str, *,
                  require: set[str], forbid: set[str] = frozenset(),
                  allow_extra: set[str] = frozenset()) -> bool:
    """Token-based match. allow_extra are flags that may appear (e.g. vlo
    no-ops). All `require` must be present; none of `forbid` may be."""
    # Reject the un-trained base HF model (it has no d2g token, no
    # training-data tag, etc.) — only Method 0's matcher should claim it.
    if BASE_PATTERN.match(model_short):
        return False
    _, flags, was_trunc = _norm_tokens(model_short)
    if not flags and not was_trunc:
        # Token-normalization couldn't find a d2g token; treat as non-match.
        return False
    flagset = set(flags)
    if flagset & forbid:
        return False
    # `full-completion` is implicit on all trained models; never counts.
    extras = flagset - require - allow_extra - {"full-completion"}
    if extras:
        return False
    missing = require - flagset
    if missing:
        if not was_trunc:
            return False
        # If the model_short was truncated, the trailing chopped tail might
        # contain the missing flags — but only for flags whose canonical
        # position is AFTER `full-completion`. EARLY_FLAGS (tc-self/tc-neg)
        # appear before `full-completion`, so their absence is definitive
        # whenever full-completion is observed.
        if "full-completion" in flagset and (missing & _EARLY_FLAGS):
            return False
    return True


# Method definitions. The flag tokens we care about are:
#   "full-completion" (always present in trained models — implicit)
#   "nllv1.0" "nllg1.0"           → comb (combined loss)
#   "pref0.0"                       → sft (no preference loss)
#   "tc-self" / "tc-neg"           → typicality correction
#   "force-same-x"                 → fsx
#   "vallogodds"                   → vlo
#   "semi0.1"                      → semi-supervised 10% labeled
#   "labelonly0.1"                 → labeled-only 10%
METHODS: list[dict] = [
    dict(num=0, label="Base",
         match=lambda s: bool(BASE_PATTERN.match(s))),
    # 1 SFT-lo: pref=0 + nllv + nllg + labelonly0.1, optional fsx.
    dict(num=1, label="SFT labelonly 10%",
         match=lambda s: _match_method(
             s,
             require={"pref0.0", "nllv1.0", "nllg1.0", "labelonly0.1"},
             forbid={"semi0.1", "tc-self", "tc-neg", "vallogodds"},
             allow_extra={"force-same-x"}),
         fsx_label_match=lambda model_short: "force-same-x" in (_norm_tokens(model_short)[1])),
    # 2 RankAlign: pref-only (no nllv/nllg) + semi optional, no TC, no fsx, no vlo.
    dict(num=2, label="RankAlign",
         match=lambda s: _match_method(
             s,
             require=set(),
             forbid={"nllv1.0", "nllg1.0", "pref0.0", "tc-self", "tc-neg",
                     "force-same-x", "vallogodds", "labelonly0.1"},
             allow_extra={"semi0.1"})),
    # 3 New + fsx [-TC]: comb + fsx + vlo + semi, no TC.
    dict(num=3, label="New + fsx [-TC]",
         match=lambda s: _match_method(
             s,
             require={"nllv1.0", "nllg1.0", "force-same-x", "vallogodds", "semi0.1"},
             forbid={"tc-self", "tc-neg", "pref0.0", "labelonly0.1"})),
    # 4 New + PMI + fsx: comb + tc-self + fsx + vlo + semi.
    dict(num=4, label="New + PMI + fsx",
         match=lambda s: _match_method(
             s,
             require={"nllv1.0", "nllg1.0", "tc-self", "force-same-x", "vallogodds", "semi0.1"},
             forbid={"tc-neg", "pref0.0", "labelonly0.1"})),
    # 5 RA + PMI + fsx [-NLL]: pref-only + tc-self + fsx + semi, vlo optional.
    dict(num=5, label="RA + PMI + fsx [-NLL]",
         match=lambda s: _match_method(
             s,
             require={"tc-self", "force-same-x", "semi0.1"},
             forbid={"nllv1.0", "nllg1.0", "tc-neg", "pref0.0", "labelonly0.1"},
             allow_extra={"vallogodds"})),
    # 6 RA + PMI [+TC]: pref-only + tc-self + semi (no fsx, no NLL).
    dict(num=6, label="RA + PMI [+TC]",
         match=lambda s: _match_method(
             s,
             require={"tc-self", "semi0.1"},
             forbid={"nllv1.0", "nllg1.0", "tc-neg", "pref0.0", "labelonly0.1",
                     "force-same-x"},
             allow_extra={"vallogodds"})),
    # 7 New + NegTC + fsx: comb + tc-neg + fsx + vlo + semi.
    dict(num=7, label="New + NegTC + fsx",
         match=lambda s: _match_method(
             s,
             require={"nllv1.0", "nllg1.0", "tc-neg", "force-same-x", "vallogodds", "semi0.1"},
             forbid={"tc-self", "pref0.0", "labelonly0.1"})),
    # 8 RA + NegTC + fsx [-NLL]
    dict(num=8, label="RA + NegTC + fsx [-NLL]",
         match=lambda s: _match_method(
             s,
             require={"tc-neg", "force-same-x", "semi0.1"},
             forbid={"nllv1.0", "nllg1.0", "tc-self", "pref0.0", "labelonly0.1"},
             allow_extra={"vallogodds"})),
    # 9 RA + NegTC [+TC]
    dict(num=9, label="RA + NegTC [+TC]",
         match=lambda s: _match_method(
             s,
             require={"tc-neg", "semi0.1"},
             forbid={"nllv1.0", "nllg1.0", "tc-self", "pref0.0", "labelonly0.1",
                     "force-same-x"},
             allow_extra={"vallogodds"})),
    # 11 New + PMI [-fsx]: comb + tc-self + vlo + semi, NO fsx.
    dict(num=11, label="New + PMI [-fsx]",
         match=lambda s: _match_method(
             s,
             require={"nllv1.0", "nllg1.0", "tc-self", "vallogodds", "semi0.1"},
             forbid={"force-same-x", "tc-neg", "pref0.0", "labelonly0.1"})),
    # 12 New + NegTC [-fsx]
    dict(num=12, label="New + NegTC [-fsx]",
         match=lambda s: _match_method(
             s,
             require={"nllv1.0", "nllg1.0", "tc-neg", "vallogodds", "semi0.1"},
             forbid={"force-same-x", "tc-self", "pref0.0", "labelonly0.1"})),
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


def find_score_files(method: dict, eval_prefix) -> dict[str, list[Path]]:
    if isinstance(eval_prefix, str):
        prefixes = [eval_prefix]
    else:
        prefixes = list(eval_prefix)
    matches: dict[str, list[Path]] = {t: [] for t in EVAL_TASKS}
    for d in SEARCH_DIRS:
        if not d.is_dir():
            continue
        for pfx in prefixes:
            pattern = f"scores_{pfx}*_test_log-odds*.csv"
            for p in d.glob(pattern):
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
        path = sorted(candidates)[-1]
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
    long_rows, cell_rows, table_rows = [], [], []
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
                        name = p.name[len("scores_"):]
                        for prefix in ("self-", "neg-", "basetyp-", "basetypneg-", ""):
                            if name.startswith(prefix):
                                rest = name[len(prefix):]
                                break
                        ms_only = rest.split(f"_{EVAL_TASKS[0].rsplit('_', 1)[0]}_", 1)[0]
                        # The above is a hack — fall back to substring check.
                        if m["fsx_label_match"](rest):
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
                    column=col_label, mean=None, se=None, n=0, note="NA per template"))
                continue
            c = cell_value(m, eval_prefix, variant)
            cells[col_label] = fmt_cell(c, N_EXPECTED)
            cell_rows.append(dict(
                method_num=m["num"], method=m["label"],
                column=col_label, mean=c["mean"], se=c["se"], n=c["n"],
                note=("OK" if c["n"] == N_EXPECTED else f"missing {N_EXPECTED - c['n']}/{N_EXPECTED}")))
            for task, val, fn in c["vals"]:
                long_rows.append(dict(
                    method_num=m["num"], method=m["label"],
                    column=col_label, eval_prefix=eval_prefix, variant=variant,
                    task=task, value=val, file=fn))
        table_rows.append(dict(num=m["num"], label=m["label"], **cells))

    tag = "humaneval_v2.1correct-multi_g4-31B-it"
    long_csv = METRICS_DIR / f"{tag}_{METRIC}_table_long.csv"
    cells_csv = METRICS_DIR / f"{tag}_{METRIC}_table_cells.csv"
    pd.DataFrame(long_rows).to_csv(long_csv, index=False)
    pd.DataFrame(cell_rows).to_csv(cells_csv, index=False)

    print(f"\nHumaneval-v2.1correct-multi {METRIC_LABEL} × 100 — mean ± SE across {N_EXPECTED} problems")
    print(f"Model: gemma-4-31B-it, trained on {TRAIN_DATA_TAG}, epoch2\n")
    header = ["Method"] + [c[0] for c in COLUMNS]
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")
    for r in table_rows:
        cells_display = [str(r[c[0]]) for c in COLUMNS]
        suf = fsx_label_suffix.get(r["num"], "")
        print(f"| {r['num']} {r['label']}{suf} | " + " | ".join(cells_display) + " |")

    print(f"\nCSVs:\n- [{long_csv.relative_to(REPO)}]({long_csv.relative_to(REPO)})\n- [{cells_csv.relative_to(REPO)}]({cells_csv.relative_to(REPO)})")


if __name__ == "__main__":
    main()
