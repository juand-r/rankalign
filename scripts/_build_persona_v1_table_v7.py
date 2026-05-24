#!/usr/bin/env python3
"""Build a persona-v1 metrics table for v7 / fix1 trained models.

This is the **v7** sibling of `_build_persona_v1_table.py`. It consumes
score CSVs produced by evaluations of models trained with
`scripts/ranking_loss_ref_fix.py` (which prefixes model dirs with `v7-`,
appends `--fix1`, allows variable `--delta-bins`-derived deltas, and
introduces the `--ppd` token for `--per-prompt-delta` runs).

Rows: methods (0=Base, 1..9, 11, 12). NA structure follows the v7
dispatcher (`scripts/_overnight_launch.sh`) which only runs ONE TC
variant per setting at eval time:
  s1, s2, s3, s4, s5, s6, s11 -> self-tc eval only -> Neg* are NA
  s7, s8, s9, s12             -> neg-tc eval only  -> PMI* are NA

Cols: Raw, PMI self, PMI base, Neg self, Neg base.

Each cell = mean ± SE of <metric> × 100 across the persona-v1 personas in
the selected split.

Env vars (mirror the v6 builder's API):

  PERSONA_BASE   ∈ {9b-it, 2b-it, 2b}        (default 9b-it)
  PERSONA_SPLIT  ∈ {id, ood, all}            (default all)
                   id  = 3 in-domain personas (label-flipped)
                   ood = 3 held-out personas
                   all = all 6
  PERSONA_DISC   ∈ {auto, zero, few}         (default auto = no filter)
                   When zero/few: only include rows from CSVs whose
                   `strategy` column matches `disc:{zero|few}`.
  PERSONA_METRIC ∈ {gen_roc, pearson, spearman, val_roc, val_acc}
                                              (default gen_roc)

Output: metrics-from-scores/persona_v1_v7_{base}_{split}_{metric}{_disc-X?}_table_{long,cells}.csv

NOTE on filenames: persona-v1 uses `--` as the model_short separator (not
`_` like rosch/ifeval). Long 9b-it names (>160 chars) get md5-truncated
by `src/tasks/common.py:build_model_short()`. The v7 builder handles
this by maintaining a `truncated_short -> full_basename` map built from
the actual `v7-*` directories on disk (in /datastor2 + /datastor1), then
runs the per-method regex against the FULL basename. This way new
deltas + new epochs are matched correctly without needing to enumerate
them up-front.
"""

from __future__ import annotations
import os
import re
import sys
import math
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
from summarize_scores_file import load_scores, compute_all_metrics  # noqa: E402
from checkpoint_name_parser import matches_v7_setting  # noqa: E402

OUT_DIR = REPO / "outputs"  # kept for back-compat; SEARCH_DIRS is the truth
# Multiple search locations: local outputs/ plus the datastor2 mirror where
# trained-adapter persona-v1 score files actually live (Base-row files are
# symlinked into outputs/, but trained-method files are not). Order matters
# only for the (rare) case where the same exact basename exists in both:
# first hit wins, then duplicates are collapsed by keeping the newest filename.
SEARCH_DIRS = [
    OUT_DIR,
    Path("/datastor2/jdr/rankalign/outputs"),
]
# v7 model directories. We scan these to build a truncated_short ->
# full_basename map so that hashed/truncated model_shorts in CSV
# filenames can be resolved back to the full canonical name for regex
# matching against per-method patterns.
MODEL_DIRS = [
    Path("/datastor2/jdr/rankalign/models2"),
    Path("/datastor1/jdr/gv-gap/rankalign/models"),
    Path("/datastor1/jdr/gv-gap/rankalign/models2"),
]
METRICS_DIR = REPO / "metrics-from-scores"
METRICS_DIR.mkdir(exist_ok=True)

# Mirror src/tasks/common.py:build_model_short
_MODEL_SHORT_MAX_LEN = 160


def _maybe_truncate_for_match(raw: str) -> str:
    """Return the actual model_short that would land in filenames."""
    if len(raw) > _MODEL_SHORT_MAX_LEN:
        h = hashlib.md5(raw.encode()).hexdigest()[:8]
        return raw[:_MODEL_SHORT_MAX_LEN - 9] + '_' + h
    return raw


# Built lazily on first use.
_DIR_INDEX: dict[str, str] | None = None


def _build_dir_index() -> dict[str, str]:
    """Map `model_short_in_csv -> full_basename` for all v7-* dirs on disk.

    For dirs whose basename fits in 160 chars, this is identity. For
    long ones it captures the md5-truncated form that build_model_short
    would produce, so we can resolve a truncated model_short found in
    a CSV filename back to its full basename for full-regex matching.
    """
    idx: dict[str, str] = {}
    for d in MODEL_DIRS:
        if not d.is_dir():
            continue
        for sub in d.glob("v7-*"):
            full = sub.name
            short = _maybe_truncate_for_match(full)
            # If two paths produce the same short (rare), prefer the
            # most-recently-modified one so retrains take precedence.
            existing = idx.get(short)
            if existing is None or sub.stat().st_mtime > (d / existing).stat().st_mtime:
                idx[short] = full
    return idx


def _resolve_full_basename(model_short: str) -> str:
    """Return the FULL on-disk basename for a model_short (may be truncated).

    Falls back to the input if no matching dir is found. Important: when
    you're building v7 tables BEFORE training finishes, the dir may not
    exist yet — the regex match then runs against the (possibly
    truncated) `model_short` directly, which is fine for short names
    but will fail-closed for hashed-truncated 9b-it names. Once the
    real dir is on disk, the match works.
    """
    global _DIR_INDEX
    if _DIR_INDEX is None:
        _DIR_INDEX = _build_dir_index()
    return _DIR_INDEX.get(model_short, model_short)


# ============================================================================
# Env vars / configuration
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

BASE_KEY = (os.environ.get("PERSONA_BASE", "9b-it")).lower()
if BASE_KEY not in {"9b-it", "2b-it", "2b"}:
    raise SystemExit(f"PERSONA_BASE must be one of 9b-it/2b-it/2b, got {BASE_KEY!r}")
BASE_HF = f"google/gemma-2-{BASE_KEY}"
# 9b-it is LoRA → merged dir name carries `_merged`. 2b/2b-it: no merge step.
USE_MERGED = (BASE_KEY == "9b-it")
MERGED_TAG = "_merged" if USE_MERGED else ""

DISC = (os.environ.get("PERSONA_DISC", "auto")).lower()
if DISC not in {"auto", "zero", "few"}:
    raise SystemExit(f"PERSONA_DISC must be 'auto'/'zero'/'few', got {DISC!r}")

METRIC = (os.environ.get("PERSONA_METRIC", "gen_roc")).lower()
SUPPORTED_METRICS = {"gen_roc", "pearson", "spearman", "val_roc", "val_acc"}
if METRIC not in SUPPORTED_METRICS:
    raise SystemExit(f"PERSONA_METRIC must be one of {SUPPORTED_METRICS}, got {METRIC!r}")
METRIC_LABEL = {
    "gen_roc": "GenROC", "pearson": "Pearson(gen, val)",
    "spearman": "Spearman(gen, val)", "val_roc": "ValROC", "val_acc": "ValAcc",
}[METRIC]

# Trained-model identification is now structural (parse → field-compare),
# handled by `matches_v7_setting()` from checkpoint_name_parser. This works
# uniformly across the three on-disk filename formats:
#   (A) legacy abs-path-embedded: `v6-_datastor2_..._v7-google--gemma-...-fix1`
#   (B) un-abbreviated:           `v7-google--gemma-2-9b-it-delta1.42-...-fix1[_merged]`
#   (C) abbreviated (>160ch):     `v7-gemma-2-9b-it-d1.42-e2-...-sm0.1-fix1`
# (md5-truncated names are no longer produced by build_model_short — we now
# fall back to `to_hf_repo_name()` instead — but the dir-index resolution
# below still helps if a CSV happened to be written before that change.)
TASK_SEG = "persona-v1-all"
GEMMA_MODEL = f"gemma-2-{BASE_KEY}"

COLUMNS = [
    ("Raw",      ["self-", "neg-", "basetyp-", "basetypneg-", ""], "raw"),
    ("PMI self", "self-",       "tc"),
    ("PMI base", "basetyp-",    "tc"),
    ("Neg self", "neg-",        "tc"),
    ("Neg base", "basetypneg-", "tc"),
]


# ============================================================================
# Methods (rows)
# ============================================================================
#
# v7 persona-v1 launcher SUFFIXES (per scripts/_overnight_launch.sh):
#   s1  SFT-lo:                --full-completion--pref0.0--nllv1.0--nllg1.0--labelonly0.1
#   s2  RankAlign:             --full-completion--semi0.1
#   s3  New+fsx [-TC]:         --full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1
#   s4  New+fsx+selfTC:        --tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1
#   s5  RankAlign+fsx+selfTC:  --tc-self--full-completion--force-same-x--ppd--semi0.1   (NO vlo — clean)
#   s6  RankAlign+selfTC:      --tc-self--full-completion--semi0.1
#   s7  New+fsx+negTC:         --tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1
#   s8  RankAlign+fsx+negTC:   --tc-neg--full-completion--force-same-x--ppd--semi0.1    (NO vlo — clean; not in v7 launcher)
#   s9  RankAlign+negTC:       --tc-neg--full-completion--semi0.1                       (not in v7 launcher)
#   s11 New+selfTC [-fsx]:     --tc-self--full-completion--nllv1.0--nllg1.0--vallogodds--semi0.1
#   s12 New+negTC [-fsx]:      --tc-neg--full-completion--nllv1.0--nllg1.0--vallogodds--semi0.1
# Each suffix is followed by FIX_TAIL (`--fix1[_merged]`).


def _setting_match(**expected):
    """Build a structural matcher closing over `expected`. Resolves
    md5-truncated model_shorts (legacy artefact) via the DIR_INDEX before
    parsing — for non-truncated names this is identity.
    """
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
        # Base HF path → "v6-google_gemma-2-<base>" with `_` separator.
        match=lambda s: s == f"v6-google_gemma-2-{BASE_KEY}",
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
    # (Not in the v7 overnight launcher; forward-compat.)
    dict(num=8, label="RA + NegTC + fsx [-NLL]",
         match=_setting_match(tc='neg', force_same_x=True, ppd=True,
                              semi=0.1)),
    # 9 RA + NegTC [+TC]: pref-only + tc-neg, no fsx, no NLL, no vlo.
    dict(num=9, label="RA + NegTC [+TC]",
         match=_setting_match(tc='neg', semi=0.1)),
    # 10 RankAlign + fsx: pref-only + fsx (+ ppd), no TC, no NLL, no vlo.
    # (Forward-compat; not in v7 launcher.)
    dict(num=10, label="RankAlign + fsx",
         match=_setting_match(force_same_x=True, ppd=True, semi=0.1)),
    # 11 New + PMI [-fsx]: comb + tc-self + vlo, no fsx.
    dict(num=11, label="New + PMI [-fsx]",
         match=_setting_match(tc='self', nll_v=1.0, nll_g=1.0,
                              vallogodds=True, semi=0.1)),
    # 12 New + NegTC [-fsx]: comb + tc-neg + vlo, no fsx.
    dict(num=12, label="New + NegTC [-fsx]",
         match=_setting_match(tc='neg', nll_v=1.0, nll_g=1.0,
                              vallogodds=True, semi=0.1)),
]

# Persona-v1 NA structure:
#   - Base (#0): baseline used --self-typcorr and --neg-typcorr (no
#     --base-typcorr). So `PMI base` / `Neg base` are NA.
#   - All trained methods (#1..#12): TC_POLICY in run_eval_persona_v1_trained.sh
#     always passes --base-typcorr; eval files only ever exist with `basetyp-`
#     or `basetypneg-` prefixes (never bare `self-`/`neg-`). So `PMI self` and
#     `Neg self` are NA.
#   - Plus the standard per-TC-type masking: methods that train with a
#     specific TC are only meaningfully evaluable with that TC.
# v7 NA structure: the dispatcher (`_overnight_launch.sh`) sets
# TC_EVAL_LIST per setting. Settings that train without TC still get
# evaluated with --self-typcorr only (TC_EVAL_LIST="self"), so there
# is no neg-tc score CSV for them. Settings that train with --tc-neg
# get TC_EVAL_LIST="neg" → no self-tc CSV.
#
#   self-eval-only (TC_EVAL_LIST="self"): rows 1, 2, 3, 4, 5, 6, 11
#   neg-eval-only  (TC_EVAL_LIST="neg"):  rows 7, 8, 9, 12
#
# This is a structural NA: those CSVs literally cannot exist. (Differs
# from the v6 builder which had no NA on rows 1-3 because v6 ran both
# self and neg evals on no-TC-trained methods.)
NA_COLS = {
    # Base: only self/neg-tc evals were run on the bare HF model; no
    # base-tc/basetypneg-tc CSV exists. Same as v6.
    0: {"PMI base", "Neg base"},
    1:  {"Neg self", "Neg base"},
    2:  {"Neg self", "Neg base"},
    3:  {"Neg self", "Neg base"},
    4:  {"Neg self", "Neg base"},
    5:  {"Neg self", "Neg base"},
    6:  {"Neg self", "Neg base"},
    11: {"Neg self", "Neg base"},
    7:  {"PMI self", "PMI base"},
    8:  {"PMI self", "PMI base"},
    9:  {"PMI self", "PMI base"},
    12: {"PMI self", "PMI base"},
    # 10 (RankAlign + fsx): not in v7 launcher's setting list. Leave
    # unconstrained so a forward-compat training would populate all
    # four TC columns if both evals run.
    10: set(),
}


# ============================================================================
# File discovery + cell computation
# ============================================================================

def _check_disc_match(df: pd.DataFrame) -> bool:
    if DISC == "auto":
        return True
    if "strategy" not in df.columns or df["strategy"].empty:
        return False
    strat = str(df["strategy"].iloc[0])
    return f"disc:{DISC}" in strat


# Index persona-v1 score files once: parse each filename into
# (eval_prefix, model_short, task, path). Then per-method lookups are O(N)
# linear scans over this small list (a few hundred files) instead of O(M*P)
# globs against the whole 10k+ outputs/ dir.
def _build_index() -> list[tuple[str, str, str, Path]]:
    KNOWN_PREFIXES = ("self-", "neg-", "basetyp-", "basetypneg-")
    index: list[tuple[str, str, str, Path]] = []
    seen_basenames: set[str] = set()
    # Persona-v1-specific glob narrows from ~10k files to ~hundreds per dir.
    for d in SEARCH_DIRS:
        if not d.is_dir():
            continue
        for p in d.glob("scores_*persona-v1-*_test_log-odds*.csv"):
            name = p.name
            # Same basename present in a later dir: skip (avoids double-counting
            # if the file is also symlinked locally).
            if name in seen_basenames:
                continue
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
            seen_basenames.add(name)
    return index


_FILE_INDEX: list[tuple[str, str, str, Path]] | None = None


def find_score_files(method: dict, eval_prefix) -> dict[str, list[Path]]:
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
        if not method["match"](model_short):
            continue
        matches[task].append(p)
    return matches


def cell_value(method: dict, eval_prefix, variant: str):
    files_by_task = find_score_files(method, eval_prefix)
    vals = []
    used_files = []
    for task in EVAL_TASKS:
        candidates = files_by_task.get(task, [])
        if not candidates:
            continue
        # Pick the latest-by-name candidate that also passes the disc filter.
        chosen_path, chosen_df = None, None
        for path in sorted(candidates, reverse=True):
            try:
                df = load_scores(path)
            except Exception:
                continue
            if not _check_disc_match(df):
                continue
            chosen_path, chosen_df = path, df
            break
        if chosen_path is None:
            continue
        # Sanity: a Base-row file should never contain training-checkpoint
        # markers. `delta<lr>-epoch<N>` is appended to model_short by the
        # training pipeline for every fine-tuned adapter; its presence in a
        # filename matched to method #0 means our matcher leaked.
        if method.get("num") == 0 and "delta" in chosen_path.name.lower():
            raise SystemExit(
                f"[persona-v1 Base contamination] file matched to Base row "
                f"contains 'delta' (fine-tuned adapter marker):\n  {chosen_path}"
            )
        try:
            metrics = compute_all_metrics(chosen_df)
        except Exception:
            continue
        if variant not in metrics:
            continue
        v = metrics[variant].get(METRIC)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        vals.append((task, float(v), chosen_path.name))
        used_files.append((task, chosen_path.name, float(v)))
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


# ============================================================================
# Main
# ============================================================================

def main():
    long_rows = []
    cell_rows = []
    table_rows = []

    # Row order. #10, #11, #12 commented out — re-enable when trained.
    ROW_ORDER = [
        0, 1, 2, 3, 4, 5, 6,
        # 11,
        7, 8, 9,
        # 12,
        # 10,
    ]
    methods_by_num = {m["num"]: m for m in METHODS}
    ordered = [methods_by_num[n] for n in ROW_ORDER]

    for m in ordered:
        cells = {}
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
                note=("OK" if c["n"] == N_EXPECTED
                      else f"missing {N_EXPECTED - c['n']}/{N_EXPECTED}"),
            ))
            for task, val, fn in c["vals"]:
                long_rows.append(dict(
                    method_num=m["num"], method=m["label"],
                    column=col_label, eval_prefix=str(eval_prefix), variant=variant,
                    task=task, value=val, file=fn,
                ))
        table_rows.append(dict(num=m["num"], label=m["label"], **cells))

    # Output naming
    disc_suffix = "" if DISC == "auto" else f"_disc-{DISC}"
    base_tag = f"gemma-2-{BASE_KEY}"
    long_csv = METRICS_DIR / f"persona_v1_v7_{base_tag}_{SPLIT}_{METRIC}{disc_suffix}_table_long.csv"
    cells_csv = METRICS_DIR / f"persona_v1_v7_{base_tag}_{SPLIT}_{METRIC}{disc_suffix}_table_cells.csv"
    pd.DataFrame(long_rows).to_csv(long_csv, index=False)
    pd.DataFrame(cell_rows).to_csv(cells_csv, index=False)

    # Print markdown table
    split_human = {"id": "in-domain", "ood": "held-out OOD", "all": "all 6"}[SPLIT]
    print(f"\nPersona-v1 [v7/fix1] {split_human} {METRIC_LABEL} × 100 — mean ± SE across "
          f"{N_EXPECTED} personas ({', '.join(PERSONAS)})")
    disc_human = {"auto": "any", "zero": "zero", "few": "few"}[DISC]
    print(f"Model: {BASE_HF}, persona-v1-all (epoch picked from epoch[012] glob)  |  disc-shots: {disc_human}\n")
    header = ["Method"] + [c[0] for c in COLUMNS]
    sep = ["---"] * len(header)
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(sep) + " |")
    for r in table_rows:
        cells_display = [str(r[c[0]]) for c in COLUMNS]
        print(f"| {r['num']} {r['label']} | " + " | ".join(cells_display) + " |")

    print(f"\nCSVs:\n- [{long_csv.relative_to(REPO)}]({long_csv.relative_to(REPO)})\n"
          f"- [{cells_csv.relative_to(REPO)}]({cells_csv.relative_to(REPO)})")


if __name__ == "__main__":
    main()
