#!/usr/bin/env python3
"""Per-cell eval coverage matrix: 4 columns × N tasks expected.

For each (dataset, model, setting) cell, show counts (X/N) for each of the
four CSV-prefix families:

  PMI base : --self-typcorr --base-typcorr   →  scores_basetyp-*
  PMI self : --self-typcorr (no base)        →  scores_self-*
  Neg base : --neg-typcorr  --base-typcorr   →  scores_basetypneg-*
  Neg self : --neg-typcorr  (no base)        →  scores_neg-*

Cells that are "n/a" by the trained-TC rule are marked n/a (e.g. for an
s4 self-TC-trained model, Neg base/Neg self are n/a).

Also includes a `T` column showing whether the trained model exists on
disk (epoch level) plus any in-flight slurm jobs (train/eval).

Usage: python scripts/_eval_coverage_matrix.py
"""
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path("/datastor1/jdr/gv-gap/rankalign")
sys.path.insert(0, str(REPO / "scripts"))

# Reuse the heavy lifting from _status_matrix.
from _status_matrix import (  # type: ignore
    DATASETS, MODELS, SETTINGS, EVAL_TC_VARIANTS,
    scan_models, scan_score_csvs,
    get_active_jobs, get_train_jobs_from_log, parse_active_job,
)

EXPECTED_TASKS = {
    "membership": 10,
    "persona":    6,
    "ifeval":     21,
    "humaneval":  82,  # rough; varies, not used as gate
}

PREFIX_BY_COL = {
    "PMI base": ("basetyp-",    "self"),  # base-self → basetyp-
    "PMI self": ("self-",       "self"),
    "Neg base": ("basetypneg-", "neg"),
    "Neg self": ("neg-",        "neg"),
}

COLS = ["PMI base", "PMI self", "Neg base", "Neg self"]


def render_cell(count_csvs, expected, applicable, queued):
    if not applicable:
        return "n/a"
    base = f"{count_csvs}/{expected}" if expected else f"{count_csvs}"
    if queued:
        base = base + f" [{queued}]"
    return base


def main():
    models_on_disk = scan_models()
    csvs = scan_score_csvs()  # key: (dataset, model, setting, tc_prefix) -> set of tasks

    active_jobs = get_active_jobs()
    train_log_map = get_train_jobs_from_log()

    # Track active eval jobs per (dataset, setting, tc, no_base?)
    # Names look like "eval-s4-membership-self" or "eval-s1-membership-neg-only" or
    # "eval-s5-membership-self-nobase-only".
    eval_active = defaultdict(list)  # (dataset, setting, tc, no_base_bool) -> [(jobid,state)]
    train_active = {}  # (dataset, model, setting) -> (jobid, state)
    import re
    eval_re = re.compile(r"^eval-(s\d+)-(\w+?)-(self|neg)(-nobase)?(-only)?$")
    for j in active_jobs:
        m = eval_re.match(j["name"])
        if m:
            setting = m.group(1)
            dataset = m.group(2)
            tc = m.group(3)
            no_base = bool(m.group(4))
            eval_active[(dataset, setting, tc, no_base)].append((j["jobid"], j["state"]))
        elif j["name"] == "wrap":
            jid = j["jobid"]
            tl = train_log_map.get(jid)
            if tl:
                train_active[(tl["dataset"], tl["model"], tl["setting"])] = (jid, j["state"])

    print("# Eval coverage matrix — per cell, 4 prefix-families")
    print()
    print(("For each (dataset, model, setting) cell, counts of `scores_*.csv` files "
           "found per CSV-prefix family, vs expected N eval-tasks for that dataset. "
           "`n/a` = the TC-variant doesn't apply to a model trained with the "
           "opposite or no TC objective."))
    print()
    print("Trained-TC rule (controls n/a):")
    print()
    print("- s1, s2, s3, s13   (no TC trained)   → eval both `self` and `neg`")
    print("- s4, s5, s6, s11   (self-TC trained) → eval `self` only (Neg cols n/a)")
    print("- s7, s12           (neg-TC trained)  → eval `neg` only  (PMI cols n/a)")
    print()
    print("Annotations: `[42140]` = pending/running slurm eval job covering this cell.")
    print("`done(eN)` / `running(JID)` for the train column.")
    print()

    for ds_short in DATASETS:
        n = EXPECTED_TASKS[ds_short]
        print(f"## {ds_short}  (N = {n} expected eval tasks per cell)")
        print()
        print("| model | s# | T | PMI base | PMI self | Neg base | Neg self |")
        print("|---|---|---|---|---|---|---|")
        for model in MODELS:
            # Skip g4-31B for non-humaneval rows (only s13/humaneval applies for it)
            if model == "gemma-4-31B-it" and ds_short != "humaneval":
                continue
            for setting in SETTINGS:
                # humaneval only applies to s13/g4-31B for now
                if ds_short == "humaneval" and not (model == "gemma-4-31B-it" and setting == "s13"):
                    continue
                # Skip g4-31B/non-humaneval already filtered above
                key_train = (ds_short, model, setting)
                # Train state
                if key_train in models_on_disk:
                    t_str = f"done(e{models_on_disk[key_train]['max_epoch']})"
                elif key_train in train_active:
                    jid, st = train_active[key_train]
                    t_str = f"{st.lower()}({jid})"
                else:
                    t_str = "—"
                # 4 columns
                applicable = EVAL_TC_VARIANTS[setting]
                # PMI base/self apply only if "self" in applicable
                # Neg base/self apply only if "neg"  in applicable
                row_cells = []
                for col in COLS:
                    csv_pfx, tc_variant = PREFIX_BY_COL[col]
                    is_applicable = tc_variant in applicable
                    if not is_applicable:
                        row_cells.append("n/a")
                        continue
                    cnt = len(csvs.get((ds_short, model, setting, csv_pfx), set()))
                    # Find queued evals matching this prefix family.
                    # Heuristic: dispatcher evals (no_base=False) yield basetyp-*/basetypneg-*.
                    # NO_BASE=1 evals (no_base=True) yield self-*/neg-*.
                    no_base_for_col = (csv_pfx in ("self-", "neg-"))
                    queued = eval_active.get((ds_short, setting, tc_variant, no_base_for_col), [])
                    queued_str = ",".join(qid for qid, _ in queued) if queued else ""
                    row_cells.append(render_cell(cnt, n, True, queued_str))
                print(f"| {model} | {setting} | {t_str} | " + " | ".join(row_cells) + " |")
        print()

    # Summary: how many missing cells (need a fresh eval submission)?
    print("## Gaps requiring submission")
    print()
    needed = []
    for ds_short in DATASETS:
        n = EXPECTED_TASKS[ds_short]
        for model in MODELS:
            if model == "gemma-4-31B-it" and ds_short != "humaneval":
                continue
            for setting in SETTINGS:
                if ds_short == "humaneval" and not (model == "gemma-4-31B-it" and setting == "s13"):
                    continue
                key_train = (ds_short, model, setting)
                if key_train not in models_on_disk and key_train not in train_active:
                    continue  # no train, skip — train must come first
                applicable = EVAL_TC_VARIANTS[setting]
                for col in COLS:
                    csv_pfx, tc_variant = PREFIX_BY_COL[col]
                    if tc_variant not in applicable:
                        continue
                    cnt = len(csvs.get((ds_short, model, setting, csv_pfx), set()))
                    no_base_for_col = (csv_pfx in ("self-", "neg-"))
                    queued = eval_active.get((ds_short, setting, tc_variant, no_base_for_col), [])
                    if cnt < (n or cnt + 1) and not queued:
                        needed.append((ds_short, model, setting, col, csv_pfx, no_base_for_col))
    if not needed:
        print("- (none — all applicable cells either covered by CSVs or have queued jobs)")
    else:
        print(f"Total: **{len(needed)} (cell × prefix) gaps with no in-flight job**.")
        print()
        print("| dataset | model | setting | column | csv prefix | NO_BASE? |")
        print("|---|---|---|---|---|---|")
        for ds, m, s, col, pfx, nb in needed:
            print(f"| {ds} | {m} | {s} | {col} | `{pfx}` | {'1' if nb else '0'} |")


if __name__ == "__main__":
    main()
