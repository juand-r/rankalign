#!/usr/bin/env python
"""Build the eval-coverage inventory (TASK 2) from captured scores filenames.

Input:  _raw/scores_v7_local_listing.txt  (21k+ `subdir/scores_*.csv` basenames,
        captured from the LOCAL repo outputs_gemma4_from_pod-v7{,b}/ dirs)
Output: v7/eval_inventory_v7.md  + _raw/eval_coverage.csv

Two filename schemes (see SETTINGS_REFERENCE / score_filename_convention.md):
  A. full v7 dir-name:  scores_{prefix}v7-google--<model>-delta<D>-epoch<E>--<task>...fix1_<evaltask>_test_...
     -> model/delta/epoch/setting all parseable from the name.
  B. eval_model_sN:     scores_{prefix}eval_model_s<N>_<evaltask>_test_...
     -> setting = sN; MODEL + trained-task come from the SUBDIR; epoch NOT in name.

eval_prefix -> eval-time TC column:
  self- = PMI self | neg- = Neg self | basetyp- = PMI base | basetypneg- = Neg base | (none) = raw-only
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

from _inv_common import (classify_setting, eval_task_to_train_task, norm_model,
                         setting_sort_key, SETTING_NAME)

HERE = Path(__file__).parent
RAW = HERE / "_raw"

# subdir -> (model, default trained-task or None if task comes from eval-task, version, default-epoch)
SUBDIR_MAP = {
    "correct_multi_s1s2": ("gemma-4-31B-it", "humaneval-cm", "v7", None),
    "correct_multi_s3s4s7": ("gemma-4-31B-it", "humaneval-cm", "v7", None),
    "correct_upper_s1s4s7": ("gemma-4-31B-it", "humaneval-cu", "v7", None),
    "s13_ep0": ("gemma-4-31B-it", "humaneval-cu", "v7", 0),
    "s1_sft_v6": ("gemma-4-31B-it", "humaneval-cu", "v6", None),
    "qw35_ifeval": ("qwen3.5-9b", "ifeval", "v7", None),
    "qw35_persona_member": ("qwen3.5-9b", None, "v7", None),
    "ra9b_ifeval": ("gemma-2-9b-it", "ifeval", "v7", None),
    "ra9b_persona_member": ("gemma-2-9b-it", None, "v7", None),
    "v7b_ifeval": ("gemma-2-9b-it", "ifeval", "v7b", None),
    "v7b_persona_member": ("gemma-2-9b-it", None, "v7b", None),
}

PREFIXES = ["self-", "neg-", "basetyp-", "basetypneg-"]
PREFIX_COL = {"self-": "PMI self", "neg-": "Neg self", "basetyp-": "PMI base",
              "basetypneg-": "Neg base", "": "raw-only"}

EVALTASK_RE = re.compile(
    r"(rosch-[a-z-]+|persona-v1-[a-z0-9-]+|ifeval-prompt_\d+|humaneval-v2\.1correct-(?:upper|multi)-humaneval_\d+)")
# Form B (full):   v7-google--gemma-2-9b-it-delta1.42-epoch2--...   OR  v7-Qwen--Qwen3.5-9B-delta0.96-epoch2--...
FULL_B_RE = re.compile(r"(v[67]b?)-(?:google|Qwen)--(.+?)-delta([0-9.]+)-epoch(\d+)")
# Form C (abbrev): v7-Qwen3.5-9B-d0.96-e2-...   OR  v7-gemma-2-9b-it-d1.42-e2-...
FULL_C_RE = re.compile(r"(v[67]b?)-([A-Za-z0-9.-]+?)-d([0-9.]+)-e(\d+)-")
EVALMODEL_RE = re.compile(r"eval_model_s(\d+)")
# Base (unfinetuned) model:  v6-google_gemma-4-31B-it_<evaltask>  or  v6-Qwen_Qwen3.5-9B_<evaltask>
BASE_RE = re.compile(r"v6-(?:google|Qwen)_([A-Za-z0-9.-]+?)_(?=rosch|persona|ifeval|humaneval)")


def parse_line(line: str) -> dict | None:
    if "/" not in line:
        return None
    subdir, fname = line.split("/", 1)
    if subdir not in SUBDIR_MAP:
        return None
    model_def, task_def, version, epoch_def = SUBDIR_MAP[subdir]

    prefix = ""
    body = fname[len("scores_"):] if fname.startswith("scores_") else fname
    for p in PREFIXES:
        if body.startswith(p):
            prefix = p
            break

    # eval task (domain) — robust regex search anywhere in the name
    mt = EVALTASK_RE.search(fname)
    eval_task = mt.group(1) if mt else "?"
    trained_task = task_def or eval_task_to_train_task(eval_task)

    mem = EVALMODEL_RE.search(fname)
    mfull = FULL_B_RE.search(fname) or FULL_C_RE.search(fname)
    mbase = BASE_RE.search(fname)
    if mem:
        setting = "s" + mem.group(1)
        model = model_def
        epoch = epoch_def  # unknown unless subdir pins it
        delta = ""
    elif mfull:
        model = norm_model(mfull.group(2))
        delta = mfull.group(3)
        epoch = int(mfull.group(4))
        toks = set()
        for t in ("pref0.0", "p0", "nllv1.0", "nv1", "nllg1.0", "ng1", "cft",
                  "force-same-x", "fsx", "tc-self", "tcs", "tc-neg", "tcn"):
            if t in fname:
                toks.add(t)
        setting = classify_setting(toks)
        if epoch_def is not None:
            epoch = epoch_def
    elif mbase:
        setting = "base"
        model = norm_model(mbase.group(1))
        epoch = "base"
        delta = ""
        trained_task = eval_task_to_train_task(eval_task)  # base has no train-task
    else:
        return {"subdir": subdir, "ok": False, "raw": fname}

    return {"subdir": subdir, "ok": True, "model": model, "trained_task": trained_task,
            "setting": setting, "epoch": epoch if epoch is not None else "?",
            "delta": delta, "prefix": prefix, "eval_task": eval_task, "version": version}


def main() -> None:
    rows = []
    bad = []
    for line in (RAW / "scores_v7_local_listing.txt").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = parse_line(line)
        if r is None:
            continue
        (rows.append(r) if r.get("ok") else bad.append(r["raw"]))

    # aggregate: (version, model, trained_task, setting, epoch) -> {prefix: set(eval_task)}
    agg: dict = defaultdict(lambda: defaultdict(set))
    for r in rows:
        key = (r["version"], r["model"], r["trained_task"], r["setting"], r["epoch"])
        agg[key][r["prefix"] or ""].add(r["eval_task"])

    with (RAW / "eval_coverage.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["version", "model", "trained_task", "setting", "epoch",
                    "PMI self (self-)", "Neg self (neg-)", "PMI base (basetyp-)",
                    "Neg base (basetypneg-)", "raw-only", "subdirs"])
        subdir_of = defaultdict(set)
        for r in rows:
            subdir_of[(r["version"], r["model"], r["trained_task"], r["setting"], r["epoch"])].add(r["subdir"])
        for key in sorted(agg, key=lambda k: (k[0], k[1], k[2], setting_sort_key(k[3]), str(k[4]))):
            v, model, task, setting, epoch = key
            pc = agg[key]
            w.writerow([v, model, task, setting, epoch,
                        len(pc.get("self-", set())), len(pc.get("neg-", set())),
                        len(pc.get("basetyp-", set())), len(pc.get("basetypneg-", set())),
                        len(pc.get("", set())), "|".join(sorted(subdir_of[key]))])

    # markdown
    out = ["# Eval inventory — v7 (raw scores_ coverage)", "",
           "Auto-generated by `_build_eval_inventory.py` from the LOCAL "
           "`outputs_gemma4_from_pod-v7{,b}/` scores files (also mirrored on "
           "`/datastor2/jdr/rankalign/`). Numbers are **distinct eval tasks** with a "
           "`scores_*.csv` of that eval-time-TC type. Expected eval-task counts: "
           "humaneval-cu/cm **82**, ifeval **21**, persona **6**, membership(rosch) **10**.",
           "",
           "Columns are eval-time TC (= result-table columns): **PMI self** (`self-`), "
           "**Neg self** (`neg-`), **PMI base** (`basetyp-`), **Neg base** (`basetypneg-`), "
           "**raw-only** (no-prefix). Every CSV also contains the raw score internally.",
           "",
           "> Scheme note: gemma-4 cu/cm files carry the full model name (epoch parsed). "
           "qw35_* / ra9b_* / v7b_* files use the `eval_model_sN` placeholder — **model + "
           "trained-task come from the subdir, and epoch is NOT in the filename** (shown `?`; "
           "the evaluated epoch must be read from the eval wrapper / `eval_coverage_matrix.md`). "
           "qw35_persona_member is heavily **duplicated** (re-runs); the metric builders dedup "
           "via `metrics-from-scores/*_files_used.csv` + `*_dups_collapsed.csv`.",
           "",
           "> **`ver=v7b` is only a label** for the fixed-`delta 0.15` batch (no delta-bins / ppd / "
           "sbm-global). It is **NOT a filename prefix** — every v7b model dir and score file still "
           "starts with `v7-` (no \"b\"); the `v7b` value here is inferred from the download-folder "
           "name (`outputs_gemma4_from_pod-v7b/`). Literal `v7b` exists only in the HF repo names "
           "and those folder names. Tell v7b from delta-bins v7 by the delta (0.15 fixed vs computed).",
           "",
           "> The **`subdir` column is a source download-folder, not a task.** `*_persona_member` "
           "bundles BOTH persona and membership(rosch) scores for that model (e.g. "
           "`qw35_persona_member` = Qwen3.5-9B persona + membership together).",
           ""]
    tasks = sorted({k[2] for k in agg})
    for task in tasks:
        out.append(f"\n## {task}\n")
        out.append("| ver | model | setting | epoch | PMI self | Neg self | PMI base | Neg base | raw | subdir |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        keys = sorted([k for k in agg if k[2] == task],
                      key=lambda k: (k[0], k[1], setting_sort_key(k[3]), str(k[4])))
        for key in keys:
            v, model, _task, setting, epoch = key
            pc = agg[key]
            sd = sorted({r["subdir"] for r in rows if (r["version"], r["model"], r["trained_task"], r["setting"], r["epoch"]) == key})
            def n(p):
                return len(pc.get(p, set())) or "·"
            sname = SETTING_NAME.get(setting, setting)
            out.append(f"| {v} | {model} | {setting} ({sname}) | {epoch} | "
                       f"{n('self-')} | {n('neg-')} | {n('basetyp-')} | {n('basetypneg-')} | "
                       f"{n('')} | {','.join(sd)} |")
    (HERE / "v7" / "eval_inventory_v7.md").write_text("\n".join(out) + "\n")

    print(f"parsed {len(rows)} scores rows; {len(bad)} unparseable")
    if bad:
        (RAW / "eval_unparsed.txt").write_text("\n".join(bad[:200]) + "\n")
        print("sample bad:", bad[:3])


if __name__ == "__main__":
    main()
