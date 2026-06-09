#!/usr/bin/env python3
"""Fast FILENAME-ONLY inventory of v7 score files across ALL outputs* dirs.

The v7 ifeval/rosch evals were produced in two places and two naming schemes:
  - pod downloads : outputs_gemma4_from_pod-v7*/...  -> `eval_model_sN` scheme (v7)
  - local mll runs: outputs*/...                     -> full model-name scheme

This classifies every ifeval/rosch score file (no CSV reading) by:
  model {9b-it, qwen}, task {ifeval, rosch}, eval-TC variant {self,neg,basetyp,basetypneg,raw},
  split {ood,id,all}, and VERSION CLASS:
      v6            : name has 'v6-'                         (EXCLUDE)
      v7b           : name has 'v7-' and 'delta0.15'         (EXCLUDE - fixed delta)
      v7-deltabins  : name has 'v7-' and delta != 0.15       (KEEP)
      v7-evalmodel  : 'eval_model_sN' scheme (pod, v7)        (KEEP)
      other         : unclassified
and reports, per kept v7 cell, the distinct eval-task/prompt count + which dirs.

Usage (mll):  python scripts/_inventory_v7_scores.py [root]
"""
from __future__ import annotations

import os
import re
import sys
from collections import defaultdict

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."

PREFIX = re.compile(r"scores_(self|neg|basetyp|basetypneg)-")
PROMPT = re.compile(r"prompt[_-](\d+)")
SETT = re.compile(r"eval_model_s(\d+)")
DELTA = re.compile(r"delta([0-9.]+)")


def model_of(name: str, dirpath: str) -> str | None:
    if "gemma-2-9b-it" in name:
        return "9b-it"
    if "Qwen3.5-9B" in name or "Qwen_Qwen3.5" in name or "Qwen--Qwen3.5" in name:
        return "qwen"
    # eval_model scheme -> model from dir
    if "eval_model_s" in name:
        if "ra9b" in dirpath:
            return "9b-it"
        if "qw35" in dirpath:
            return "qwen"
    return None


def vclass(name: str) -> str:
    if "eval_model_s" in name:
        return "v7-evalmodel"
    # strip the scores_<tc>- prefix; the next token is the version
    rest = PREFIX.sub("", name)
    if rest.startswith("v6-") or rest.startswith("v6_"):
        return "v6"
    if rest.startswith("v7-") or rest.startswith("v7b"):
        d = DELTA.search(name)
        if d and d.group(1).startswith("0.15"):
            return "v7b"
        return "v7-deltabins"
    return "other"


def main() -> None:
    # cov[(model,task,vclass,split,variant)] = set of (prompt-or-setting key)
    cov: dict[tuple, set] = defaultdict(set)
    dirs: dict[tuple, set] = defaultdict(set)
    for dirpath, _, files in os.walk(ROOT):
        b = os.path.basename(dirpath)
        if not any(p in dirpath for p in ("outputs",)):
            continue
        for name in files:
            if not name.startswith("scores_") or not name.endswith(".csv"):
                continue
            task = "ifeval" if "ifeval" in name else ("rosch" if "rosch" in name else None)
            if task is None:
                continue
            mp = PREFIX.match(name)
            variant = mp.group(1) if mp else "raw"
            model = model_of(name, dirpath)
            if model is None:
                continue
            vc = vclass(name)
            pm = PROMPT.search(name)
            if task == "ifeval" and pm:
                p = int(pm.group(1))
                split = "ood" if p <= 21 else "id"
                key = p
            else:
                split = "all"
                # rosch task key = the rosch subtask token after 'rosch'
                rm = re.search(r"rosch[-_]([a-z0-9]+)", name)
                key = rm.group(1) if rm else name
            sm = SETT.search(name)
            setting = f"s{sm.group(1)}" if sm else "full-name"
            cov[(model, task, vc, split, variant, setting)].add(key)
            dirs[(model, task, vc, split, variant, setting)].add(os.path.relpath(dirpath, ROOT))

    print("=== KEPT v7 cells (v7-evalmodel + v7-deltabins): distinct eval-task counts ===")
    for (model, task, vc, split, variant, setting), keys in sorted(cov.items()):
        if vc not in ("v7-evalmodel", "v7-deltabins"):
            continue
        ds = ",".join(sorted(d.split("/")[0] for d in dirs[(model, task, vc, split, variant, setting)]))
        print(f"  {model:6s} {task:6s} {vc:13s} {split:3s} {variant:11s} {setting:9s} "
              f"n={len(keys):3d}  dirs=[{ds}]")
    print("\n=== EXCLUDED (v6 / v7b / other) summary counts ===")
    excl: dict[tuple, int] = defaultdict(int)
    for (model, task, vc, split, variant, setting), keys in cov.items():
        if vc in ("v6", "v7b", "other"):
            excl[(model, task, vc)] += len(keys)
    for (model, task, vc), n in sorted(excl.items()):
        print(f"  {model:6s} {task:6s} {vc:6s}: {n}")


if __name__ == "__main__":
    main()
