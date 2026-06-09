#!/usr/bin/env python3
"""Recompute v7 eval metrics DIRECTLY from the raw v7 `eval_model_sN` score files.

Why: the pre-computed markdown tables (pod-results-*, v7_ra9b_results) are INCOMPLETE
for ifeval (literal `soon` placeholders), so the only trustworthy source is the raw
scores. This recomputes, per (model, task, setting, eval-TC variant, split), the mean
+/- SE over eval tasks/prompts of every metric, plus the coverage count (n tasks).

v7 ONLY. Sources are the `eval_model_sN` download dirs (these are the v7 delta-bins pod
evals; NOT v6, NOT v7b/delta-0.15):
  - gemma-2-9b-it ifeval : outputs_gemma4_from_pod-v7/ra9b_ifeval/
  - qwen3.5-9b   ifeval : outputs_gemma4_from_pod-v7/qw35_ifeval/
  - qwen3.5-9b   rosch  : outputs_gemma4_from_pod-v7/qw35_persona_member/  (membership eval)
(gemma-2-9b-it rosch is taken from docs/v7_rosch_all_metrics_*.md by the table builder,
 which is the complete, paper-matching v7 source; not recomputed here.)

eval-TC variant <- filename prefix (the file's gen_score_typcorr already carries it):
  self-  -> "PMI self" (own) ;  basetyp-    -> "PMI base"
  neg-   -> "Neg self" (own) ;  basetypneg- -> "Neg base"
  (Raw column = the 'raw' gen variant, prefix-independent.)

ifeval split: OOD = prompt id <= 21, ID = prompt id >= 22 (paper tables are OOD).
rosch: 10 tasks, no split.

Output CSV (one row per populated (model,task,split,setting,column,metric)):
  metrics-from-scores/v7_recompute_eval_metrics.csv
columns: model,task,split,setting,column,metric,mean,se,n

Run on mll:  source ~/venvs/venv_lexcons/bin/activate && python scripts/_recompute_v7_eval_metrics.py
"""
from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_scores_file import load_scores, compute_all_metrics  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
POD = REPO / "outputs_gemma4_from_pod-v7"
POD_V7B = REPO / "outputs_gemma4_from_pod-v7b"
OUT = REPO / "metrics-from-scores" / "v7_recompute_eval_metrics.csv"

# (model, task, dir, eval-task token, only_settings)
# only_settings=None -> all settings. The v7b folder is the delta-0.15 regime, so we take
# from it ONLY the delta-INSENSITIVE settings (SFT s1, CFT s13) — for those, the v7b eval IS
# the paper's actual qwen value (the paper's qwen SFT/CFT IFEval came from here: e.g. s1=73.1).
# Delta-sensitive settings (s2/s4/s7) come only from the delta-bins v7 folder.
SOURCES = [
    ("9b-it", "ifeval", POD / "ra9b_ifeval", "ifeval", None),
    ("qwen", "ifeval", POD / "qw35_ifeval", "ifeval", None),
    ("qwen", "rosch", POD / "qw35_persona_member", "rosch", None),
    ("qwen", "ifeval", POD_V7B / "qw35_ifeval", "ifeval", {1, 13}),
    ("qwen", "rosch", POD_V7B / "qw35_persona_member", "rosch", {1, 13}),
]
PREFIX_COL = {"self": "PMI self", "basetyp": "PMI base",
              "neg": "Neg self", "basetypneg": "Neg base"}
# gen variant feeding each column: own/base use the typ-corrected ('tc') score
COL_VARIANT = {"PMI self": "tc", "PMI base": "tc", "Neg self": "tc", "Neg base": "tc"}
METRICS = ["gen_roc", "spearman", "val_roc", "val_acc"]

_FNAME = re.compile(
    r"scores_(self|neg|basetyp|basetypneg)-eval_model_s(\d+)_([a-z0-9]+)[-_]"
    r"(?:prompt[_-](\d+)|.*?)"
)


def split_of(task: str, prompt: int | None) -> str:
    if task != "ifeval" or prompt is None:
        return "all"
    return "ood" if prompt <= 21 else "id"


def _taskkey(name: str, task: str) -> str:
    """Identity of the eval task, for dedup of the qw35 grab-bag re-runs."""
    if task == "ifeval":
        m = re.search(r"prompt[_-](\d+)", name)
        return f"p{m.group(1)}" if m else name
    m = re.search(r"(rosch[-_][a-z0-9-]+?)_test", name)
    return m.group(1) if m else name


def main() -> None:
    # nested: bucket[(model,task,split,setting,column)][taskkey] -> metric dict (deduped)
    bucket: dict[tuple, dict] = {}
    rawbucket: dict[tuple, dict] = {}
    valbucket: dict[tuple, dict] = {}

    for model, task, d, etok, only_settings in SOURCES:
        if not d.exists():
            print(f"  MISSING DIR {d}", file=sys.stderr)
            continue
        pats = [f"scores_{p}-eval_model_s*_{etok}*.csv" for p in PREFIX_COL]
        files = sorted({f for pat in pats for f in glob.glob(str(d / pat))})
        print(f"{model}/{task}: {len(files)} files in {d.name}"
              + (f" (settings {sorted(only_settings)})" if only_settings else ""))
        for fp in files:
            name = os.path.basename(fp)
            m = re.match(_FNAME, name)
            if not m:
                continue
            prefix, snum, ftask = m.group(1), int(m.group(2)), m.group(3)
            if ftask != etok:
                continue
            if only_settings is not None and snum not in only_settings:
                continue
            pm = re.search(r"prompt[_-](\d+)", name)
            prompt = int(pm.group(1)) if pm else None
            sp = split_of(task, prompt)
            col = PREFIX_COL[prefix]
            tk = _taskkey(name, task)
            try:
                mets = compute_all_metrics(load_scores(fp))
            except Exception as e:
                print(f"  SKIP {name}: {e}", file=sys.stderr)
                continue
            tcv = mets.get(COL_VARIANT[col])
            if tcv is not None and not np.isnan(tcv["gen_roc"]):
                bucket.setdefault((model, task, sp, snum, col), {})[tk] = tcv  # dedup: last wins
            rawv = mets.get("raw")
            if rawv is not None and not np.isnan(rawv["gen_roc"]):
                rawbucket.setdefault((model, task, sp, snum), {})[tk] = rawv
                valbucket.setdefault((model, task, sp, snum), {})[tk] = rawv

    # collapse the per-taskkey dicts to lists for aggregation
    bucket = {k: list(v.values()) for k, v in bucket.items()}
    rawbucket = {k: list(v.values()) for k, v in rawbucket.items()}
    valbucket = {k: list(v.values()) for k, v in valbucket.items()}

    def agg(dicts, key):
        vals = [d[key] for d in dicts if not np.isnan(d[key])]
        if not vals:
            return None
        mean = float(np.mean(vals))
        se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        return mean, se, len(vals)

    rows = []
    SCALE = {"gen_roc": 100, "spearman": 100, "val_roc": 100, "val_acc": 100}
    # eval-TC columns
    for (model, task, sp, snum, col), dicts in sorted(bucket.items()):
        for metric in ("gen_roc", "spearman"):
            a = agg(dicts, metric)
            if a:
                rows.append(dict(model=model, task=task, split=sp, setting=snum,
                                 column=col, metric=metric, mean=a[0] * SCALE[metric],
                                 se=a[1] * SCALE[metric], n=a[2]))
    # Raw column (prefix-independent)
    for (model, task, sp, snum), dicts in sorted(rawbucket.items()):
        for metric in ("gen_roc", "spearman"):
            a = agg(dicts, metric)
            if a:
                rows.append(dict(model=model, task=task, split=sp, setting=snum,
                                 column="Raw", metric=metric, mean=a[0] * SCALE[metric],
                                 se=a[1] * SCALE[metric], n=a[2]))
    # validator metrics (one per setting, prefix-independent)
    for (model, task, sp, snum), dicts in sorted(valbucket.items()):
        for metric in ("val_roc", "val_acc"):
            a = agg(dicts, metric)
            if a:
                rows.append(dict(model=model, task=task, split=sp, setting=snum,
                                 column="val", metric=metric, mean=a[0] * SCALE[metric],
                                 se=a[1] * SCALE[metric], n=a[2]))

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(df)} rows)")
    # coverage summary
    print("\n--- gen_roc coverage (model/task/split: setting=col(n) ...) ---")
    g = df[df.metric == "gen_roc"]
    for (model, task, sp), sub in g.groupby(["model", "task", "split"]):
        cells = ", ".join(f"s{r.setting}:{r.column.replace('PMI ','').replace('Neg ','n-')}={r['mean']:.1f}(n{r.n})"
                          for _, r in sub.iterrows())
        print(f"  {model}/{task}/{sp}: {cells}")


if __name__ == "__main__":
    main()
