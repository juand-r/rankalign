#!/usr/bin/env python
"""TASK 1c: map each training-run JSON log to its setting.

Reads _raw/training_logs_flags.jsonl (dumped from
/datastor2/jdr/rankalign/models2/training_run_logs/*.json) and classifies each
log's setting from its recorded flags, then writes v7/training_logs_map_v7.md.

Family rule (flags only; loss-weights not stored in the JSON `flags` dict, but
the family is recoverable from vlo + semi/labeled_only):
  labeled_only set        -> SFT family  (s13 if consistency_ft else s1)
  vlo True                 -> comb family ("New")
  vlo False/None + semi    -> pref-only family ("RankAlign")
then split by fsx + self/neg typicality (see SETTINGS_REFERENCE.md).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from _inv_common import norm_model, norm_task, setting_sort_key, SETTING_NAME

HERE = Path(__file__).parent
RAW = HERE / "_raw"


def classify(rec: dict) -> str:
    fsx = bool(rec.get("fsx"))
    vlo = bool(rec.get("vlo"))
    self_tc = bool(rec.get("self_tc"))
    neg_tc = bool(rec.get("neg_tc"))
    lo = rec.get("lo") is not None
    cft = bool(rec.get("cft"))
    if lo:  # SFT family
        return "s13" if cft else "s1"
    if vlo:  # comb family
        if fsx:
            return "s4" if self_tc else "s7" if neg_tc else "s3"
        return "s11" if self_tc else "s12" if neg_tc else "comb-notc-nofsx?"
    # pref-only / RankAlign
    if fsx:
        return "s5" if self_tc else "s8" if neg_tc else "s10"
    return "s6" if self_tc else "s9" if neg_tc else "s2"


def main() -> None:
    recs = [json.loads(l) for l in (RAW / "training_logs_flags.jsonl").read_text().splitlines() if l.strip()]
    # group: (model, task, setting) -> [ (timestamp-from-filename, file, delta_arg) ]
    grp: dict = defaultdict(list)
    for r in recs:
        if r.get("error"):
            continue
        setting = classify(r)
        model = norm_model((r.get("model") or "").split("/")[-1])
        task = norm_task(r.get("task") or "")
        ts = r["file"].split("_")[0]
        grp[(model, task, setting)].append((ts, r["file"], r.get("delta_arg"), r.get("cft")))

    out = ["# Training-run JSON logs -> setting map (TASK 1c)", "",
           "Each v7 training run wrote a provenance JSON to "
           "`/datastor2/jdr/rankalign/models2/training_run_logs/<timestamp>_<model>_<task>.json`. "
           "The **setting is not in the filename** — it's classified here from the JSON's recorded "
           "`flags` (force_same_x, validator_log_odds, self/neg_typicality, semi/labeled_only, "
           "consistency_ft). 85 logs total.",
           "",
           "Each JSON records: delta config (`delta_arg`, `delta_bins`), shape-budget weights, "
           "label partition, sampled-pair counts, score stats, seed, and (s13) the `consistency_ft` "
           "filter block. To read one: `jq . <file>` on mll.",
           "",
           "> Multiple timestamps under one (model,task,setting) = re-runs / delta-sweep points. "
           "> These logs are gemma-2 / qwen runs (the 2026-05-24+ overnight batch). gemma-4 cu runs "
           "> logged separately — see `../provenance-cu-s2-rankalign/models_g4it/training_run_logs/`.",
           ""]
    models = sorted({k[0] for k in grp})
    for model in models:
        out.append(f"\n## {model}\n")
        out.append("| Task | Setting | # logs | Training JSON log file(s) |")
        out.append("|---|---|---|---|")
        keys = sorted([k for k in grp if k[0] == model],
                      key=lambda k: (k[1], setting_sort_key(k[2])))
        for (m, task, setting) in keys:
            items = sorted(grp[(m, task, setting)])
            files = "<br>".join(f"`{f}`" for _, f, _, _ in items)
            sname = SETTING_NAME.get(setting, setting)
            out.append(f"| {task} | {setting} ({sname}) | {len(items)} | {files} |")
    (HERE / "v7" / "training_logs_map_v7.md").write_text("\n".join(out) + "\n")
    n = sum(len(v) for v in grp.values())
    print(f"mapped {n} logs into {len(grp)} (model,task,setting) cells across {len(models)} models")


if __name__ == "__main__":
    main()
