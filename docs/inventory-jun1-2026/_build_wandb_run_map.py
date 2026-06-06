#!/usr/bin/env python
"""Map wandb cloud runs -> final paper models (per task x model x setting).

Input:  _raw/wandb_paper_runs.json (210 runs in the 3 paper tasks, dumped from
        wandb cloud juand-r/rankalign with full args/state/url).
Output: WANDB_RUN_MAP.md — for each (task, model, setting): the finished run(s)
        with name/date/delta-regime/URL, plus a count of failed/other-state runs.

Setting is classified from the run's actual flags (force-same-x / typicality /
vlo / cft / pref+nll), per SETTINGS_REFERENCE.md.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from _inv_common import classify_setting, norm_task, setting_sort_key, SETTING_NAME

HERE = Path(__file__).parent
runs = json.loads((HERE / "_raw" / "wandb_paper_runs.json").read_text())

TASK_LABEL = {"ifeval": "ifeval", "membership": "rosch / membership", "humaneval-cu": "humaneval-cu"}
MODEL_ORDER = ["gemma-2-2b", "gemma-2-2b-it", "gemma-2-9b-it", "Qwen3.5-9B", "gemma-4-31B-it"]


def setting_of(r: dict) -> str:
    toks = set()
    if r.get("pref") in (0, 0.0, "0", "0.0"):
        toks.add("pref0.0")
    if r.get("nllv") not in (None, 0, 0.0, "0", "0.0"):
        toks.add("nllv1.0")
    if r.get("nllg") not in (None, 0, 0.0, "0", "0.0"):
        toks.add("nllg1.0")
    if r.get("fsx"):
        toks.add("force-same-x")
    if r.get("self_tc"):
        toks.add("tc-self")
    if r.get("neg_tc"):
        toks.add("tc-neg")
    if r.get("cft"):
        toks.add("cft")
    return classify_setting(toks)


# group: (task, model, setting) -> list of run dicts (annotated)
grp: dict = defaultdict(list)
def _dval(r):
    try:
        return f"{float(r.get('delta')):.2f}"
    except (TypeError, ValueError):
        return "?"


for r in runs:
    r["_setting"] = setting_of(r)
    r["_task"] = norm_task(r.get("task") or "")
    r["_delta_regime"] = "delta-bins(v7)" if r.get("delta_bins") else "fixed-delta(v7b/early)"
    # compact per-run delta string, e.g. "bins δ2.69" (realized) or "fixed δ0.15"
    r["_delta_str"] = ("bins δ" + _dval(r)) if r.get("delta_bins") else ("fixed δ" + _dval(r))
    grp[(r["_task"], r["model"], r["_setting"])].append(r)


def delta_summary(fin_runs):
    """Distinct delta regimes+values among a cell's finished runs, e.g. 'bins δ2.69 | fixed δ0.15'.
    The bins(v7) entry is the canonical paper run; fixed is v7b/early."""
    seen, order = set(), []
    for r in sorted(fin_runs, key=lambda r: (not r.get("delta_bins"), r.get("_delta_str", ""))):
        s = r["_delta_str"]
        if s not in seen:
            seen.add(s); order.append(s)
    return " \\| ".join(order) if order else "—"

out = ["# WandB run → final-model map (paper datasets)", "",
       "For each **paper** dataset × model × setting: the wandb cloud runs in "
       "`juand-r/rankalign`, so you can open the training curves for any final model. "
       "Built from `_raw/wandb_paper_runs.json` by `_build_wandb_run_map.py`.",
       "",
       "- **Setting** classified from each run's real flags (force-same-x / typicality / vlo / cft / pref+nll).",
       "- **state**: prefer `finished`; `failed`/`crashed`/`running` runs are counted but not the model source.",
       "- **delta regime** (NB: all are v7-code `ranking_loss_ref_fix.py`; none are v6): "
       "`bins δX` = canonical **delta-bins(v7)**, realized adaptive delta X (matches the on-disk "
       "model name, e.g. `delta2.69`); `fixed δ0.15` = **v7b / pre-delta-bins** fixed delta. The "
       "**`delta (finished)`** column lists which regimes a cell's finished runs cover — the paper "
       "model is normally the **bins** one.",
       "- A cell with several finished runs = re-runs / a delta sweep; newest first.",
       ""]

tasks_present = [t for t in ["ifeval", "membership", "humaneval-cu"] if any(k[0] == t for k in grp)]
for task in tasks_present:
    out.append(f"\n## {TASK_LABEL.get(task, task)}\n")
    out.append("| Model | Setting | finished | other states | delta (finished) | newest finished run (date · regime) | URL(s) of finished runs |")
    out.append("|---|---|---|---|---|---|---|")
    models = [m for m in MODEL_ORDER if any(k[0] == task and k[1] == m for k in grp)]
    models += sorted({k[1] for k in grp if k[0] == task and k[1] not in MODEL_ORDER})
    for model in models:
        keys = sorted([k for k in grp if k[0] == task and k[1] == model],
                      key=lambda k: setting_sort_key(k[2]))
        for (_t, _m, setting) in keys:
            rs = grp[(task, model, setting)]
            fin = sorted([r for r in rs if r.get("state") == "finished"],
                         key=lambda r: r.get("created", ""), reverse=True)
            other = [r for r in rs if r.get("state") != "finished"]
            if fin:
                newest = fin[0]
                newest_str = f"`{newest['name']}` ({newest.get('created','?')} · {newest['_delta_regime']})"
                urls = " ".join(f"[{r['id']}]({r['url']})" for r in fin[:4])
                if len(fin) > 4:
                    urls += f" …(+{len(fin)-4})"
            else:
                newest_str = "— (no finished run)"
                # still give a URL to inspect the latest attempt
                latest_other = sorted(other, key=lambda r: r.get("created", ""), reverse=True)
                urls = (f"(latest {latest_other[0]['state']}: [{latest_other[0]['id']}]({latest_other[0]['url']}))"
                        if latest_other else "—")
            sname = SETTING_NAME.get(setting, setting)
            dsum = delta_summary(fin) if fin else "—"
            out.append(f"| {model} | {setting} ({sname}) | {len(fin)} | {len(other)} | {dsum} | {newest_str} | {urls} |")

# summary of states overall
from collections import Counter
st = Counter(r.get("state") for r in runs)
out.append("\n## Run-state summary (all 210 paper-task runs)\n")
out.append("| state | count |\n|---|---|")
for s, n in st.most_common():
    out.append(f"| {s} | {n} |")
out.append("\n> Many runs are `failed`/`crashed` (OOM, preemption, early bugs). Only `finished` runs "
           "carry complete curves. If a cell shows finished=0, the curves for that exact model are "
           "incomplete in wandb even though the trained checkpoint exists (see training_inventory).")

(HERE / "WANDB_RUN_MAP.md").write_text("\n".join(out) + "\n")
print("wrote WANDB_RUN_MAP.md")
print("cells:", len(grp), "| states:", dict(st))
