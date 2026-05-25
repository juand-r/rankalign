#!/usr/bin/env python3
"""s13 inventory — corrected to disambiguate via chained-eval job names.

Outputs:
- Train state per (ds, model) cell: DONE (epochs) | RUNNING (jobid) | MISSING.
- Eval coverage: which prefixes (basetyp/basetypneg/self/neg) have CSVs on disk.
- In-flight evals queued for that cell.
"""
import re, subprocess
from pathlib import Path

TARGETS = []
for ds in ("membership", "persona", "ifeval", "humaneval"):
    for model in ("gemma-2-2b", "gemma-2-2b-it", "gemma-2-9b-it", "gemma-4-31B-it"):
        if ds == "ifeval"    and model != "gemma-2-9b-it":  continue
        if ds == "humaneval" and model != "gemma-4-31B-it": continue
        if ds == "membership" and model == "gemma-4-31B-it": continue
        if ds == "persona"    and model == "gemma-4-31B-it": continue
        TARGETS.append((ds, model))

DS2DIR = {
    "membership": "membership-sans-rosch-v0",
    "persona":    "persona-v1",
    "ifeval":     "ifeval-concat",
    "humaneval":  "humaneval-v2.1correct-upper",
}

MODELS_BASE = Path("/datastor2/jdr/rankalign/models2")
OUTPUT_DIRS = [Path("/datastor2/jdr/rankalign/outputs"),
               Path("/datastor1/jdr/gv-gap/rankalign/outputs")]

def model_dirs_for(ds, model):
    pat = re.compile(
        rf"^v7-google--{re.escape(model)}-delta[\d.]+-epoch(\d+)--{re.escape(DS2DIR[ds])}-all"
        rf"--d2g--random--alpha1\.0--full-completion--pref0\.0--nllv1\.0--nllg1\.0--cft--labelonly0\.1--fix1$"
    )
    epochs = set()
    for entry in MODELS_BASE.iterdir() if MODELS_BASE.exists() else []:
        m = pat.match(entry.name)
        if m:
            epochs.add(int(m.group(1)))
    return sorted(epochs)

def score_inventory(ds, model):
    """Return {(prefix, epoch): set_of_unique_test_tasks}.

    Handles both long form (`v7-google--<model>-delta<X>-epoch<Y>--<ds>-all`) and
    short form (`v7-<model>-d<X>-e<Y>-<ds>-all`). The cft-vs-non-cft distinction
    comes from `--cft--` (long) or `-cft-` (short) somewhere in the model fingerprint.
    """
    long_pat = re.compile(
        rf"^scores_(basetyp|basetypneg|self|neg)-v7-google--{re.escape(model)}-delta[\d.]+-epoch(\d+)--{re.escape(DS2DIR[ds])}-all--.*?--cft--.*?_([A-Za-z0-9._-]+?)_test_"
    )
    short_pat = re.compile(
        rf"^scores_(basetyp|basetypneg|self|neg)-v7-{re.escape(model)}-d[\d.]+-e(\d+)-{re.escape(DS2DIR[ds])}-all-.*?-cft-.*?_([A-Za-z0-9._-]+?)_test_"
    )
    by_prefix_epoch = {}  # (prefix, epoch) -> set of test tasks
    for base in OUTPUT_DIRS:
        if not base.exists():
            continue
        for fname in base.iterdir():
            for pat in (long_pat, short_pat):
                m = pat.match(fname.name)
                if m:
                    key = (m.group(1), int(m.group(2)))
                    by_prefix_epoch.setdefault(key, set()).add(m.group(3))
                    break
    return by_prefix_epoch

def queue_state():
    """Return {jobid: {state, name, parent, dataset, setting, model_tag}} for my squeue."""
    out = subprocess.check_output(
        ["squeue", "-u", "jdr", "-h", "-o", "%i|%t|%j|%R"], text=True
    ).strip().split("\n")
    rows = {}
    for line in out:
        if not line.strip():
            continue
        jobid, state, name, reason = line.split("|", 3)
        rows[jobid] = {"jobid": jobid, "state": state, "name": name, "reason": reason}

    # for PD eval-* jobs, look up dependency parent.
    for jid, row in list(rows.items()):
        if row["state"] != "PD" or not row["name"].startswith("eval-s"):
            continue
        try:
            dep = subprocess.check_output(
                ["scontrol", "show", "job", jid], text=True
            )
            m = re.search(r"afterany:(\d+)", dep)
            if m:
                row["parent"] = m.group(1)
        except Exception:
            pass

    # parse dataset/setting/model_tag from eval names. Only treat known
    # model_tags as model_tags (otherwise it's a chained-eval name with
    # tc-suffix only).
    KNOWN_TAGS = ("g431Bit", "9b-it", "2b-it", "2b")
    for row in rows.values():
        n = row["name"]
        m = re.match(r"eval-(s\d+)-(membership|persona|ifeval|humaneval)(?:-(.+?))?(-only)?$", n)
        if m:
            row["setting"] = m.group(1)
            row["dataset"] = m.group(2)
            tail = m.group(3) or ""
            tag = ""
            for kt in KNOWN_TAGS:
                if tail.startswith(kt + "-") or tail == kt:
                    tag = kt
                    break
            row["model_tag"] = tag
    return rows

def setting_of(parent_jobid, jobs):
    """Find a chained eval whose parent==parent_jobid; return its setting/dataset/model_tag."""
    for j in jobs.values():
        if j.get("parent") == parent_jobid:
            return j.get("setting"), j.get("dataset"), j.get("model_tag")
    return (None, None, None)

def in_flight_train_for(ds, model, jobs):
    """Return list of running 'wrap' jobs whose chained evals say (ds, model, s13)."""
    model_tag_map = {"gemma-2-2b": "2b", "gemma-2-2b-it": "2b-it",
                     "gemma-2-9b-it": "9b-it", "gemma-4-31B-it": "g431Bit"}
    target_tag = model_tag_map[model]
    out = []
    for j in jobs.values():
        if j["state"] != "R" or j["name"] != "wrap":
            continue
        s, d, tag = setting_of(j["jobid"], jobs)
        if s == "s13" and d == ds:
            # confirm via stdout that it's the right model (chained-eval names from
            # _overnight_launch.sh don't include model tag, so we cross-check).
            log = Path(f"/datastor2/jdr/logs/{j['jobid']}.out")
            if log.exists():
                try:
                    head = log.read_text(errors="replace")[:500]
                    if f"google/{model}" in head and DS2DIR[ds] in head:
                        out.append(j)
                except Exception:
                    pass
    return out

def in_flight_evals_for(ds, model, jobs):
    """eval-only jobs are R; chained evals are PD. Match by setting/dataset/model_tag."""
    model_tag_map = {"gemma-2-2b": "2b", "gemma-2-2b-it": "2b-it",
                     "gemma-2-9b-it": "9b-it", "gemma-4-31B-it": "g431Bit"}
    target_tag = model_tag_map[model]
    out = []
    for j in jobs.values():
        if not j["name"].startswith("eval-s13-"):
            continue
        if j.get("dataset") != ds:
            continue
        # If model_tag in the name, must match. Otherwise it's a chained eval —
        # use the parent train's stdout/model.
        nm_tag = j.get("model_tag", "")
        if nm_tag:
            if nm_tag == target_tag:
                out.append(j)
        else:
            parent = j.get("parent")
            if parent and parent in jobs:
                log = Path(f"/datastor2/jdr/logs/{parent}.out")
                if log.exists():
                    head = log.read_text(errors="replace")[:500]
                    if f"google/{model}" in head:
                        out.append(j)
    return out

def fmt_train_state(ds, model, jobs):
    epochs = model_dirs_for(ds, model)
    in_flight = in_flight_train_for(ds, model, jobs)
    parts = []
    if epochs:
        parts.append(f"DONE epochs={epochs}")
    if in_flight:
        parts.append(f"RUNNING ({in_flight[0]['jobid']})")
    if not parts:
        return "MISSING"
    return ", ".join(parts)

def fmt_evals(ds, model):
    inv = score_inventory(ds, model)
    if not inv:
        return "(none)"
    parts = []
    by_prefix = {}
    for (pref, ep), tasks in inv.items():
        by_prefix.setdefault(pref, []).append((ep, len(tasks)))
    for p in ("basetyp", "basetypneg", "self", "neg"):
        if p in by_prefix:
            ep_strs = [f"e{e}={n}" for e, n in sorted(by_prefix[p])]
            parts.append(f"{p}[{','.join(ep_strs)}]")
    return ", ".join(parts) if parts else "(none)"

def fmt_inflight_evals(ds, model, jobs):
    rows = in_flight_evals_for(ds, model, jobs)
    if not rows:
        return "-"
    return ",".join(j['jobid'] for j in rows)

def main():
    jobs = queue_state()

    print("# s13 (consistency-FT) inventory — 2026-05-25")
    print()
    print(f"{'Dataset':<11}{'Model':<18}{'Train':<26}{'Evals (per epoch)':<55}{'Eval-PD/R'}")
    print("-" * 120)
    for ds, model in TARGETS:
        train = fmt_train_state(ds, model, jobs)
        evs = fmt_evals(ds, model)
        pd_jobs = fmt_inflight_evals(ds, model, jobs)
        print(f"{ds:<11}{model:<18}{train:<26}{evs:<55}{pd_jobs}")

    print()
    print("# Legend")
    print("- Train: 'DONE epochs=[0,1,2]' = 3 checkpoints on disk.")
    print("- Evals: prefix[e<E>=<count>] = count of CSVs at that epoch (one per per-prompt task).")
    print("    Expected per-epoch counts: membership=10 rosch, persona ID=6, persona ID+OOD=18, ifeval=21, humaneval≈variable.")
    print("- Eval-PD/R = currently queued / running eval job IDs for this cell.")

if __name__ == "__main__":
    main()
