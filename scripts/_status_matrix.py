#!/usr/bin/env python3
"""Build a comprehensive train/eval status matrix.

For each (dataset, model, setting) cell of interest, report:
  - whether a v7 trained model exists on disk (which epoch is latest)
  - whether train is in flight (slurm RUNNING/PENDING train job)
  - whether each TC eval variant has score CSVs on disk
  - whether each TC eval is in flight (slurm)

Output: human-readable markdown printed to stdout.
"""
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path("/datastor1/jdr/gv-gap/rankalign")
sys.path.insert(0, str(REPO / "src"))
from checkpoint_name_parser import parse_checkpoint_name  # type: ignore
MODELS_DIR = Path("/datastor2/jdr/rankalign/models2")
# Mirror SEARCH_DIRS from _build_rosch_table_v7.py:
SEARCH_DIRS = [
    REPO / "outputs",
    Path("/datastor2/jdr/rankalign/outputs"),
]

# ---- Dimensions of interest ----
DATASETS = {
    # dataset_short -> (train_task, eval_task_prefix)
    "membership": ("membership-sans-rosch-v0", "rosch-"),
    "persona":    ("persona-v1",               "persona-v1-"),
    "ifeval":     ("ifeval-concat",            "ifeval-prompt_"),
    "humaneval":  ("humaneval-v2.1correct-upper", "humaneval-v2.1correct-upper-"),
}
MODELS = ["gemma-2-2b-it", "gemma-2-9b-it", "gemma-4-31B-it"]
SETTINGS = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s11", "s12", "s13"]

# Each setting's expected directory-suffix flags (between alpha1.0 and --semi/--labelonly).
# Mirrors _overnight_launch.sh build_setting + _eval_only.sh.
SETTING_FLAGS = {
    "s1":  dict(tc="", pref="--pref0.0",  nllv="--nllv1.0", nllg="--nllg1.0", fsx="", ppd="", vlo="", cft="", semi="--labelonly0.1"),
    "s2":  dict(tc="", pref="",           nllv="",          nllg="",          fsx="", ppd="", vlo="", cft="", semi="--semi0.1"),
    "s3":  dict(tc="", pref="",           nllv="--nllv1.0", nllg="--nllg1.0", fsx="--force-same-x", ppd="--ppd", vlo="--vallogodds", cft="", semi="--semi0.1"),
    "s4":  dict(tc="--tc-self", pref="",  nllv="--nllv1.0", nllg="--nllg1.0", fsx="--force-same-x", ppd="--ppd", vlo="--vallogodds", cft="", semi="--semi0.1"),
    "s5":  dict(tc="--tc-self", pref="",  nllv="",          nllg="",          fsx="--force-same-x", ppd="--ppd", vlo="", cft="", semi="--semi0.1"),
    "s6":  dict(tc="--tc-self", pref="",  nllv="",          nllg="",          fsx="", ppd="", vlo="", cft="", semi="--semi0.1"),
    "s7":  dict(tc="--tc-neg",  pref="",  nllv="--nllv1.0", nllg="--nllg1.0", fsx="--force-same-x", ppd="--ppd", vlo="--vallogodds", cft="", semi="--semi0.1"),
    "s11": dict(tc="--tc-self", pref="",  nllv="--nllv1.0", nllg="--nllg1.0", fsx="", ppd="", vlo="--vallogodds", cft="", semi="--semi0.1"),
    "s12": dict(tc="--tc-neg",  pref="",  nllv="--nllv1.0", nllg="--nllg1.0", fsx="", ppd="", vlo="--vallogodds", cft="", semi="--semi0.1"),
    "s13": dict(tc="", pref="--pref0.0",  nllv="--nllv1.0", nllg="--nllg1.0", fsx="", ppd="", vlo="", cft="--cft", semi="--labelonly0.1"),
}

# Which TC variants are valid evals for a given setting:
# - no-TC trained (s1/s2/s3/s13): both self and neg
# - self-TC trained (s4/s5/s6/s11): self only
# - neg-TC trained (s7/s12): neg only
EVAL_TC_VARIANTS = {
    "s1":  ["self", "neg"],
    "s2":  ["self", "neg"],
    "s3":  ["self", "neg"],
    "s4":  ["self"],
    "s5":  ["self"],
    "s6":  ["self"],
    "s7":  ["neg"],
    "s11": ["self"],
    "s12": ["neg"],
    "s13": ["self", "neg"],
}

# CSV prefix when --base-typcorr is also passed (always is in v7 evals):
# TC=self -> "basetyp-" prefix, TC=neg -> "basetypneg-" prefix.
TC_TO_PREFIX = {"self": "basetyp-", "neg": "basetypneg-"}

# ---- Active jobs from squeue ----
def get_active_jobs():
    """Return list of (jobid, state, name)."""
    out = subprocess.check_output(
        ["squeue", "-u", "jdr", "-h", "-o", "%i|%T|%j"]).decode()
    jobs = []
    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        jobid, state, name = parts[0], parts[1], parts[2]
        jobs.append(dict(jobid=jobid, state=state, name=name))
    return jobs

# Map active jobs to (dataset, model, setting, kind, tc?)
def parse_active_job(job):
    """Returns dict with keys dataset/model/setting/kind/tc or None if not parseable."""
    name = job["name"]
    # Train jobs have name="wrap" (no info inline). Need to look at our own log.
    # Eval jobs: eval-s{N}-{dataset}-{tc}[-only]
    m = re.match(r"^eval-(s\d+)-(\w+)-(self|neg)(-only)?$", name)
    if m:
        return dict(setting=m.group(1), dataset=m.group(2), tc=m.group(3),
                    kind="eval", jobid=job["jobid"], state=job["state"])
    # Train jobs as "wrap" -- parse from _overnight_jobids.txt
    return None

def get_train_jobs_from_log():
    """Map jobid -> (dataset, model, setting) from overnight log."""
    log = REPO / "overnight" / "_overnight_jobids.txt"
    train_map = {}
    if not log.is_file():
        return train_map
    pattern = re.compile(
        r"\s+(\w+)-(gemma-[\w.-]+)-(s\d+)\s+TRAIN=(\d+)\s+TC_EVAL_LIST=")
    for line in log.read_text().splitlines():
        m = pattern.search(line)
        if m:
            ds, model, setting, jid = m.groups()
            train_map[jid] = dict(dataset=ds, model=model, setting=setting)
    return train_map

# ---- Scan trained model dirs ----
def parse_model_dir(path):
    """Return dict(model, dataset, setting, epoch, full=str) or None."""
    name = path.name
    is_merged = name.endswith("_merged")
    try:
        parsed = parse_checkpoint_name(name)
    except ValueError:
        return None
    if parsed.get("version") != "v7":
        return None
    if not parsed.get("fix1"):
        return None
    dataset = task_to_dataset(parsed["task_segment"])
    if dataset is None:
        return None
    setting = match_setting_from_parsed(parsed)
    if setting is None:
        return None
    return dict(model=parsed["model_short"], dataset=dataset, setting=setting,
                epoch=parsed["epoch"], full=str(path), merged=is_merged)

# Map setting -> structured fingerprint (parsed-dict expected values).
# We compare against the fields returned by parse_checkpoint_name(...).
SETTING_FINGERPRINT = {
    # tc, pref, nll_v, nll_g, force_same_x, ppd, vallogodds, cft, semi, labelonly
    "s1":  dict(tc=None,   pref=0.0, nll_v=1.0, nll_g=1.0, force_same_x=False, ppd=False, vallogodds=False, cft=False, semi=None,  labelonly=0.1),
    "s2":  dict(tc=None,   pref=1.0, nll_v=0.0, nll_g=0.0, force_same_x=False, ppd=False, vallogodds=False, cft=False, semi=0.1,  labelonly=None),
    "s3":  dict(tc=None,   pref=1.0, nll_v=1.0, nll_g=1.0, force_same_x=True,  ppd=True,  vallogodds=True,  cft=False, semi=0.1,  labelonly=None),
    "s4":  dict(tc="self", pref=1.0, nll_v=1.0, nll_g=1.0, force_same_x=True,  ppd=True,  vallogodds=True,  cft=False, semi=0.1,  labelonly=None),
    "s5":  dict(tc="self", pref=1.0, nll_v=0.0, nll_g=0.0, force_same_x=True,  ppd=True,  vallogodds=False, cft=False, semi=0.1,  labelonly=None),
    "s6":  dict(tc="self", pref=1.0, nll_v=0.0, nll_g=0.0, force_same_x=False, ppd=False, vallogodds=False, cft=False, semi=0.1,  labelonly=None),
    "s7":  dict(tc="neg",  pref=1.0, nll_v=1.0, nll_g=1.0, force_same_x=True,  ppd=True,  vallogodds=True,  cft=False, semi=0.1,  labelonly=None),
    "s11": dict(tc="self", pref=1.0, nll_v=1.0, nll_g=1.0, force_same_x=False, ppd=False, vallogodds=True,  cft=False, semi=0.1,  labelonly=None),
    "s12": dict(tc="neg",  pref=1.0, nll_v=1.0, nll_g=1.0, force_same_x=False, ppd=False, vallogodds=True,  cft=False, semi=0.1,  labelonly=None),
    "s13": dict(tc=None,   pref=0.0, nll_v=1.0, nll_g=1.0, force_same_x=False, ppd=False, vallogodds=False, cft=True,  semi=None,  labelonly=0.1),
}

def match_setting_from_parsed(parsed):
    """Given a parse_checkpoint_name() dict, return the setting key or None."""
    for s, fp in SETTING_FINGERPRINT.items():
        if all(parsed.get(k) == v for k, v in fp.items()):
            return s
    return None

def task_to_dataset(task_segment):
    """Map task_segment (e.g. 'membership-sans-rosch-v0-all') to dataset key."""
    # Strip "-all" suffix
    ts = task_segment[:-4] if task_segment.endswith("-all") else task_segment
    for ds, (train_task, _) in DATASETS.items():
        if ts == train_task:
            return ds
    return None

def scan_models():
    """Group trained model dirs by (dataset, model, setting). Track latest epoch."""
    found = defaultdict(list)
    for path in MODELS_DIR.iterdir():
        if not path.is_dir():
            continue
        info = parse_model_dir(path)
        if info is None:
            continue
        key = (info["dataset"], info["model"], info["setting"])
        found[key].append(info)
    # Pick max epoch per cell, prefer merged.
    summary = {}
    for key, lst in found.items():
        # If any merged exists, keep only those.
        merged = [i for i in lst if i["merged"]]
        if merged:
            lst = merged
        max_ep = max(i["epoch"] for i in lst)
        summary[key] = dict(max_epoch=max_ep, count=len(lst))
    return summary

# ---- Scan score CSVs ----
def scan_score_csvs():
    """Group score CSVs by (dataset, model, setting, tc_prefix). Count tasks per group."""
    found = defaultdict(set)  # (dataset, model, setting, tc_prefix) -> set of eval_tasks seen
    seen_basenames = set()
    files = []
    for d in SEARCH_DIRS:
        if not d.is_dir():
            continue
        files.extend(d.glob("scores_*test_log-odds*.csv"))
    for f in files:
        # Dedupe across SEARCH_DIRS (a basename in both = same eval).
        if f.name in seen_basenames:
            continue
        seen_basenames.add(f.name)
        name = f.name[len("scores_"):]
        # Strip the tc-prefix (one of the four), if any.
        tc_prefix = ""
        for p in ("basetypneg-", "basetyp-", "self-", "neg-"):
            if name.startswith(p):
                tc_prefix = p
                name = name[len(p):]
                break
        # Now name is "{model_short}_{eval_task}_test_log-odds...csv"
        # Strip trailing "_test_log-odds...csv" and split on the LAST underscore
        # before that — that delimiter separates model_short from eval_task.
        # (Format-A model_shorts contain many underscores, so we cannot use a
        # greedy `.+_(task)_test_log-odds` regex naively without anchoring.)
        idx = name.rfind("_test_log-odds")
        if idx < 0:
            continue
        head = name[:idx]  # "{model_short}_{eval_task}"
        # The eval_task is the last "_-separated" component. But task names
        # contain dashes (e.g. rosch-bird). Walk backwards through the known
        # task prefixes.
        task = None
        for ds_short, (_, ev_pfx) in DATASETS.items():
            # Find the LAST occurrence of "_{prefix}" in head; the suffix from
            # there to end of head is the eval_task.
            sep = "_" + ev_pfx
            j = head.rfind(sep)
            if j > 0:
                task = head[j + 1:]
                model_short = head[:j]
                break
        if task is None:
            continue
        # Reverse-map task -> dataset (find which dataset's eval-task-prefix matches)
        dataset = None
        for ds_short, (_, ev_pfx) in DATASETS.items():
            if task.startswith(ev_pfx):
                dataset = ds_short
                break
        if dataset is None:
            continue
        # Parse model_short to get (model, setting).
        cell = parse_model_short(model_short)
        if cell is None:
            continue
        cell_dataset, cell_model, cell_setting = cell
        if cell_dataset != dataset:
            continue  # mismatched eval task vs train task
        found[(dataset, cell_model, cell_setting, tc_prefix)].add(task)
    return found

def parse_model_short(model_short):
    """Parse a v7 model_short into (dataset, model, setting). Handles A/B/C forms via
    checkpoint_name_parser.parse_checkpoint_name. Returns None if unparseable."""
    try:
        parsed = parse_checkpoint_name(model_short)
    except ValueError:
        return None
    if parsed.get("version") != "v7":
        return None
    if not parsed.get("fix1"):
        return None
    dataset = task_to_dataset(parsed["task_segment"])
    if dataset is None:
        return None
    setting = match_setting_from_parsed(parsed)
    if setting is None:
        return None
    # Format C ("abbreviated") doesn't recover the full HF model_name; model_short
    # is the bare model id (e.g. "gemma-2-9b-it") in both formats B and C.
    return (dataset, parsed["model_short"], setting)

# ---- Render ----
def render(models_on_disk, csvs, active_jobs, train_log_map):
    # active_jobs: list of dicts; train_log_map: jobid -> {dataset, model, setting}
    # Build per-(dataset, model, setting) active state.
    train_active = {}  # key -> (jobid, state)
    eval_active = defaultdict(dict)  # key -> {tc: (jobid, state)}
    for j in active_jobs:
        ev = parse_active_job(j)
        if ev:
            key = (ev["dataset"], None, ev["setting"])  # model unknown from name
            # Try to pin model from train_log_map of pending dependency
            # For simplicity, store eval entries keyed by (dataset, setting, tc, jobid).
            # We'll cross-reference via the train_log when rendering.
            eval_active[(ev["dataset"], ev["setting"], ev["tc"])].setdefault("jobs", []).append(
                dict(jobid=ev["jobid"], state=ev["state"])
            )
        elif j["name"] == "wrap":
            jid = j["jobid"]
            if jid in train_log_map:
                tl = train_log_map[jid]
                key = (tl["dataset"], tl["model"], tl["setting"])
                train_active[key] = (jid, j["state"])
    # Number of eval tasks expected per dataset (for "X/N" rendering)
    EXPECTED_TASKS = {
        "membership": 10,
        "persona":    6,
        "ifeval":     21,
        "humaneval":  None,  # variable; we'll just print count
    }
    print("# Status matrix — train + eval inventory")
    print()
    print("Generated by `python scripts/_status_matrix.py`. Reads `squeue`, model dirs in")
    print(f"`{MODELS_DIR}`, and score CSVs in {[str(d) for d in SEARCH_DIRS]}.")
    print()
    print("**Caveat**: this matrix counts only CSVs whose `model_short` parses cleanly")
    print("with `checkpoint_name_parser.parse_checkpoint_name` (Format A/B/C). It")
    print("**does NOT** match md5-hash-truncated CSVs from the brief broken-eval window")
    print("(filenames ending in `..-_<8hex>_<task>_test_log-odds...csv`). The v7 table")
    print("builders handle those via `_resolve_full_basename`. Cells where the")
    print("matrix shows fewer files than the table builder's `metrics-from-scores/*_long.csv`")
    print("output are likely affected — known cases: mem/2b-it/s11/self, mem/9b-it/s5/self.")
    print()
    print("Legend:")
    print("- T (train): `done(epochN)` / `running` / `none`")
    print("- E_self / E_neg: `X/N` count of eval-task CSVs found "
          "(N = expected task count for that dataset). `pend(JOBID)` if the eval is queued/running.")
    print("- `n/a` = TC variant not applicable for this setting (e.g. neg-TC for s4 self-TC trained model).")
    print()
    for ds in DATASETS:
        ntasks = EXPECTED_TASKS.get(ds)
        ntasks_str = f"{ntasks}" if ntasks else "?"
        print(f"## {ds}  (eval tasks expected per cell: {ntasks_str})")
        print()
        print("| model | s# | T | E_self | E_neg |")
        print("|---|---|---|---|---|")
        for model in MODELS:
            for setting in SETTINGS:
                # Skip combinations that don't make sense
                if ds == "humaneval" and model != "gemma-4-31B-it":
                    continue
                if ds != "humaneval" and model == "gemma-4-31B-it":
                    continue
                if ds == "humaneval" and setting != "s13":
                    # We only ran s13 on humaneval per request
                    continue
                key = (ds, model, setting)
                # Train state
                if key in train_active:
                    jid, st = train_active[key]
                    t_str = f"{st.lower()}({jid})"
                elif key in models_on_disk:
                    info = models_on_disk[key]
                    t_str = f"done(e{info['max_epoch']})"
                else:
                    t_str = "—"
                # Eval state (per TC variant)
                evs = []
                for tc in ["self", "neg"]:
                    if tc not in EVAL_TC_VARIANTS[setting]:
                        evs.append("n/a")
                        continue
                    pfx = TC_TO_PREFIX[tc]
                    csv_key = (ds, model, setting, pfx)
                    n_csvs = len(csvs.get(csv_key, set()))
                    # Active eval for this cell?
                    ev_key = (ds, setting, tc)
                    pending_ids = []
                    for j in eval_active.get(ev_key, {}).get("jobs", []):
                        # Filter to model: hard to disambiguate with squeue name, but
                        # if there's only one cell with this (ds,setting,tc), it's fine.
                        # For now, append all.
                        pending_ids.append(f"{j['state'][:1].lower()}{j['jobid']}")
                    if n_csvs > 0:
                        s = f"{n_csvs}"
                        if ntasks:
                            s = f"{n_csvs}/{ntasks}"
                    else:
                        s = "—"
                    if pending_ids:
                        s += f" [+{','.join(pending_ids)}]"
                    evs.append(s)
                print(f"| {model} | {setting} | {t_str} | {evs[0]} | {evs[1]} |")
        print()
    # Show counts of active jobs
    print("## Slurm queue right now")
    print()
    print(f"- Total active jobs: {len(active_jobs)}")
    print(f"- Training jobs (running): "
          f"{sum(1 for j in active_jobs if j['name']=='wrap' and j['state']=='RUNNING')}")
    print(f"- Eval jobs running: "
          f"{sum(1 for j in active_jobs if j['name'].startswith('eval-') and j['state']=='RUNNING')}")
    print(f"- Eval jobs pending: "
          f"{sum(1 for j in active_jobs if j['name'].startswith('eval-') and j['state']=='PENDING')}")

def main():
    models = scan_models()
    csvs = scan_score_csvs()
    active = get_active_jobs()
    tlog = get_train_jobs_from_log()
    render(models, csvs, active, tlog)

if __name__ == "__main__":
    main()
