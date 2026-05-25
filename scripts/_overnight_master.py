#!/usr/bin/env python3
"""Overnight master scheduler.

Run from the wake loop every ~hour. Two passes:

  Pass 1: For every trained model on disk, check which TC-eval prefixes are
          missing in `outputs/scores_*.csv`, and submit the missing eval jobs.
          This catches:
            - non-TC trains (s1, s2, s3, s13) that need all 4 prefixes
              (basetyp-, self-, basetypneg-, neg-) for tables.
            - TC-trained settings whose `_overnight_launch.sh` chained eval
              only wrote the `basetyp[neg]-` prefix and didn't produce the
              `self-`/`neg-` (NO_BASE) variant.

  Pass 2: For the next-priority (DATASET, MODEL, SETTING) gap, submit a fresh
          train + eval bundle via `_overnight_launch.sh`, subject to the slurm
          queue cap.

Constraints:
  - Total slurm queue (PD+R) must not exceed QUEUE_CAP (default 30).
  - Pass 1 takes precedence over Pass 2 (eval gaps are cheap).
"""

import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/datastor1/jdr/gv-gap/rankalign")
MODELS_DIR = Path("/datastor2/jdr/rankalign/models2")
SCORES_DIR = REPO / "outputs"
LOGS_DIR = Path("/datastor2/jdr/logs")
SCRIPT_DIR = REPO / "scripts"

# Paths added to sys.path so we can import the audit + parser helpers.
sys.path.insert(0, str(REPO / "scripts"))

QUEUE_CAP = int(os.environ.get("QUEUE_CAP", "30"))
DRYRUN = os.environ.get("DRYRUN") == "1"

# Map dataset short-name <-> long-name.
DS_LONG = {
    "membership": "membership-sans-rosch-v0",
    "persona": "persona-v1",
    "ifeval": "ifeval-concat",
    "humaneval": "humaneval-v2.1correct-upper",
}
DS_SHORT = {v: k for k, v in DS_LONG.items()}

# Setting -> what TC-eval prefixes the table NEEDS:
#   {basetyp-, self-, basetypneg-, neg-}
# For TC-trained settings, only the matching variant is needed.
SETTING_NEEDS = {
    # non-TC: all 4 prefixes
    "s1":  {"basetyp-", "self-", "basetypneg-", "neg-"},
    "s2":  {"basetyp-", "self-", "basetypneg-", "neg-"},
    "s3":  {"basetyp-", "self-", "basetypneg-", "neg-"},
    "s13": {"basetyp-", "self-", "basetypneg-", "neg-"},
    # tc-self: only PMI variants
    "s4":  {"basetyp-", "self-"},
    "s5":  {"basetyp-", "self-"},
    "s6":  {"basetyp-", "self-"},
    "s11": {"basetyp-", "self-"},
    # tc-neg: only Neg variants
    "s7":  {"basetypneg-", "neg-"},
    "s12": {"basetypneg-", "neg-"},
}

# Eval-task list per dataset (used to count expected CSVs per cell).
DATASET_TASKS = {
    "membership": [
        "rosch-bird", "rosch-carpenters-tool", "rosch-clothing", "rosch-fruit",
        "rosch-furniture", "rosch-sport", "rosch-toy", "rosch-vehicle",
        "rosch-vegetable", "rosch-weapon",
    ],
    "persona": [
        "persona-v1-psychopathy", "persona-v1-machiavellianism",
        "persona-v1-narcissism", "persona-v1-desire-to-create-allies",
        "persona-v1-interest-in-music", "persona-v1-interest-in-science",
    ],
    "ifeval": [f"ifeval-prompt_{n}" for n in range(1, 22)],
    # humaneval: dynamic, leave blank for now
    "humaneval": [],
}

# Priority gap list (DATASET, MODEL, SETTING, WALLTIME_HOURS).
GAP_LIST = [
    ("persona",    "gemma-2-9b-it",  "s3",  9),
    ("ifeval",     "gemma-2-2b-it",  "s1",  6),
    ("ifeval",     "gemma-2-2b-it",  "s2",  6),
    ("ifeval",     "gemma-2-2b-it",  "s3",  7),
    ("ifeval",     "gemma-2-2b-it",  "s4",  7),
    ("ifeval",     "gemma-2-2b-it",  "s7",  7),
    ("ifeval",     "gemma-2-2b-it",  "s5",  7),
    ("ifeval",     "gemma-2-2b-it",  "s6",  7),
    ("ifeval",     "gemma-2-2b",     "s1",  6),
    ("ifeval",     "gemma-2-2b",     "s2",  6),
    ("ifeval",     "gemma-2-2b",     "s3",  7),
    ("ifeval",     "gemma-2-2b",     "s4",  7),
    ("ifeval",     "gemma-2-2b",     "s7",  7),
    ("ifeval",     "gemma-2-2b",     "s5",  7),
    ("ifeval",     "gemma-2-2b",     "s6",  7),
    ("membership", "gemma-2-2b",     "s1",  6),
    ("membership", "gemma-2-2b",     "s2",  6),
    ("membership", "gemma-2-2b",     "s3",  7),
    ("membership", "gemma-2-2b",     "s4",  7),
    ("membership", "gemma-2-2b",     "s7",  7),
    ("membership", "gemma-2-2b",     "s5",  7),
    ("membership", "gemma-2-2b",     "s6",  7),
    ("persona",    "gemma-2-2b",     "s1",  6),
    ("persona",    "gemma-2-2b",     "s2",  6),
    ("persona",    "gemma-2-2b",     "s3",  7),
    ("persona",    "gemma-2-2b",     "s4",  7),
    ("persona",    "gemma-2-2b",     "s7",  7),
    ("persona",    "gemma-2-2b",     "s5",  7),
    ("persona",    "gemma-2-2b",     "s6",  7),
    ("ifeval",     "gemma-2-9b-it",  "s1",  10),
    ("ifeval",     "gemma-2-9b-it",  "s2",  10),
    ("ifeval",     "gemma-2-9b-it",  "s3",  12),
    ("ifeval",     "gemma-2-9b-it",  "s4",  12),
    ("ifeval",     "gemma-2-9b-it",  "s7",  12),
    ("ifeval",     "gemma-2-9b-it",  "s5",  12),
    ("ifeval",     "gemma-2-9b-it",  "s6",  12),
]


def _import_audit():
    """Import classify_setting/model/dataset helpers from /tmp/_audit_v7_coverage.py.
    Falls back to inlined definitions if /tmp version is missing.
    """
    try:
        sys.path.insert(0, "/tmp")
        from _audit_v7_coverage import (  # type: ignore
            classify_setting, model_from_dirname, dataset_from_dirname,
        )
        return classify_setting, model_from_dirname, dataset_from_dirname
    except Exception:
        # Inline fallback (mirror /tmp/_audit_v7_coverage.py)
        MODELS = ["gemma-2-2b", "gemma-2-2b-it", "gemma-2-9b-it", "gemma-4-31B-it"]
        DS_FULLS = list(DS_LONG.values())

        def _classify(d: str):
            if "--cft--" in d:
                return "s13"
            if "--labelonly0.1--" in d and "--pref0.0--" in d and "--cft--" not in d:
                return "s1"
            has_tc_self = "--tc-self--" in d
            has_tc_neg = "--tc-neg--" in d
            has_fsx = "--force-same-x--" in d
            has_ppd = "--ppd--" in d
            has_nllv = "--nllv1.0--" in d
            has_vlo = "--vallogodds--" in d
            if not has_tc_self and not has_tc_neg:
                if not has_fsx and not has_nllv and not has_vlo:
                    return "s2"
                if has_fsx and has_ppd and has_nllv and has_vlo:
                    return "s3"
                return None
            if has_tc_self:
                if has_fsx and has_ppd and has_nllv and has_vlo:
                    return "s4"
                if has_fsx and has_ppd and not has_nllv and not has_vlo:
                    return "s5"
                if not has_fsx and not has_nllv and not has_vlo:
                    return "s6"
                if not has_fsx and has_nllv and has_vlo:
                    return "s11"
            if has_tc_neg:
                if has_fsx and has_ppd and has_nllv and has_vlo:
                    return "s7"
                if not has_fsx and has_nllv and has_vlo:
                    return "s12"
            return None

        def _model(d: str):
            for m in MODELS:
                if f"v7-google--{m}-delta" in d:
                    return m
            return None

        def _ds(d: str):
            for full in DS_FULLS:
                if f"--{full}-all--" in d:
                    return full, DS_SHORT[full]
            return None, None

        return _classify, _model, _ds


def collect_disk_coverage():
    """Return dict (ds_short, model, setting) -> {epoch_int: dirname}.
    Uses the audit helpers.
    """
    classify_setting, model_from_dirname, dataset_from_dirname = _import_audit()
    out = {}
    if not MODELS_DIR.exists():
        return out
    for d in os.listdir(MODELS_DIR):
        if not d.startswith("v7-google--"):
            continue
        if d.endswith("_merged"):
            continue
        m = model_from_dirname(d)
        dsf_dss = dataset_from_dirname(d)
        if dsf_dss is None:
            continue
        dsf, dss = dsf_dss
        s = classify_setting(d)
        m_ep = re.search(r"-epoch(\d+)--", d)
        ep = int(m_ep.group(1)) if m_ep else -1
        if not (m and dsf and s and ep >= 0):
            continue
        key = (dss, m, s)
        out.setdefault(key, {})[ep] = d
    return out


def collect_score_coverage():
    """Return dict (ds_short, model, setting) -> dict[task] -> set(prefixes_seen).
    Uses parse_checkpoint_name to handle both Format B and Format C names.
    """
    sys.path.insert(0, str(REPO / "src"))
    try:
        from checkpoint_name_parser import parse_checkpoint_name  # type: ignore
    except Exception as e:
        print(f"  WARN: cannot import parse_checkpoint_name: {e}")
        return {}

    out = {}
    if not SCORES_DIR.exists():
        return out
    PREFIXES = ("basetypneg-", "basetyp-", "self-", "neg-")

    def _setting_from_parsed(p):
        if p.get("cft"):
            return "s13"
        tc = p.get("tc")
        nll_v = p.get("nll_v", 0.0) or 0.0
        nll_g = p.get("nll_g", 0.0) or 0.0
        fsx = bool(p.get("force_same_x"))
        ppd = bool(p.get("ppd"))
        vlo = bool(p.get("vallogodds"))
        labelonly = p.get("labelonly")
        pref = p.get("pref", 1.0)
        if labelonly is not None and pref == 0.0 and not p.get("cft"):
            return "s1"
        if tc is None:
            if not fsx and not vlo and nll_v == 0 and nll_g == 0:
                return "s2"
            if fsx and ppd and nll_v == 1.0 and nll_g == 1.0 and vlo:
                return "s3"
            return None
        if tc == "self":
            if fsx and ppd and nll_v == 1.0 and nll_g == 1.0 and vlo:
                return "s4"
            if fsx and ppd and nll_v == 0 and nll_g == 0 and not vlo:
                return "s5"
            if not fsx and nll_v == 0 and nll_g == 0 and not vlo:
                return "s6"
            if not fsx and nll_v == 1.0 and nll_g == 1.0 and vlo:
                return "s11"
        if tc == "neg":
            if fsx and ppd and nll_v == 1.0 and nll_g == 1.0 and vlo:
                return "s7"
            if not fsx and nll_v == 1.0 and nll_g == 1.0 and vlo:
                return "s12"
        return None

    DS_SEG_TO_SHORT = {
        "membership-sans-rosch-v0-all": "membership",
        "persona-v1-all": "persona",
        "ifeval-concat-all": "ifeval",
        "humaneval-v2.1correct-upper-all": "humaneval",
    }

    for f in SCORES_DIR.glob("scores_*_test_log-odds*.csv"):
        name = f.name
        after = name[len("scores_"):]
        chosen = None
        for p in PREFIXES:
            if after.startswith(p):
                chosen = p
                break
        if chosen is None:
            continue
        rest = after[len(chosen):]
        # rest = "<model_short>_<task>_test_log-odds...csv"
        for ds_short, tasks in DATASET_TASKS.items():
            chosen_task = None
            for t in tasks:
                if f"_{t}_test_log-odds" in rest:
                    chosen_task = t
                    break
            if chosen_task is None:
                continue
            model_short = rest.split(f"_{chosen_task}_test_log-odds", 1)[0]
            try:
                parsed = parse_checkpoint_name(model_short)
            except ValueError:
                break
            m = parsed.get("model_short")
            s = _setting_from_parsed(parsed)
            if not (m and s):
                break
            # Verify task_segment matches
            seg_short = DS_SEG_TO_SHORT.get(parsed.get("task_segment", ""))
            if seg_short and seg_short != ds_short:
                break
            key = (ds_short, m, s)
            out.setdefault(key, {}).setdefault(chosen_task, set()).add(chosen)
            break
    return out


JOB_LOG_PATH = REPO / "overnight" / "_overnight_jobids.txt"


def _parse_label(label: str):
    """Parse label like 'persona-gemma-2-9b-it-s3' or 'membership-gemma-2-2b-s13'
    into (dataset, model, setting). Settings are sN. Datasets are persona,
    membership, ifeval, humaneval. Returns (ds_short, model, setting) or None."""
    # Setting is the last token sN
    m_set = re.search(r"-s(\d+)$", label)
    if not m_set:
        return None
    setting = f"s{m_set.group(1)}"
    rest = label[: m_set.start()]
    # Dataset is the first token
    for ds in ("persona", "membership", "ifeval", "humaneval"):
        if rest.startswith(ds + "-"):
            model = rest[len(ds) + 1 :]
            return ds, model, setting
    return None


def _build_jobid_index():
    """Read the overnight JOB_LOG and build {jobid: (ds, model, setting)} map.
    The JOB_LOG has lines like:
        2026-05-25T09:47:11Z  persona-gemma-2-9b-it-s3  TRAIN=42377  TC_EVAL_LIST=self neg
    We use the lines with TRAIN_JOBID + TC_EVAL_LIST (canonical train-only entries)
    to map jobid -> (ds, model, setting).
    """
    idx = {}
    if not JOB_LOG_PATH.is_file():
        return idx
    try:
        for line in JOB_LOG_PATH.read_text(errors="replace").splitlines():
            # Match a line that has "<label> TRAIN=<jobid>" - includes both the
            # train-only line (TC_EVAL_LIST=...) and the per-eval lines (EVAL=).
            # We don't care which, since label is the same.
            mm = re.search(r"\s+(\S+)\s+TRAIN=(\d+)\b", line)
            if not mm:
                continue
            label, jobid = mm.group(1), mm.group(2)
            # Strip eval suffix (-evaltc-{tc}, -evaltc-{tc}-EVALONLY, etc.)
            label = re.sub(r"-evaltc-(self|neg)(-EVALONLY)?$", "", label)
            parsed = _parse_label(label)
            if parsed:
                idx[jobid] = parsed
    except Exception:
        pass
    return idx


def queue_status():
    """Return (n_jobs, list_of_(jobid, name, state, model_or_none, dataset_or_none, setting_or_none))."""
    sq = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", "jdr"), "-h", "-o", "%i %j %t"],
        capture_output=True, text=True,
    ).stdout.splitlines()
    parsed = []
    jobid_idx = _build_jobid_index()
    for line in sq:
        parts = line.split()
        if not parts:
            continue
        jid = parts[0]; jname = parts[1] if len(parts) > 1 else ""; st = parts[2] if len(parts) > 2 else ""
        m = ds = s = None
        if jname == "wrap":
            # PRIMARY: look up in the canonical _overnight_jobids.txt map.
            # This is unambiguous because _overnight_launch.sh writes a
            # `<label> TRAIN=<jobid>` line at submit time. The .out file is
            # NOT enough on its own - many flags (--force-same-x, --cft, etc.)
            # never appear in it.
            if jid in jobid_idx:
                ds, m, s = jobid_idx[jid]
            else:
                # FALLBACK: best-effort .out parsing for jobs submitted before
                # the JOB_LOG existed or by external tools.
                p = LOGS_DIR / f"{jid}.out"
                if p.is_file():
                    try:
                        text = p.read_text(errors="replace")[:3000]
                        mm = re.search(r"Model:\s*google/(\S+)", text)
                        if mm:
                            m = mm.group(1)
                        if "persona-v1" in text: ds = "persona"
                        elif "membership-sans-rosch" in text: ds = "membership"
                        elif "ifeval" in text: ds = "ifeval"
                        elif "humaneval" in text: ds = "humaneval"
                        if "--cft" in text or "--consistency-ft" in text: s = "s13"
                        elif "--labelonly0.1" in text and "--pref0.0" in text: s = "s1"
                        # NOTE: most setting flags don't appear in .out; this
                        # fallback collapses many settings to "s2". Pre-fer
                        # the JOB_LOG path above.
                        else: s = "s2"
                    except Exception:
                        pass
        parsed.append((jid, jname, st, m, ds, s))
    return len(parsed), parsed


def fire_eval(dataset, model, setting, tc, no_base, dep_jobid=None):
    """Submit one _eval_only.sh job. Returns jobid or None on failure."""
    env = os.environ.copy()
    if no_base:
        env["NO_BASE"] = "1"
    if dep_jobid:
        env["DEP_JOBID"] = str(dep_jobid)
    if DRYRUN:
        env["DRYRUN"] = "1"
    cmd = ["bash", str(SCRIPT_DIR / "_eval_only.sh"), dataset, model, setting, tc]
    print(f"  fire_eval: {dataset} {model} {setting} {tc} no_base={no_base} dep={dep_jobid}")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"    FAIL: {proc.stderr[-500:]}")
        return None
    m = re.search(r"Submitted batch job (\d+)", proc.stdout)
    if m:
        return m.group(1)
    return None


def fire_train(dataset, model, setting, walltime):
    env = os.environ.copy()
    env["WALLTIME"] = str(walltime)
    if DRYRUN:
        env["DRYRUN"] = "1"
    cmd = ["bash", str(SCRIPT_DIR / "_overnight_launch.sh"), dataset, model, setting]
    print(f"  fire_train: {dataset} {model} {setting} WALLTIME={walltime}")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"    FAIL: {proc.stderr[-500:]}")
        return None
    m = re.search(r"Train submitted: jobid=(\d+)", proc.stdout)
    return m.group(1) if m else None


def main():
    print(f"=== overnight master @ {datetime.now(timezone.utc).strftime('%FT%TZ')} ===")
    n_q, q = queue_status()
    slots = QUEUE_CAP - n_q
    print(f"queue: {n_q} jobs ({QUEUE_CAP} cap, {slots} slots free)")

    if slots <= 0:
        print("queue at cap; no submissions")
        return

    disk = collect_disk_coverage()
    scores = collect_score_coverage()
    queued_train_cells = {(ds, m, s) for (_, _, _, m, ds, s) in q if m and ds and s}
    print(f"disk-trained cells: {len(disk)}; score cells: {len(scores)}; "
          f"in-queue train cells: {len(queued_train_cells)}")

    # PASS 1: eval-gap fills for cells with epoch>=1 on disk.
    # Cap at MAX_P1 evals per tick — many "missing prefix" detections are
    # false positives (hash-suffixed CSVs that parse_checkpoint_name fails on),
    # so spamming all of them wastes queue slots that should go to Pass 2 trains.
    submitted_evals = 0
    MAX_P1 = int(os.environ.get("MAX_P1", "6"))
    for (ds_short, m, s), eps in disk.items():
        if submitted_evals >= MAX_P1:
            print(f" [P1] reached MAX_P1={MAX_P1} cap; skipping rest")
            break
        # Need at least epoch >= 1 for eval to have something to load
        if max(eps) < 1:
            continue
        needs = SETTING_NEEDS.get(s, set())
        if not needs:
            continue
        seen_prefixes = set()
        for t, pfxs in scores.get((ds_short, m, s), {}).items():
            seen_prefixes.update(pfxs)
        # Also count: how many tasks have full coverage for each prefix.
        # (Pass 1 fires the missing PREFIX classes only.)
        missing = needs - seen_prefixes
        if not missing:
            continue
        for prefix in missing:
            # Prefix -> (TC, no_base)
            if prefix == "basetyp-":     tc, nb = "self", False
            elif prefix == "self-":      tc, nb = "self", True
            elif prefix == "basetypneg-": tc, nb = "neg",  False
            elif prefix == "neg-":       tc, nb = "neg",  True
            else: continue
            # Skip if this same eval is already in queue (avoid double-fire).
            already = False
            for jid, jname, st, mm, dds, ss in q:
                if jname.startswith("eval-") and dds == ds_short and ss == s:
                    # crude — eval job names don't carry model/no_base; accept some duplication
                    pass
            print(f" [P1] {ds_short} × {m} × {s}: missing prefix '{prefix}' → eval {tc} no_base={nb}")
            if slots <= 0:
                print(" [P1] queue full; stopping pass 1")
                break
            jid = fire_eval(ds_short, m, s, tc, nb, dep_jobid=None)
            if jid:
                slots -= 1
                submitted_evals += 1
        if slots <= 0:
            break

    # PASS 2: gap-fill new trains.
    submitted_trains = 0
    for (ds_short, m, s, walltime) in GAP_LIST:
        if slots < 3:
            print(" [P2] not enough slots for train+2evals; stopping")
            break
        key = (ds_short, m, s)
        if key in disk and max(disk[key].keys()) >= 1:
            continue  # already trained
        if key in queued_train_cells:
            continue  # already in queue
        print(f" [P2] launching {ds_short} × {m} × {s} (walltime={walltime}h)")
        train_jid = fire_train(ds_short, m, s, walltime)
        if train_jid:
            slots -= 3  # train + 2 chained evals
            submitted_trains += 1
            queued_train_cells.add(key)
            # Also schedule NO_BASE evals for non-TC settings (s1/s2/s3/s13)
            # OR for TC-trained settings whose self-/neg- prefix matches.
            needs = SETTING_NEEDS.get(s, set())
            no_base_prefixes = needs & {"self-", "neg-"}
            for prefix in no_base_prefixes:
                tc = "self" if prefix == "self-" else "neg"
                if slots <= 0:
                    break
                jid = fire_eval(ds_short, m, s, tc, no_base=True, dep_jobid=train_jid)
                if jid:
                    slots -= 1

    print(f"=== done: P1 evals={submitted_evals}, P2 trains={submitted_trains}, slots_left={slots} ===")


if __name__ == "__main__":
    main()
