#!/usr/bin/env python3
"""Per-(model, task) status table with: trained?, eval status, ETA for missing.

Reads BOTH /datastor1/.../outputs and /datastor2/.../outputs for score CSVs
(since today's runs split between them). Reads /datastor2/.../models2 for
trained adapters. Reads `squeue` for ETA on in-flight evals.
"""
import os, re, subprocess, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/tmp")
from _audit_v7_coverage import (
    classify_setting, model_from_dirname, dataset_from_dirname, DATASETS,
)

MODELS_DIR = Path("/datastor2/jdr/rankalign/models2")
SCORE_DIRS = [
    Path("/datastor1/jdr/gv-gap/rankalign/outputs"),
    Path("/datastor2/jdr/rankalign/outputs"),
]

# What prefixes the table NEEDS for each setting
# (TC-self trained: basetyp- + self-; TC-neg: basetypneg- + neg-; non-TC: all 4)
SETTING_NEEDS = {
    "s1":  ["basetyp-", "self-", "basetypneg-", "neg-"],
    "s2":  ["basetyp-", "self-", "basetypneg-", "neg-"],
    "s3":  ["basetyp-", "self-", "basetypneg-", "neg-"],
    "s4":  ["basetyp-", "self-"],
    "s5":  ["basetyp-", "self-"],
    "s6":  ["basetyp-", "self-"],
    "s11": ["basetyp-", "self-"],
    "s7":  ["basetypneg-", "neg-"],
    "s12": ["basetypneg-", "neg-"],
    "s13": ["basetyp-", "self-", "basetypneg-", "neg-"],
}
SETTING_ORDER = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s11", "s12", "s13"]

PREFIX_DISPLAY = {
    "basetyp-":    "PMIb",   # PMI base
    "self-":       "PMIs",   # PMI self
    "basetypneg-": "Negb",   # Neg base
    "neg-":        "Negs",   # Neg self
}

# 1) Disk: which (model, ds, setting) cells have an adapter saved (epoch>=1)
disk = defaultdict(set)
for d in os.listdir(MODELS_DIR):
    if not d.startswith("v7-google--") or d.endswith("_merged"):
        continue
    m = model_from_dirname(d)
    dsf, dss = dataset_from_dirname(d)
    s = classify_setting(d)
    me = re.search(r"-epoch(\d+)--", d)
    ep = int(me.group(1)) if me else -1
    if not (m and dsf and s and ep >= 0):
        continue
    if ep >= 1:
        disk[(dss, m, s)].add(ep)

# 2) Score CSVs: which prefix-columns have been computed
# Filename pattern (both old and new):
#   scores_{prefix}v7-google--{model}-delta...--{ds_full}-all--...{setting-sig}_{task}_test_log-odds_tc_{date}.csv
def parse_score(p: Path):
    name = p.name
    if not name.startswith("scores_"):
        return None
    rest = name[len("scores_"):]
    # find the v7-google-- anchor
    idx = rest.find("v7-google--")
    if idx < 0:
        return None
    prefix = rest[:idx]
    if prefix and not prefix.endswith("-"):
        prefix = prefix + "-"  # normalize
    if prefix not in ("", "basetyp-", "self-", "basetypneg-", "neg-"):
        return None
    if prefix == "":
        # untyped (legacy "no prefix"); skip
        return None
    body = rest[idx:]
    m = model_from_dirname(body)
    dsf, dss = dataset_from_dirname(body)
    s = classify_setting(body)
    if not (m and dsf and s):
        return None
    return (dss, m, s, prefix)

scores = defaultdict(set)
for sd in SCORE_DIRS:
    if not sd.exists():
        continue
    for p in sd.glob("scores_*.csv"):
        rec = parse_score(p)
        if rec:
            ds, m, s, prefix = rec
            scores[(ds, m, s)].add(prefix)

# 3) Queue: ETA for in-flight evals (decoded from job names)
# Job names (from _eval_only.sh): eval-{s}-{ds}-{tag}-{tc}{-nobase?}-only
# Or (from _overnight_launch.sh chained): eval-{s}-{ds}-{tc}  (no model tag for some)
# Plus train jobs: parsed from /datastor1/jdr/gv-gap/rankalign/overnight/_overnight_jobids.txt
def model_to_tag(model_full):
    m = model_full.lower()
    if "gemma-2-2b-it" in m: return "2b-it"
    if "gemma-2-9b-it" in m: return "9b-it"
    if "gemma-2-2b"   in m: return "2b"
    if "gemma-2-9b"   in m: return "9b"
    if "gemma-4-31b-it" in m: return "g431Bit"
    return m

# parse squeue
sq = subprocess.run(
    ["squeue", "-u", "jdr", "-h", "-o", "%i|%t|%j|%L"],
    capture_output=True, text=True,
).stdout.strip().splitlines()

# Map (ds, model_tag, setting, prefix) -> "ETA" str ; also track trains
in_flight_evals = {}  # (ds, m_tag, s, prefix) -> remaining time string
in_flight_trains = {}  # (ds, m_tag, s) -> remaining time string

# Read jobid -> train identity map AND eval-jobid -> (ds, tag, setting, tc) map
JIDS = Path("/datastor1/jdr/gv-gap/rankalign/overnight/_overnight_jobids.txt")
train_lookup = {}  # train_jid -> (ds, m_tag, setting)
eval_lookup = {}   # eval_jid -> (ds, m_tag, setting, tc)
if JIDS.exists():
    for line in JIDS.read_text().splitlines():
        # Lines come in two shapes:
        #   ...  membership-gemma-2-2b-s5  TRAIN=42560  TC_EVAL_LIST=self
        #   ...  ifeval-gemma-2-2b-s3-evaltc-self  TRAIN=42608  EVAL=42609  TC=self  PATH=...
        m_label = re.search(
            r"\s(membership|persona|ifeval|humaneval)-"
            r"(gemma-2-2b-it|gemma-2-2b|gemma-2-9b-it|gemma-2-9b|gemma-4-31B-it)-"
            r"(s\d+)(-evaltc-(self|neg))?(-EVALONLY)?\s",
            line,
        )
        if not m_label:
            continue
        ds, model_full, setting = m_label.group(1), m_label.group(2), m_label.group(3)
        tc_in_label = m_label.group(5)  # 'self' / 'neg' / None
        tag = model_to_tag(model_full)
        train_jid_match = re.search(r"\bTRAIN=(\d+)", line)
        eval_jid_match  = re.search(r"\bEVAL=(\d+)",  line)
        tc_field_match  = re.search(r"\bTC=(self|neg)\b", line)
        if train_jid_match and not eval_jid_match:
            train_lookup[train_jid_match.group(1)] = (ds, tag, setting)
        if eval_jid_match:
            tc = (tc_field_match.group(1) if tc_field_match else tc_in_label)
            if tc:
                eval_lookup[eval_jid_match.group(1)] = (ds, tag, setting, tc)

# Walk squeue
for line in sq:
    if "|" not in line: continue
    jid, st, jname, rem = line.split("|", 3)
    def fmt_eta(state, remaining):
        # 'R' has true remaining time; 'PD' has only allocation, not real ETA.
        if state == "R":
            # Trim trailing seconds if present (X:YY:ZZ -> X:YY)
            return f"running, ≤{remaining}"
        return "queued"

    if jname.startswith("eval-"):
        # First, try strict regex (eval-only jobs from _eval_only.sh have model tag in name).
        # Then, fall back to eval_lookup (truth source from _overnight_jobids.txt).
        ident = eval_lookup.get(jid)
        m1 = re.match(r"eval-(s\d+)-(membership|persona|ifeval|humaneval)-(2b-it|2b|9b-it|9b|g431Bit)-(self|neg)(-nobase)?-only$", jname)
        if m1:
            s, ds, tag, tc, nb = m1.groups()
            prefix = ("self-" if tc=="self" else "neg-") if nb else ("basetyp-" if tc=="self" else "basetypneg-")
            in_flight_evals[(ds, tag, s, prefix)] = fmt_eta(st, rem)
        elif ident:
            ds, tag, s, tc = ident
            eta = fmt_eta(st, rem)
            if tc == "self":
                in_flight_evals.setdefault((ds, tag, s, "basetyp-"), eta)
                in_flight_evals.setdefault((ds, tag, s, "self-"), eta)
            else:
                in_flight_evals.setdefault((ds, tag, s, "basetypneg-"), eta)
                in_flight_evals.setdefault((ds, tag, s, "neg-"), eta)
    else:
        ident = train_lookup.get(jid)
        if ident:
            in_flight_trains[ident] = fmt_eta(st, rem)

# 4) Render the table per (model, ds)
DATASETS_RENDER = [
    ("membership", "membership/rosch"),
    ("persona",    "persona-v1"),
    ("ifeval",     "ifeval-concat"),
    ("humaneval",  "humaneval-v2.1correct-upper"),
]
MODEL_RENDER = [
    ("gemma-2-9b-it", "9b-it"),
    ("gemma-2-2b-it", "2b-it"),
    ("gemma-2-2b",    "2b"),
    ("gemma-4-31B-it","g431Bit"),
]

for model_full, mtag in MODEL_RENDER:
    for ds, ds_label in DATASETS_RENDER:
        # Skip humaneval for non-gemma-4 models, since only g4 trains it
        if ds == "humaneval" and model_full != "gemma-4-31B-it":
            continue
        if ds != "humaneval" and model_full == "gemma-4-31B-it":
            continue
        print(f"\n## Model: {model_full} | Task: {ds_label}")
        print(f"  {'setting':<7}  {'Trained':<8}  {'Evals done':<28}  {'Missing / ETA':<40}")
        print(f"  {'-'*7:<7}  {'-'*8:<8}  {'-'*28:<28}  {'-'*40:<40}")
        for s in SETTING_ORDER:
            key = (ds, model_full, s)
            eps = disk.get(key, set())
            trained_str = ("ep" + ",".join(str(e) for e in sorted(eps))) if eps else "—"
            seen_prefixes = scores.get(key, set())
            needs = SETTING_NEEDS[s]
            done_disp = []
            for p in needs:
                if p in seen_prefixes:
                    done_disp.append(PREFIX_DISPLAY[p])
            done_str = ", ".join(done_disp) if done_disp else "—"
            # Missing
            missing = [p for p in needs if p not in seen_prefixes]
            miss_disp = []
            grouped = {}  # eta_string -> [prefix_short, ...]
            for p in missing:
                key2 = (ds, mtag, s, p)
                if key2 in in_flight_evals:
                    eta = in_flight_evals[key2]
                else:
                    train_key = (ds, mtag, s)
                    if train_key in in_flight_trains:
                        eta = f"train {in_flight_trains[train_key]}"
                    elif eps:
                        eta = "needs eval (not submitted)"
                    else:
                        eta = "no train"
                grouped.setdefault(eta, []).append(PREFIX_DISPLAY[p])
            miss_disp = [f"{','.join(prefs)}: {eta}" for eta, prefs in grouped.items()]
            miss_str = "; ".join(miss_disp) if miss_disp else "DONE"
            print(f"  {s:<7}  {trained_str:<8}  {done_str:<28}  {miss_str:<40}")
