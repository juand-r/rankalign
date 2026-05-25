#!/usr/bin/env python3
"""Submit eval-gap-fill jobs for trained-but-not-evaluated v7 cells.

For each target (ds, model, setting):
  - Check disk to confirm model dir exists.
  - Check existing CSVs (BOTH /datastor1 and /datastor2) for each prefix.
  - Check the queue for an already-running/queued eval with the same job name.
  - Determine the MINIMUM set of `_eval_only.sh` invocations needed.
  - In DRYRUN mode: print what would be submitted.
  - In SUBMIT mode: invoke `_eval_only.sh` for each, log jobids.
"""
import os, re, subprocess, sys
from pathlib import Path

sys.path.insert(0, "/tmp")
from _audit_v7_coverage import (
    classify_setting, model_from_dirname, dataset_from_dirname,
)

REPO = Path("/datastor1/jdr/gv-gap/rankalign")
MODELS_DIR = Path("/datastor2/jdr/rankalign/models2")
SCORE_DIRS = [
    Path("/datastor1/jdr/gv-gap/rankalign/outputs"),
    Path("/datastor2/jdr/rankalign/outputs"),
]
EVAL_ONLY = REPO / "scripts" / "_eval_only.sh"

DRYRUN = ("--submit" not in sys.argv)

# What each setting NEEDS, keyed by setting tag.
# Each entry: list of (TC, NO_BASE) pairs that we should call the eval script with,
# IF the corresponding prefix CSVs are missing. The script with NO_BASE=0 emits
# basetyp- AND self- (when TC=self), so it covers TWO prefixes.
SETTING_NEEDS = {
    # non-TC trained: emit all four prefixes (PMIb, PMIs, Negb, Negs)
    "s1":  ["PMIb,PMIs", "Negb,Negs"],
    "s2":  ["PMIb,PMIs", "Negb,Negs"],
    "s3":  ["PMIb,PMIs", "Negb,Negs"],
    "s13": ["PMIb,PMIs", "Negb,Negs"],
    # TC-self trained: emit PMIb + PMIs only
    "s4":  ["PMIb,PMIs"],
    "s5":  ["PMIb,PMIs"],
    "s6":  ["PMIb,PMIs"],
    "s11": ["PMIb,PMIs"],
    # TC-neg trained: emit Negb + Negs only
    "s7":  ["Negb,Negs"],
    "s12": ["Negb,Negs"],
}
# Map "PMIb,PMIs" call-signature -> (TC, NO_BASE) for _eval_only.sh
CALL_TO_FLAGS = {
    "PMIb,PMIs": ("self", "0"),  # writes both basetyp- and self-
    "PMIb":      ("self", "0"),  # ditto, but we asked for one
    "PMIs":      ("self", "1"),  # writes self- only
    "Negb,Negs": ("neg",  "0"),
    "Negb":      ("neg",  "0"),
    "Negs":      ("neg",  "1"),
}
# Prefix display -> filename prefix
PREFIX_FNAME = {
    "PMIb": "basetyp-",
    "PMIs": "self-",
    "Negb": "basetypneg-",
    "Negs": "neg-",
}

def model_to_tag(m):
    s = m.lower()
    if "gemma-2-2b-it" in s: return "2b-it"
    if "gemma-2-9b-it" in s: return "9b-it"
    if "gemma-2-2b" in s: return "2b"
    return s

# 1. Find the model dir on disk for a given (ds, model, setting). Returns latest epoch dir or None.
def find_model_dir(ds_short, model, setting):
    ds_full = {
        "membership": "membership-sans-rosch-v0",
        "persona":    "persona-v1",
        "ifeval":     "ifeval-concat",
        "humaneval":  "humaneval-v2.1correct-upper",
    }[ds_short]
    matches = []
    for d in os.listdir(MODELS_DIR):
        if d.endswith("_merged"):
            continue
        if not d.startswith("v7-google--"):
            continue
        m = model_from_dirname(d)
        dsf, dss = dataset_from_dirname(d)
        s = classify_setting(d)
        ep_match = re.search(r"-epoch(\d+)--", d)
        ep = int(ep_match.group(1)) if ep_match else -1
        if m == model and dss == ds_short and s == setting and ep >= 1:
            matches.append((ep, d))
    if not matches:
        return None
    matches.sort(reverse=True)  # highest epoch first
    return matches[0][1]

# 2. Check if a given prefix CSV exists for a (ds, model, setting). Match against
#    BOTH score dirs and various filename styles.
def has_prefix_csv(ds_short, model, setting, prefix_short):
    """prefix_short ∈ {PMIb, PMIs, Negb, Negs}; matches the file's leading 'basetyp-/self-/...'"""
    pname = PREFIX_FNAME[prefix_short]
    # Filename starts with: scores_{pname}v7-google--{model}-delta...{setting-sig...}_{task}_test_log-odds_tc_{date}.csv
    # We need a CSV that:
    #   * has the right prefix
    #   * mentions the right model + ds_full
    #   * matches the setting signature (we verify via classify_setting)
    ds_full = {
        "membership": "membership-sans-rosch-v0",
        "persona":    "persona-v1",
        "ifeval":     "ifeval-concat",
        "humaneval":  "humaneval-v2.1correct-upper",
    }[ds_short]
    pattern_prefix = f"scores_{pname}v7-google--{model}-delta"
    for sd in SCORE_DIRS:
        if not sd.exists():
            continue
        try:
            for f in os.listdir(sd):
                if not f.startswith(pattern_prefix):
                    continue
                if ds_full not in f:
                    continue
                # Verify setting via classify
                sig_part = f[len(f"scores_{pname}"):]
                s = classify_setting(sig_part)
                if s == setting:
                    return True
        except OSError:
            continue
    return False

# 3. Check if a job with the target name is already in the queue.
def queued_jnames():
    try:
        out = subprocess.run(
            ["squeue", "-u", "jdr", "-h", "-o", "%j"],
            capture_output=True, text=True, check=True,
        ).stdout
        return set(out.splitlines())
    except subprocess.CalledProcessError:
        return set()

# 4. Decide what to submit for a (ds, model, setting). Returns list of (TC, NO_BASE, prefixes_covered).
def plan_for_cell(ds_short, model, setting, qjnames):
    needs = SETTING_NEEDS.get(setting, [])
    plan = []
    tag = model_to_tag(model)
    for need_str in needs:
        prefixes = need_str.split(",")
        # Only schedule if AT LEAST ONE prefix is missing
        all_have = all(has_prefix_csv(ds_short, model, setting, p) for p in prefixes)
        if all_have:
            continue
        # Pick the most efficient call. If we need both X+Y and the Y-only call would
        # be wasteful, prefer the combined NO_BASE=0 call.
        # Find: are EITHER of the two prefixes missing? If only one, prefer the call
        # that emits exactly that prefix (NO_BASE=1 emits only the "self-/neg-" prefix;
        # NO_BASE=0 emits both basetyp + self).
        miss_set = [p for p in prefixes if not has_prefix_csv(ds_short, model, setting, p)]
        if len(miss_set) == 2:
            tc, nb = CALL_TO_FLAGS[need_str]
            covered = need_str
        else:
            only = miss_set[0]
            tc, nb = CALL_TO_FLAGS[only]
            covered = only
        # Build the jname that _eval_only.sh would use
        nb_tag = "-nobase" if nb == "1" else ""
        jname = f"eval-{setting}-{ds_short}-{tag}-{tc}{nb_tag}-only"
        if jname in qjnames:
            continue
        plan.append((tc, nb, covered, jname))
    return plan

# ----- TARGETS -----
TARGETS = []
# 9b-it × membership: every setting where something's missing
for s in ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s11", "s12", "s13"]:
    TARGETS.append(("membership", "gemma-2-9b-it", s))
# 2b-it × membership
for s in ["s2", "s3", "s4", "s6", "s7", "s11", "s12", "s13"]:
    TARGETS.append(("membership", "gemma-2-2b-it", s))
# 9b-it × persona
for s in ["s3", "s4", "s7"]:
    TARGETS.append(("persona", "gemma-2-9b-it", s))
# 2b-it × persona
for s in ["s3", "s4", "s7"]:
    TARGETS.append(("persona", "gemma-2-2b-it", s))
# 2b × persona × s11 (PMIs only)
TARGETS.append(("persona", "gemma-2-2b", "s11"))
# NOTE: ifeval × s13 gap-fills are SKIPPED. Investigation showed jobs
# 42603-42607 (manual ifeval-s13 evals) "completed" in 4-5 minutes
# without writing any CSVs. There are 4851 v6 ifeval CSVs on disk but
# zero v7 ones — the eval pipeline appears to silently no-op for
# ifeval-trained adapters. Re-enable these once the bug is fixed.
# TARGETS.append(("ifeval", "gemma-2-2b-it", "s13"))
# TARGETS.append(("ifeval", "gemma-2-2b", "s13"))

# Plan & report.
qjnames = queued_jnames()
print(f"Found {len(qjnames)} jobs in queue.\n")
total_plan = []
for ds, m, s in TARGETS:
    md = find_model_dir(ds, m, s)
    if md is None:
        print(f"  SKIP (no model dir): {ds} × {m} × {s}")
        continue
    plan = plan_for_cell(ds, m, s, qjnames)
    if not plan:
        print(f"  ALREADY DONE: {ds} × {m} × {s}")
        continue
    for tc, nb, covered, jname in plan:
        total_plan.append((ds, m, s, tc, nb, covered, jname))
        print(f"  PLAN: {ds:<11} {m:<15} {s:<4}  TC={tc}  NO_BASE={nb}  covers={covered:<10}  jname={jname}")

print(f"\nTotal: {len(total_plan)} jobs to submit.")

if DRYRUN:
    print("\nDRY RUN. Pass --submit to actually submit.")
    sys.exit(0)

print("\n=== SUBMITTING ===")
LOGF = REPO / "docs" / "deadline_logs" / f"gap_fills_$(date +%H%M).txt".replace("$(date +%H%M)", subprocess.run(["date","-u","+%H%M"], capture_output=True, text=True).stdout.strip())
LOGF.parent.mkdir(parents=True, exist_ok=True)
with open(LOGF, "w") as logf:
    logf.write(f"# eval gap-fill submissions @ {subprocess.run(['date','-u'],capture_output=True,text=True).stdout.strip()}\n")
    submitted = 0
    failed = 0
    for ds, m, s, tc, nb, covered, jname in total_plan:
        env = os.environ.copy()
        if nb == "1":
            env["NO_BASE"] = "1"
        else:
            env.pop("NO_BASE", None)
        proc = subprocess.run(
            ["bash", str(EVAL_ONLY), ds, m, s, tc],
            env=env, capture_output=True, text=True,
        )
        line = f"{ds} {m} {s} TC={tc} NO_BASE={nb} jname={jname}"
        if proc.returncode != 0:
            failed += 1
            print(f"  FAIL: {line}  rc={proc.returncode}  stdout-tail={proc.stdout.strip()[-200:]}  stderr-tail={proc.stderr.strip()[-200:]}")
            logf.write(f"FAIL  {line}\n")
            continue
        m_jid = re.search(r"Submitted batch job (\d+)", proc.stdout)
        if m_jid:
            jid = m_jid.group(1)
            submitted += 1
            print(f"  OK:   {line}  jid={jid}")
            logf.write(f"OK    {line}  jid={jid}\n")
        else:
            failed += 1
            print(f"  NO_JID: {line}  stdout-tail={proc.stdout.strip()[-200:]}")
            logf.write(f"NOJID {line}\n")
    print(f"\nSubmitted {submitted}/{len(total_plan)} ({failed} failed). Log: {LOGF}")
