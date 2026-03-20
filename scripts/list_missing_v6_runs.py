#!/usr/bin/env python3
"""
List v6 settings that have not been run yet:
- Train: model directory does not exist in models/
- Eval: no scores_* file in outputs/ for that model+task (test split, typicality/evaltc).

Uses the same "desired" grid as run_eval_hypernym_tasks.sh: 8 tasks, 2 configs (vanilla pref0.0 vallogodds, tc-online pref0.0 vallogodds), epoch2.
Eval is considered done if any file matching scores_<model_short>_<task>_test_v2_log-odds_evaltc*.csv exists.
"""
import os
import re
import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
OUTPUTS_DIR = os.path.join(REPO_ROOT, "outputs")

# From run_eval_hypernym_tasks.sh
EVAL_TASKS = [
    "hypernym-bananas",
    "hypernym-bazookas",
    "hypernym-cabinets",
    "hypernym-cars",
    "hypernym-chairs",
    "hypernym-crows",
    "hypernym-diapers",
    "hypernym-dogs",
]

EPOCH = 2

# Two model configs evaluated in run_eval_hypernym_tasks.sh (path suffixes)
CONFIG_VANILLA = "-all--d2g--random--alpha1.0--full-completion--pref0.0--nllv1.0--nllg1.0--vallogodds"
CONFIG_TC_ONLINE = "-all--d2g--random--alpha1.0--tc-online--full-completion--pref0.0--nllv1.0--nllg1.0--vallogodds"


def model_dir_to_short(model_dir_name):
    """Convert model dir name to model_short (-- -> _) as used in scores filenames."""
    return model_dir_name.replace("--", "_")


def task_from_model_dir(model_dir_name):
    """Extract task from v6 model dir, e.g. hypernym-bananas from ...--hypernym-bananas-all--..."""
    m = re.match(r"v6-google--gemma-2-2b-delta[\d.]+-epoch\d+--(hypernym-[a-z-]+)-all--", model_dir_name)
    return m.group(1) if m else None


def main():
    # ---- Desired train (same as run_eval): 8 tasks × 2 configs, epoch2
    base = f"v6-google--gemma-2-2b-delta0.15-epoch{EPOCH}--"
    desired_train = []
    for task in EVAL_TASKS:
        for suffix in (CONFIG_VANILLA, CONFIG_TC_ONLINE):
            desired_train.append(base + task + suffix)

    existing_models = set()
    if os.path.isdir(MODELS_DIR):
        for name in os.listdir(MODELS_DIR):
            if name.startswith("v6-") and os.path.isdir(os.path.join(MODELS_DIR, name)):
                existing_models.add(name)

    missing_train = [d for d in desired_train if d not in existing_models]

    # ---- Eval: for each v6 model dir in models/, check for scores_*_<task>_test_v2_log-odds_evaltc*.csv
    # Build set of (model_short, task) that have at least one scores file
    eval_done = set()
    if os.path.isdir(OUTPUTS_DIR):
        for f in os.listdir(OUTPUTS_DIR):
            if not f.startswith("scores_v6-") or not f.endswith(".csv"):
                continue
            if "_test_v2_log-odds_evaltc" not in f:
                continue
            # Parse: scores_<model_short>_<task>_test_v2_log-odds_evaltc_*.csv
            rest = f[len("scores_"):]
            parts = rest.split("_test_v2_log-odds_evaltc")
            if len(parts) != 2:
                continue
            left = parts[0]
            # task is the last hypernym-<noun> segment; model_short is everything before it
            # e.g. v6-google_gemma-2-2b-delta0.15-epoch2_hypernym-bananas-all_d2g_..._hypernym-bananas
            m = re.search(r'_(hypernym-[a-zA-Z]+)$', left)
            if m:
                task = m.group(1)
                model_short = left[:m.start()]
                eval_done.add((model_short, task))

    # All v6 epoch2 model dirs that exist
    v6_epoch2_models = [m for m in existing_models if m.startswith("v6-") and f"-epoch{EPOCH}--" in m]
    missing_eval = []
    for model_dir in sorted(v6_epoch2_models):
        model_short = model_dir_to_short(model_dir)
        task = task_from_model_dir(model_dir)
        if task is None:
            continue
        if (model_short, task) not in eval_done:
            missing_eval.append((model_dir, task))

    # ---- Print
    print("=== V6 settings not yet run ===\n")
    print("--- Missing TRAIN (desired model dir not in models/) ---")
    print("(Desired: 8 tasks × 2 configs epoch2, same as run_eval_hypernym_tasks.sh)\n")
    if not missing_train:
        print("None. All 16 desired train configs have model dirs.")
    else:
        for d in sorted(missing_train):
            print(d)
        print(f"\nTotal: {len(missing_train)}")

    print("\n--- Missing EVAL (model exists but no scores_*_<task>_test_v2_log-odds_evaltc*.csv in outputs/) ---")
    if not missing_eval:
        print("None.")
    else:
        for model_dir, task in missing_eval:
            print(f"  {model_dir}")
            print(f"    -> task: {task}")
        print(f"\nTotal: {len(missing_eval)}")

    return missing_train, missing_eval


if __name__ == "__main__":
    main()
