#!/usr/bin/env python3
"""
Check which result score files exist for given training/eval configurations.

Filename structure (from eval_by_claude.py with --self-typicality):
  scores_{self_pfx}{model_short}_{task}_{split}{v2}{metric}{eval_tc}{eval_lenorm}_{timestamp}.csv

Where:
  self_pfx:    "self-" if --self-typicality, "" otherwise
  model_short: model dir name with '--' replaced by '_'
               e.g. v6-google_gemma-2-2b-delta0.15-epoch2_plausibleqa-all_d2g_random_alpha1.0_...
  eval_tc:     "_tc" (eval_by_claude.py) or "_evaltc" (eval.py / older eval_by_claude.py)

Model directory naming (from ranking_loss_ref.py save_directory):
  v6-{model}-delta{d}-epoch{e}--{task}{all}--{direction}--{split_type}--alpha{a}{tc_train}{lenorm}
     {fullcomp}{pref}{nllv}{nllg}{force_same_x}{valboost}{vallogodds}{semi}

  After .replace('--', '_'):
  v6-{model}-delta{d}-epoch{e}_{task}_{direction}_{split_type}_alpha{a}_{tc_train}_{rest}

generate_semi_tables.py uses glob "*tc_*.csv" which matches both _evaltc_ and _tc_.
"""

import glob
import os
import sys
from collections import defaultdict

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')

# ── Configuration dictionaries ──

MODELS = {
    'gemma-2-2b': 'google_gemma-2-2b',
}

TRAINING_TASKS = {
    'plausibleqa': 'plausibleqa-all',
    'ambigqa': 'ambigqa-all',
    'hypernym': 'hypernym-concat-bananas-to-dogs-double-all',
    'ifeval': 'ifeval-concat-all',
}

TRAIN_TC_OPTIONS = {
    'plain': '',
    'tc-self': 'tc-self_',
    'tc-self+len': 'tc-self_lenorm_',
}

LOSS_VARIANTS = {
    'pref':   'full-completion_force-same-x',
    'pref+v': 'full-completion_force-same-x_vallogodds',
    'comb+v': 'full-completion_nllv1.0_nllg1.0_force-same-x_vallogodds',
    'comb':   'full-completion_nllv1.0_nllg1.0_force-same-x',
    'sft':    'full-completion_pref0.0_nllv1.0_nllg1.0_force-same-x',
}

SEMI_OPTIONS = {
    'labelonly0.1': 'labelonly0.1',
    'semi0.1': 'semi0.1',
}


def build_glob_pattern(
    model='gemma-2-2b',
    train_task='plausibleqa',
    train_tc='plain',
    loss='comb+v',
    semi='semi0.1',
    eval_task=None,
    delta='0.15',
    epoch='*',
):
    """Build a glob pattern to match score files for a given configuration.

    Uses *tc_* to match both _evaltc_ and _tc_ suffixes (like generate_semi_tables.py).
    Uses scores_self-* prefix to match eval_by_claude.py with --self-typicality.
    """
    model_str = MODELS[model]
    train_task_str = TRAINING_TASKS[train_task]
    tc_str = TRAIN_TC_OPTIONS[train_tc]
    loss_str = LOSS_VARIANTS[loss]
    semi_str = SEMI_OPTIONS[semi]

    model_fragment = (
        f"self-v6-{model_str}-delta{delta}-epoch{epoch}_"
        f"{train_task_str}_d2g_random_alpha1.0_"
        f"{tc_str}{loss_str}_{semi_str}"
    )

    if eval_task is None:
        eval_task_glob = '*'
    else:
        eval_task_glob = eval_task

    if eval_task and 'hypernym' in eval_task:
        test_suffix = '_test_v2_log-odds'
    elif train_task == 'hypernym' and eval_task is None:
        test_suffix = '_test_v2_log-odds'
    else:
        test_suffix = '_test_log-odds'

    pattern = f"scores_{model_fragment}_{eval_task_glob}{test_suffix}_*tc_*.csv"
    return pattern


def build_base_model_pattern(model='gemma-2-2b', eval_task_family='plausibleqa', eval_task=None):
    """Build pattern for base (non-finetuned) model scores."""
    model_str = MODELS[model]
    if eval_task is None:
        eval_task_glob = '*'
    else:
        eval_task_glob = eval_task

    if eval_task_family == 'hypernym':
        test_suffix = '_test_v2_log-odds'
    else:
        test_suffix = '_test_log-odds'

    return f"scores_self-v6-{model_str}_{eval_task_glob}{test_suffix}_*tc_*.csv"


def find_files(pattern):
    full_pattern = os.path.join(OUTPUTS_DIR, pattern)
    return sorted(glob.glob(full_pattern))


def check_config(**kwargs):
    pattern = build_glob_pattern(**kwargs)
    files = find_files(pattern)
    return pattern, files


def summarize(label, pattern, files, verbose=False):
    status = f"{len(files):4d} files" if files else "  MISSING"
    print(f"{label:55s} {status}")
    if verbose and files:
        for f in files[:3]:
            print(f"    {os.path.basename(f)}")
        if len(files) > 3:
            print(f"    ... +{len(files)-3} more")
    return len(files)


def check_task(train_task, verbose=False):
    """Check all semi-supervised configurations for one training task."""
    print(f"\n{'='*80}")
    print(f"Training task: {train_task.upper()}")
    print(f"{'='*80}")

    # Base model
    pat = build_base_model_pattern(eval_task_family=train_task)
    files = find_files(pat)
    summarize("Base model", pat, files, verbose)

    for train_tc in ['plain', 'tc-self', 'tc-self+len']:
        print(f"\n  Train TC: {train_tc}")
        print(f"  {'─'*70}")
        for loss in ['pref', 'pref+v', 'comb+v', 'sft']:
            for semi in ['labelonly0.1', 'semi0.1']:
                label = f"    {loss:8s} {semi:12s}"
                kwargs = dict(train_task=train_task, train_tc=train_tc,
                              loss=loss, semi=semi)
                pattern, files = check_config(**kwargs)
                summarize(label, pattern, files, verbose)


def check_slides():
    """Check specifically what the slides bar charts need."""
    print("=" * 80)
    print("WHAT THE SLIDES CURRENTLY SHOW (Plain training group)")
    print("=" * 80)

    for train_task in ['plausibleqa', 'ambigqa', 'hypernym', 'ifeval']:
        print(f"\n--- {train_task} ---")

        pat = build_base_model_pattern(eval_task_family=train_task)
        files = find_files(pat)
        summarize("Base", pat, files)

        for loss, label in [('pref', 'RankAlign/Pref'), ('sft', 'SFT'),
                            ('comb+v', 'Comb'), ('comb+v', 'Comb semi')]:
            semi = 'semi0.1' if 'semi' in label else 'labelonly0.1'
            pattern, files = check_config(train_task=train_task, train_tc='plain',
                                          loss=loss, semi=semi)
            summarize(f"  {label}", pattern, files)


if __name__ == '__main__':
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith('-')]

    if not args or args[0] == 'slides':
        check_slides()
    elif args[0] == 'all':
        for task in ['plausibleqa', 'ambigqa', 'hypernym', 'ifeval']:
            check_task(task, verbose)
    elif args[0] in TRAINING_TASKS:
        check_task(args[0], verbose)
    else:
        print("Usage:")
        print("  python check_result_files.py                # check slides configs")
        print("  python check_result_files.py slides         # check slides configs")
        print("  python check_result_files.py all [-v]       # check all tasks")
        print("  python check_result_files.py plausibleqa -v # check one task (verbose)")
        print("  python check_result_files.py hypernym -v")
