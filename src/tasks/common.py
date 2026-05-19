"""
Shared utilities for task modules.

Provides common functions that are repeated across multiple task implementations.
Import from here to reduce boilerplate.
"""

import csv
import os
from collections import namedtuple
from datetime import datetime

PromptCompletion = namedtuple("PromptCompletion", ["prompt", "completion"])


def normalize_yes_no(raw):
    """Normalize a raw label string to 'yes' or 'no'."""
    val = str(raw).strip().lower()
    return 'yes' if val in ('yes', 'true', '1') else 'no'


def get_field(obj, key, default=""):
    """Access a field from a dict, namedtuple, or object by key/attr name."""
    if hasattr(obj, key):
        return getattr(obj, key)
    try:
        return obj[key]
    except Exception:
        return default


def load_csv_items(filepath, fields=None):
    """Load rows from a CSV as dicts.

    Args:
        filepath: path to CSV file
        fields: if given, only keep these keys from each row
    """
    items = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if fields:
                items.append({k: row.get(k, '') for k in fields})
            else:
                items.append(dict(row))
    return items


def build_strategy_string(args, gen_shots, disc_shots, use_full_completion_logprobs):
    """Build the strategy metadata string from runtime flags.

    This was previously repeated inline 12 times across CSV-writing blocks.
    """
    strategy = f"gen:{gen_shots}_disc:{disc_shots}"
    if use_full_completion_logprobs:
        strategy += "_fullcomp"
    else:
        strategy += "_singletoken"
    if args.validator_log_odds:
        strategy += "_logodds"
    else:
        strategy += "_logprobs"
    if args.typicality_correction:
        if args.self_typicality:
            strategy += "_selftypcorr"
        elif args.neg_typicality:
            strategy += "_negtypcorr"
        else:
            strategy += "_typcorr"
        if args.base_typicality:
            strategy += "_basemodel"
    if getattr(args, 'exit_layer', None) is not None:
        strategy += f"_exit{args.exit_layer}"
    return strategy


def build_model_short(modelname):
    """Build shortened model name for filenames."""
    if '/' in modelname and not modelname.startswith('.'):
        return 'v6-' + modelname.replace('/', '_')
    else:
        return modelname.split('/')[-1].replace('--', '_')


def build_csv_filename(outputs_dir, self_prefix, modelname, task, split, args,
                       eos_suffix, v2_suffix=""):
    """Build the output CSV filename from runtime parameters."""
    model_short = build_model_short(modelname)
    metric_suffix = "_log-odds" if args.validator_log_odds else "_log-probs"
    eval_tc_suffix = "_tc" if args.typicality_correction else ""
    eval_lenorm_suffix = "_evallenorm" if args.length_normalize else ""
    timestamp = datetime.now().strftime("%Y%m%d")
    return (f"{outputs_dir}/scores_{self_prefix}{model_short}_{task}_{split}"
            f"{v2_suffix}{metric_suffix}{eval_tc_suffix}{eval_lenorm_suffix}"
            f"{eos_suffix}_{timestamp}.csv")


def compute_score_columns(i, gen_scores_raw, gen_scores_typcorr, all_num_tokens):
    """Compute the standard score columns for a single item.

    Returns: (num_toks, gen_score_raw, gen_score_typcorr_val,
              gen_score_lenorm, gen_score_typcorr_lenorm)
    """
    num_toks = all_num_tokens[i]
    gen_score_raw = gen_scores_raw[i]
    gen_score_typcorr_val = (gen_scores_typcorr[i]
                             if gen_scores_typcorr is not None else float('nan'))
    gen_score_lenorm = (gen_score_raw / num_toks
                        if num_toks > 0 else float('nan'))
    gen_score_typcorr_lenorm = (gen_score_typcorr_val / num_toks
                                if (gen_scores_typcorr is not None and num_toks > 0)
                                else float('nan'))
    return (num_toks, gen_score_raw, gen_score_typcorr_val,
            gen_score_lenorm, gen_score_typcorr_lenorm)


def write_scores_csv(filename, header, rows):
    """Write a scores CSV with the given header and rows."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    print(f"Detailed scores saved to: {filename}")
