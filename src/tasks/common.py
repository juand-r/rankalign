"""
Shared utilities for task modules.

Provides common functions that are repeated across multiple task implementations.
Import from here to reduce boilerplate.
"""

import csv
import hashlib
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
    """Build shortened model name for filenames.

    For absolute filesystem paths, uses just the basename to avoid exceeding
    Linux's 255-char filename limit when the adapter path is long.

    When the basename overflows the cap (>160 chars), the function falls back
    to the abbreviated form produced by `checkpoint_name_parser.to_hf_repo_name`,
    which preserves every flag in human-readable shorthand
    (e.g. ``v7-gemma-2-9b-it-d2.69-e2-...-tcs-nv1-ng1-vlo-fsx-ppd-sm0.1-fix1``).
    Hashing is only used as a last-resort safety net if even the abbreviated
    form exceeds the cap (should never happen with current settings; max
    observed abbreviated length is ~100 chars).
    """
    if os.path.isabs(modelname):
        # Local absolute path: use basename (avoids path-length explosion)
        raw = os.path.basename(modelname)
    elif '/' in modelname and not modelname.startswith('.'):
        # HF-style path like "org/model-name"
        raw = 'v6-' + modelname.replace('/', '_')
    else:
        raw = modelname.split('/')[-1].replace('--', '_')
    # Cap at 160 chars so total filename stays within Linux's 255-byte limit
    # (task + suffix overhead is ~90 chars; 160+90 = 250 < 255)
    MAX_LEN = 160
    if len(raw) <= MAX_LEN:
        return raw

    # Overflow: try human-readable abbreviation first.
    try:
        # Local import to avoid a circular dependency at module load time.
        from checkpoint_name_parser import (
            parse_checkpoint_name,
            to_hf_repo_name,
        )
        parsed = parse_checkpoint_name(raw)
        # Drop the "rankalign-" prefix (8+ chars saved; we know the folder is
        # outputs/scores_*.csv from rankalign already) and relax the 96-char
        # HF cap (we only need to fit MAX_LEN here, not HF repo IDs).
        short = to_hf_repo_name(parsed, prefix='', max_len=None)
        if len(short) <= MAX_LEN:
            return short
    except Exception:
        # Names that don't fit the rankalign checkpoint format fall through
        # to the hash truncation below.
        pass

    # Last-resort: deterministic hash truncation.
    h = hashlib.md5(raw.encode()).hexdigest()[:8]
    raw = raw[:MAX_LEN - 9] + '_' + h
    return raw


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
