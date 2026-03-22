#!/usr/bin/env python3
"""
Universal scores summarizer — computes Gen ROC, Val ROC, Correlation, and
Val Acc for ALL scores_*.csv files in outputs/ and writes a single summary CSV.

Handles all file types: base model, self-TC, finetuned, multi-model, all tasks.

Reuses filename parsing from score_file_parsing.py (shared with
dashboard_viz_refactor.py) — one set of task extraction regexes, not two.

Usage:
    cd rankalign-longform
    python scripts/summarize_scores.py                          # use defaults
    python scripts/summarize_scores.py -o results.csv           # custom output path
    python scripts/summarize_scores.py --outputs-dir ./outputs  # override outputs dir
"""

import argparse
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, accuracy_score

from score_file_parsing import (
    _extract_task,
    _extract_split,
    _extract_timestamp,
    _extract_float,
)


# =============================================================================
# DEFAULTS
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
DEFAULT_OUTPUTS_DIR = SCRIPT_DIR.parent / 'outputs'

# Task family configs — each defines how to find and parse files for one family.
# task_pattern: regex with a capture group for the dataset-specific part.
# split_patterns: substring -> split name mapping.
TASK_FAMILY_CONFIGS = [
    {
        'name': 'plausibleqa',
        'task_pattern': r'plausibleqa-((?:nq|webq|trivia)_\d+)',
        'file_pattern': 'scores_*plausibleqa*.csv',
        'split_patterns': {'test': '_test_', 'train': '_train_'},
    },
    {
        'name': 'ifeval',
        'task_pattern': r'ifeval-(prompt_\d+)',
        'file_pattern': 'scores_*ifeval*.csv',
        'split_patterns': {'test': '_test_', 'train': '_train_'},
    },
    {
        'name': 'ambigqa',
        'task_pattern': r'ambigqa-([a-zA-Z]+)',
        'file_pattern': 'scores_*ambigqa*.csv',
        'split_patterns': {'test': '_test_', 'train': '_train_'},
    },
    {
        'name': 'hypernym',
        'task_pattern': r'hypernym-([a-zA-Z][a-zA-Z ]+)',
        'file_pattern': 'scores_*hypernym*.csv',
        'split_patterns': {'test': '_test_', 'train': '_train_'},
    },
]

# Eval score column variants (gen_score variants; val_score is always val_score)
EVAL_COLUMNS = {
    'raw': 'gen_score',
    'tc': 'gen_score_typcorr',
    'lenorm': 'gen_score_lenorm',
    'tc+lenorm': 'gen_score_typcorr_lenorm',
}

# Label column detection: try these in order
LABEL_COLUMNS = [
    ('gpt4_ground_truth', {'yes': 1, 'no': 0}),
    ('correct', {'yes': 1, 'no': 0}),
]


# =============================================================================
# METRIC COMPUTATION
# =============================================================================

def compute_metrics(gen_scores, val_scores, labels, metric_type='log-odds'):
    """Compute ROC-AUC, accuracy, and correlation metrics."""
    gen_scores_np = np.array(gen_scores)
    val_scores_np = np.array(val_scores)
    labels_np = np.array(labels)

    valid_mask = ~(np.isnan(gen_scores_np) | np.isnan(val_scores_np))
    if valid_mask.sum() < 2:
        return {'corr': np.nan, 'corr_pos': np.nan, 'corr_neg': np.nan,
                'acc': np.nan, 'val_roc': np.nan, 'gen_roc': np.nan}

    gen_valid = gen_scores_np[valid_mask]
    val_valid = val_scores_np[valid_mask]
    labels_valid = labels_np[valid_mask]

    pos_mask = labels_valid == 1
    neg_mask = labels_valid == 0

    try:
        corr_all, _ = pearsonr(gen_valid, val_valid)
    except Exception:
        corr_all = np.nan

    try:
        corr_pos = pearsonr(gen_valid[pos_mask], val_valid[pos_mask])[0] if pos_mask.sum() > 1 else np.nan
    except Exception:
        corr_pos = np.nan

    try:
        corr_neg = pearsonr(gen_valid[neg_mask], val_valid[neg_mask])[0] if neg_mask.sum() > 1 else np.nan
    except Exception:
        corr_neg = np.nan

    threshold = 0 if metric_type == 'log-odds' else np.log(0.5)
    preds = (val_valid > threshold).astype(int)
    acc = accuracy_score(labels_valid, preds)

    try:
        val_roc = roc_auc_score(labels_valid, val_valid)
    except Exception:
        val_roc = np.nan

    try:
        gen_roc = roc_auc_score(labels_valid, gen_valid)
    except Exception:
        gen_roc = np.nan

    return {
        'corr': corr_all, 'corr_pos': corr_pos, 'corr_neg': corr_neg,
        'acc': acc, 'val_roc': val_roc, 'gen_roc': gen_roc
    }


# =============================================================================
# FILENAME METADATA EXTRACTION
# =============================================================================

def extract_metadata(csv_file, task_configs):
    """Extract metadata from a scores CSV file path.

    Uses the shared _extract_task, _extract_split, _extract_timestamp helpers
    from score_file_parsing.py (same code the dashboard uses).

    Handles features the dashboard doesn't need:
    - self- prefix (self-typicality-corrected models)
    - Multiple model families (Gemma, Llama, Qwen) with varying name formats
    - Finetuned model detection via -delta (without requiring specific base patterns)

    Returns a dict with: model, task, split, self_tc, eval_tc, finetuned,
    metric_type, timestamp, training_config.  Returns None if unparseable.
    """
    filename = Path(csv_file).name
    stem = Path(csv_file).stem

    if not stem.startswith('scores_'):
        return None

    rest = stem[len('scores_'):]

    # --- Self-TC prefix ---
    self_tc = rest.startswith('self-')
    if self_tc:
        rest = rest[len('self-'):]

    # --- Timestamp (shared helper) ---
    timestamp = _extract_timestamp(filename)
    # Strip timestamp from rest for further parsing
    ts_match = re.search(r'_(\d{8}_\d{6})$', rest)
    if ts_match:
        rest = rest[:ts_match.start()]

    # --- eval_tc suffix ---
    eval_tc = rest.endswith('_evaltc')
    if eval_tc:
        rest = rest[:-len('_evaltc')]

    # --- metric_type ---
    metric_type = 'log-odds'
    if '_log-odds' in rest:
        rest = rest.replace('_log-odds', '', 1)
    elif '_log-probs' in rest:
        metric_type = 'log-probs'
        rest = rest.replace('_log-probs', '', 1)

    # --- Split (shared helper) ---
    # Handle _test_v2 pattern (hypernym test sets) before generic split detection
    split = None
    if '_test_v2' in rest:
        split = 'test'
        rest = rest.replace('_test_v2', '')
    else:
        split_patterns = {'test': '_test', 'train': '_train'}
        for split_name, pattern in split_patterns.items():
            if pattern in rest:
                split = split_name
                rest = rest.replace(pattern, '', 1)
                break

    if split is None:
        return None

    # --- Task extraction (shared helper, try each family config) ---
    task = None
    task_match_start = None
    for cfg in task_configs:
        task_pattern = cfg['task_pattern']
        found_task, found_dataset = _extract_task(rest, task_pattern, {})
        if found_task is not None:
            # Find the match position in rest for model extraction
            all_matches = list(re.finditer(task_pattern, rest))
            if all_matches:
                task = found_task
                task_match_start = all_matches[-1].start()
                # Grab full match span for after-task extraction
                task_match_end = all_matches[-1].end()
                break

    if task is None:
        return None

    # --- Model extraction ---
    model_part = rest[:task_match_start].rstrip('_')
    if not model_part:
        return None

    # --- Finetuned detection ---
    finetuned = bool(re.search(r'-delta', model_part))

    # For finetuned models, separate base model from training config
    training_config = ''
    if finetuned:
        delta_match = re.search(r'-delta', model_part)
        if delta_match:
            base_model = model_part[:delta_match.start()]
            training_config = model_part[delta_match.start() + 1:]  # skip leading '-'
            model_part = base_model

    # Capture anything after the eval task as extra config
    after_task = rest[task_match_end:]
    if after_task:
        extra = after_task.strip('_')
        if extra:
            training_config = (training_config + '_' + extra).strip('_') if training_config else extra

    return {
        'model': model_part,
        'task': task,
        'split': split,
        'self_tc': self_tc,
        'eval_tc': eval_tc,
        'finetuned': finetuned,
        'metric_type': metric_type,
        'timestamp': timestamp,
        'training_config': training_config,
    }


# =============================================================================
# LABEL LOADING
# =============================================================================

def load_labels(df):
    """Detect the label column and return integer labels, or None on failure."""
    for col_name, label_map in LABEL_COLUMNS:
        if col_name not in df.columns:
            continue
        gt = df[col_name]
        if gt.dtype in ('int64', 'float64', 'int', 'float'):
            return gt.astype(int).values
        mapped = gt.str.strip().str.lower().map(label_map)
        if mapped.isna().all():
            continue
        return mapped.fillna(0).astype(int).values
    return None


# =============================================================================
# FILE DISCOVERY, DEDUP, SUMMARY
# =============================================================================

def discover_and_summarize(outputs_dir):
    """Scan all scores_*.csv, extract metadata, compute metrics, dedup."""
    outputs_path = Path(outputs_dir)
    csv_files = sorted(outputs_path.glob('scores_*.csv'))

    print(f"Found {len(csv_files)} scores_*.csv files in {outputs_path}")

    # Phase 1: parse all filenames
    parsed = []
    skipped = 0
    for csv_file in csv_files:
        meta = extract_metadata(csv_file, TASK_FAMILY_CONFIGS)
        if meta is None:
            skipped += 1
            continue
        meta['path'] = str(csv_file)
        meta['filename'] = csv_file.name
        parsed.append(meta)

    print(f"Parsed {len(parsed)} files, skipped {skipped} (unrecognized pattern)")

    # Phase 2: dedup — group by identity key, keep newest
    groups = {}
    for meta in parsed:
        key = (meta['model'], meta['task'], meta['split'], meta['self_tc'],
               meta['eval_tc'], meta['training_config'])
        if key not in groups or meta['timestamp'] > groups[key]['timestamp']:
            groups[key] = meta

    deduped = list(groups.values())
    n_dropped = len(parsed) - len(deduped)
    if n_dropped > 0:
        print(f"Dedup: kept {len(deduped)}, dropped {n_dropped} older duplicates")

    # Phase 3: compute metrics for each file x eval_variant
    rows = []
    errors = 0
    for meta in deduped:
        try:
            df = pd.read_csv(meta['path'])
        except Exception as e:
            print(f"  [ERROR] Loading {meta['filename']}: {e}", file=sys.stderr)
            errors += 1
            continue

        labels = load_labels(df)
        if labels is None:
            print(f"  [WARN] No label column found in {meta['filename']}", file=sys.stderr)
            errors += 1
            continue

        if 'val_score' not in df.columns:
            print(f"  [WARN] No val_score column in {meta['filename']}", file=sys.stderr)
            errors += 1
            continue

        val_scores = df['val_score'].values

        for eval_name, gen_col in EVAL_COLUMNS.items():
            if gen_col not in df.columns:
                continue

            metrics = compute_metrics(
                df[gen_col].values,
                val_scores,
                labels,
                meta['metric_type'],
            )

            rows.append({
                'model': meta['model'],
                'task': meta['task'],
                'split': meta['split'],
                'self_tc': meta['self_tc'],
                'eval_tc': meta['eval_tc'],
                'finetuned': meta['finetuned'],
                'training_config': meta['training_config'],
                'eval_variant': eval_name,
                'gen_roc': metrics['gen_roc'],
                'val_roc': metrics['val_roc'],
                'val_acc': metrics['acc'],
                'corr': metrics['corr'],
                'corr_pos': metrics['corr_pos'],
                'corr_neg': metrics['corr_neg'],
                'n_samples': len(df),
                'filename': meta['filename'],
            })

    if errors:
        print(f"  {errors} files had errors/warnings")

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ['model', 'task', 'split', 'eval_variant']
        ).reset_index(drop=True)

    return summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Summarize ALL scores_*.csv files into a single metrics table.'
    )
    parser.add_argument('-o', '--output', default=None,
                        help='Output CSV path (default: outputs/summary_scores.csv)')
    parser.add_argument('--outputs-dir', default=str(DEFAULT_OUTPUTS_DIR),
                        help='Outputs directory (default: outputs/)')
    args = parser.parse_args()

    summary = discover_and_summarize(args.outputs_dir)

    if summary.empty:
        print("No scores files found or no metrics computed.", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or str(Path(args.outputs_dir) / 'summary_scores.csv')
    summary.to_csv(output_path, index=False)
    print(f"\nSummary: {len(summary)} rows written to {output_path}")

    # Quick overview
    print(f"\nModels ({len(summary['model'].unique())}): {sorted(summary['model'].unique())}")
    print(f"Tasks ({len(summary['task'].unique())}): {sorted(summary['task'].unique())[:20]}...")
    print(f"Splits: {sorted(summary['split'].unique())}")
    print(f"Eval variants: {sorted(summary['eval_variant'].unique())}")
    print(f"Self-TC files: {summary['self_tc'].sum()} rows")
    print(f"Finetuned files: {summary['finetuned'].sum()} rows")

    # Per-family task counts for base models (non-self, non-finetuned)
    base_nonselfeval = summary[
        (~summary['finetuned']) & (~summary['self_tc']) &
        (summary['eval_variant'] == 'raw') & (summary['split'] == 'test')
    ]
    if not base_nonselfeval.empty:
        print("\n--- Tasks per base model (raw, test, non-self) ---")
        for model in sorted(base_nonselfeval['model'].unique()):
            tasks = base_nonselfeval[base_nonselfeval['model'] == model]['task']
            by_family = {}
            for t in tasks:
                fam = t.split('-')[0]
                by_family[fam] = by_family.get(fam, 0) + 1
            counts = ', '.join(f"{fam}={n}" for fam, n in sorted(by_family.items()))
            print(f"  {model}: {len(tasks)} tasks ({counts})")

    # Compact view: Gen ROC for raw scores, test split, base models
    raw_test = summary[
        (summary['eval_variant'] == 'raw') &
        (summary['split'] == 'test') &
        (~summary['finetuned'])
    ]
    if not raw_test.empty:
        print("\n--- Quick view: Gen ROC (raw, test, base models) ---")
        pd.set_option('display.float_format', '{:.3f}'.format)
        pd.set_option('display.max_columns', 20)
        pd.set_option('display.width', 160)
        pivot = raw_test.pivot_table(
            index='model',
            columns='task',
            values='gen_roc',
            aggfunc='first',
        )
        if pivot.shape[1] > 10:
            print(pivot.iloc[:, :10].to_string())
            print(f"  ... ({pivot.shape[1]} tasks total)")
        else:
            print(pivot.to_string())


if __name__ == '__main__':
    main()
