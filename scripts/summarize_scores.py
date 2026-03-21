#!/usr/bin/env python3
"""
Lightweight scores summarizer — computes Gen ROC and Correlation for all
scores_*.csv files and outputs a single summary CSV.

Metric computation is identical to dashboard_viz_refactor.py:
  - Gen ROC: sklearn roc_auc_score(labels, gen_scores)
  - Val ROC: sklearn roc_auc_score(labels, val_scores)
  - Correlation: scipy pearsonr(gen_scores, val_scores)  [all / pos / neg]
  - Val Acc: sklearn accuracy_score with threshold 0 (log-odds) or log(0.5) (log-probs)

Usage:
    cd rankalign-longform
    python scripts/summarize_scores.py                          # use defaults
    python scripts/summarize_scores.py -c config/my_config.json # custom config
    python scripts/summarize_scores.py -o results.csv           # custom output path
    python scripts/summarize_scores.py --outputs-dir ./outputs  # override outputs dir
"""

import argparse
import json
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, accuracy_score


# =============================================================================
# DEFAULTS (same as dashboard_viz_refactor.py)
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
DEFAULT_OUTPUTS_DIR = SCRIPT_DIR.parent / 'outputs'
DEFAULT_CONFIG_FILE = SCRIPT_DIR.parent / 'config' / 'dashboard_config.json'

DEFAULT_CONFIG = {
    'outputs_dir': str(DEFAULT_OUTPUTS_DIR),
    'file_pattern': 'scores_*.csv',
    'task_pattern': r'hypernym-([a-zA-Z]+)',
    'split_patterns': {
        'train': '_train_',
        'test': '_test_'
    },
    'label_column': 'gpt4_ground_truth',
    'label_map': {'yes': 1, 'no': 0},
    'gen_score_col': 'gen_score',
    'val_score_col': 'val_score',
    'base_pattern': r'^scores_v6-google_gemma-2-2b_',
    'finetuned_pattern': r'^scores_v6-google_gemma-2-2b-delta',
    'union_models': {
        'training_task': 'hypernym-concat-bananas-to-dogs',
        'eval_task_pattern': r'force-same-x_(hypernym-[a-zA-Z]+)_',
    },
    'eval_columns': {
        'raw': 'gen_score',
        'tc': 'gen_score_typcorr',
        'lenorm': 'gen_score_lenorm',
        'tc+lenorm': 'gen_score_typcorr_lenorm'
    },
}


# =============================================================================
# FILEINFO DATACLASS (identical to dashboard)
# =============================================================================

@dataclass
class FileInfo:
    path: str
    filename: str
    task: str
    dataset: str
    split: str
    is_base: bool
    direction: Optional[str]
    is_union: bool
    pref_weight: Optional[float]
    nllv_weight: Optional[float]
    nllg_weight: Optional[float]
    has_tco: bool
    has_norm: bool
    has_vallogodds: bool
    training_mode: Optional[str]
    category: str
    row_label: str
    timestamp: str


# =============================================================================
# FILENAME PARSING (identical to dashboard)
# =============================================================================

def _extract_float(pattern, text):
    match = re.search(pattern, text)
    if match:
        try:
            return float(match.group('weight'))
        except (ValueError, IndexError):
            return None
    return None


def _extract_timestamp(filename):
    match = re.search(r'(\d{8}_\d{6})\.csv$', filename)
    return match.group(1) if match else '00000000_000000'


def _extract_task(stem, task_pattern, union_config):
    if union_config:
        training_task = union_config.get('training_task', '')
        eval_pattern = union_config.get('eval_task_pattern', '')
        if training_task and training_task in stem and 'force-same-x' in stem:
            if eval_pattern:
                eval_match = re.search(eval_pattern, stem)
                if eval_match and eval_match.groups():
                    full_match = re.search(r'(hypernym-[a-zA-Z]+)', eval_match.group(0))
                    if full_match:
                        task = full_match.group(1)
                        dataset = task.split('-')[-1] if '-' in task else task
                        return task, dataset

    all_matches = list(re.finditer(task_pattern, stem))
    if not all_matches:
        return None, None

    match = all_matches[-1]
    if match.groups():
        base_task = task_pattern.split('(')[0].rstrip('-').rstrip('_')
        if not base_task:
            base_task = 'task'
        dataset = match.group(1)
        task = f"{base_task}-{dataset}" if base_task else dataset
    else:
        task = match.group(0)
        dataset = task
    return task, dataset


def _extract_split(stem, split_patterns):
    for split_name, pattern in split_patterns.items():
        if pattern in stem:
            return split_name
    return 'unknown'


def build_row_label(category, training_mode, has_tco, has_norm, has_vallogodds):
    if category == 'Base':
        return 'Base'
    parts = [f"{category}-{training_mode}"]
    if has_tco:
        parts.append('tco')
    if has_norm:
        parts.append('norm')
    if has_vallogodds:
        parts.append('v')
    return '-'.join(parts)


def parse_filename(csv_file, config):
    name = Path(csv_file).name
    stem = Path(csv_file).stem
    task_pattern = config.get('task_pattern', r'hypernym-([a-zA-Z]+)')
    base_pattern = config.get('base_pattern')
    finetuned_pattern = config.get('finetuned_pattern')
    union_config = config.get('union_models', {})

    task, dataset = _extract_task(stem, task_pattern, union_config)
    if task is None:
        return None

    split = _extract_split(stem, config.get('split_patterns', {}))
    if split == 'unknown':
        return None

    timestamp = _extract_timestamp(name)

    is_finetuned = bool(finetuned_pattern and re.match(finetuned_pattern, stem))
    is_base = bool(base_pattern and re.match(base_pattern, stem) and not is_finetuned)

    if is_base:
        return FileInfo(
            path=str(csv_file), filename=name,
            task=task, dataset=dataset, split=split,
            is_base=True, direction=None, is_union=False,
            pref_weight=None, nllv_weight=None, nllg_weight=None,
            has_tco=False, has_norm=False, has_vallogodds=False,
            training_mode=None, category='Base', row_label='Base',
            timestamp=timestamp,
        )

    if not is_finetuned:
        return None

    if '_d2g_' not in stem:
        return None

    is_union = False
    if union_config:
        training_task = union_config.get('training_task', '')
        eval_pattern = union_config.get('eval_task_pattern', '')
        if training_task and training_task in stem and 'force-same-x' in stem:
            is_union = True
            if eval_pattern:
                eval_match = re.search(eval_pattern, stem)
                if eval_match and eval_match.groups():
                    full_match = re.search(r'(hypernym-[a-zA-Z]+)', eval_match.group(0))
                    if full_match:
                        task = full_match.group(1)
                        dataset = task.split('-')[-1] if '-' in task else task

    if not is_union:
        all_matches = list(re.finditer(task_pattern, stem))
        if len(all_matches) >= 2:
            training_match = all_matches[0]
            eval_match = all_matches[-1]
            if training_match.groups() and eval_match.groups():
                if training_match.group(1) != eval_match.group(1):
                    return None

    category = 'U' if is_union else 'S'

    pref_weight = _extract_float(r'(?:^|[_-])pref(?P<weight>\d+(?:\.\d+)?)', stem)
    nllv_weight = _extract_float(r'nllv(?P<weight>\d+(?:\.\d+)?)', stem)
    nllg_weight = _extract_float(r'nllg(?P<weight>\d+(?:\.\d+)?)', stem)

    eff_pref = pref_weight if pref_weight is not None else 1.0
    eff_nllv = nllv_weight if nllv_weight is not None else 0.0
    eff_nllg = nllg_weight if nllg_weight is not None else 0.0

    is_sft = (eff_pref == 0.0 and eff_nllv == 1.0 and eff_nllg == 1.0)
    is_pref_only = (eff_nllv == 0.0 and eff_nllg == 0.0)
    is_pref_nll = (eff_nllv == 1.0 and eff_nllg == 1.0 and eff_pref == 1.0)

    modes_matched = sum([is_sft, is_pref_only, is_pref_nll])
    if modes_matched != 1:
        print(f"  [WARN] Skipping ambiguous training mode: {name} "
              f"(pref={pref_weight}, nllv={nllv_weight}, nllg={nllg_weight})",
              file=sys.stderr)
        return None

    if is_sft:
        training_mode = 'SFT'
    elif is_pref_only:
        training_mode = 'Pref'
    else:
        training_mode = 'Comb'

    has_tco = '_tc-online_' in stem
    has_norm = '_lenorm_' in stem
    has_vallogodds = '_vallogodds' in stem

    row_label = build_row_label(category, training_mode, has_tco, has_norm, has_vallogodds)

    return FileInfo(
        path=str(csv_file), filename=name,
        task=task, dataset=dataset, split=split,
        is_base=False, direction='d2g', is_union=is_union,
        pref_weight=pref_weight, nllv_weight=nllv_weight, nllg_weight=nllg_weight,
        has_tco=has_tco, has_norm=has_norm, has_vallogodds=has_vallogodds,
        training_mode=training_mode, category=category, row_label=row_label,
        timestamp=timestamp,
    )


# =============================================================================
# DATA LOADING AND METRICS (identical to dashboard)
# =============================================================================

def load_scores_data(csv_path, config):
    df = pd.read_csv(csv_path)
    label_col = config.get('label_column', 'gpt4_ground_truth')
    label_map = config.get('label_map')

    if label_col in df.columns:
        gt_col = df[label_col]
        if gt_col.dtype in ['int64', 'float64', 'int', 'float']:
            df['label'] = gt_col.astype(int)
        elif label_map:
            df['label'] = gt_col.str.strip().str.lower().map(label_map)
            df['label'] = df['label'].fillna(0).astype(int)
        else:
            df['label'] = gt_col.str.strip().str.lower().map(
                {'yes': 1, 'no': 0, 'true': 1, 'false': 0}
            )
            df['label'] = df['label'].fillna(0).astype(int)
    return df


def compute_metrics(gen_scores, val_scores, labels, metric_type='log-odds'):
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
# FILE DISCOVERY AND DEDUP (identical to dashboard)
# =============================================================================

def discover_files(config):
    outputs_dir = Path(config['outputs_dir'])
    file_pattern = config.get('file_pattern', 'scores_*.csv')

    all_files = []
    skipped = 0
    for csv_file in sorted(outputs_dir.glob(file_pattern)):
        info = parse_filename(csv_file, config)
        if info is not None:
            all_files.append(info)
        else:
            skipped += 1

    print(f"Discovered {len(all_files)} valid files, skipped {skipped}")
    return all_files


def dedup_files(file_infos):
    """Group by (task, split, row_label), keep newest by timestamp."""
    groups = {}
    for f in file_infos:
        key = (f.task, f.split, f.row_label)
        if key not in groups:
            groups[key] = []
        groups[key].append(f)

    deduped = []
    for key, files in groups.items():
        files_sorted = sorted(files, key=lambda f: f.timestamp, reverse=True)
        deduped.append(files_sorted[0])
        if len(files) > 1:
            print(f"  [DEDUP] {key}: kept {files_sorted[0].filename}, "
                  f"dropped {[f.filename for f in files_sorted[1:]]}")
    return deduped


# =============================================================================
# MAIN: BUILD SUMMARY TABLE
# =============================================================================

def build_summary(config):
    file_infos = discover_files(config)
    file_infos = dedup_files(file_infos)

    eval_columns = config.get('eval_columns', DEFAULT_CONFIG['eval_columns'])
    rows = []

    for fi in file_infos:
        try:
            df = load_scores_data(fi.path, config)
        except Exception as e:
            print(f"  [ERROR] Loading {fi.filename}: {e}", file=sys.stderr)
            continue

        metric_type = 'log-odds' if 'log-odds' in fi.path else 'log-probs'

        for eval_name, gen_col in eval_columns.items():
            if gen_col not in df.columns or 'val_score' not in df.columns or 'label' not in df.columns:
                continue

            metrics = compute_metrics(
                df[gen_col].values,
                df['val_score'].values,
                df['label'].values,
                metric_type,
            )

            rows.append({
                'task': fi.task,
                'dataset': fi.dataset,
                'split': fi.split,
                'category': fi.category,
                'training_mode': fi.training_mode or 'none',
                'row_label': fi.row_label,
                'eval_variant': eval_name,
                'is_union': fi.is_union,
                'has_tco': fi.has_tco,
                'has_norm': fi.has_norm,
                'has_vallogodds': fi.has_vallogodds,
                'gen_roc': metrics['gen_roc'],
                'val_roc': metrics['val_roc'],
                'val_acc': metrics['acc'],
                'corr': metrics['corr'],
                'corr_pos': metrics['corr_pos'],
                'corr_neg': metrics['corr_neg'],
                'n_samples': len(df),
                'filename': fi.filename,
            })

    summary = pd.DataFrame(rows)

    # Sort for readability
    if not summary.empty:
        summary = summary.sort_values(
            ['task', 'split', 'row_label', 'eval_variant']
        ).reset_index(drop=True)

    return summary


def load_config(config_path=None):
    config = DEFAULT_CONFIG.copy()
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            user_config = json.load(f)
        config.update(user_config)
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Summarize scores_*.csv files into a single metrics table.'
    )
    parser.add_argument('-c', '--config', default=None,
                        help='Path to config JSON (default: config/dashboard_config.json if it exists)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output CSV path (default: outputs/summary_scores.csv)')
    parser.add_argument('--outputs-dir', default=None,
                        help='Override outputs directory')
    parser.add_argument('--no-config', action='store_true',
                        help='Ignore config file, use built-in defaults')
    args = parser.parse_args()

    # Load config
    if args.no_config:
        config = DEFAULT_CONFIG.copy()
    else:
        config_path = args.config or (DEFAULT_CONFIG_FILE if DEFAULT_CONFIG_FILE.exists() else None)
        config = load_config(config_path)
        if config_path:
            print(f"Using config: {config_path}")

    # Override outputs dir if specified
    if args.outputs_dir:
        config['outputs_dir'] = args.outputs_dir

    # Build summary
    summary = build_summary(config)

    if summary.empty:
        print("No scores files found or no metrics computed.", file=sys.stderr)
        sys.exit(1)

    # Output
    output_path = args.output or str(Path(config['outputs_dir']) / 'summary_scores.csv')
    summary.to_csv(output_path, index=False)
    print(f"\nSummary: {len(summary)} rows written to {output_path}")

    # Print a quick overview
    print(f"\nTasks: {sorted(summary['task'].unique())}")
    print(f"Splits: {sorted(summary['split'].unique())}")
    print(f"Row labels: {sorted(summary['row_label'].unique())}")
    print(f"Eval variants: {sorted(summary['eval_variant'].unique())}")

    # Show a compact table of Gen ROC and Correlation for the 'raw' eval variant
    raw = summary[summary['eval_variant'] == 'raw'].copy()
    if not raw.empty:
        print("\n--- Quick view: Gen ROC & Correlation (raw scores, test split) ---")
        test_raw = raw[raw['split'] == 'test']
        if test_raw.empty:
            test_raw = raw  # fall back to whatever splits exist
        pivot = test_raw.pivot_table(
            index='row_label',
            columns='task',
            values=['gen_roc', 'corr'],
            aggfunc='first',
        )
        pd.set_option('display.float_format', '{:.3f}'.format)
        pd.set_option('display.max_columns', 20)
        pd.set_option('display.width', 160)
        print(pivot.to_string())


if __name__ == '__main__':
    main()
