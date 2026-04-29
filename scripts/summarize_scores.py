#!/usr/bin/env python3
"""
Universal scores summarizer -- computes Gen ROC, Val ROC, Correlation, and
Val Acc for ALL scores_*.csv files in outputs/ and writes a single summary CSV.

Handles all file types: base model, self-TC, finetuned, multi-model, all tasks.

Reuses filename parsing from score_file_parsing.py (shared with
dashboard_viz_refactor.py) -- one set of task extraction regexes, not two.

Supports incremental mode: pulls existing summary from HuggingFace, computes
metrics only for new score files, merges, and re-uploads.

Usage:
    cd rankalign-longform
    python scripts/summarize_scores.py                          # full scan, local CSV
    python scripts/summarize_scores.py -o results.csv           # custom output path
    python scripts/summarize_scores.py --outputs-dir ./outputs  # override outputs dir
    python scripts/summarize_scores.py --upload-hf              # upload to HuggingFace
    python scripts/summarize_scores.py --upload-hf --incremental  # only process new files
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

# Add parent src/ to path for checkpoint_name_parser
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from checkpoint_name_parser import parse_checkpoint_name, to_hf_repo_name


# =============================================================================
# DEFAULTS
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
DEFAULT_OUTPUTS_DIR = SCRIPT_DIR.parent / 'outputs'
DEFAULT_SUMMARIES_DIR = SCRIPT_DIR.parent / 'output-metrics'
DEFAULT_HF_DATASET = 'rankalign-eval-summary'

# Hypernym dedup: pairs to remove entirely and conflict resolution
HYPERNYM_REMOVE_PAIRS = set()
HYPERNYM_CONFLICT_KEEPS = {}  # (noun1, noun2) -> keep_label

_remove_csv = DATA_DIR / 'hypernym_bad_examples.csv'
if _remove_csv.exists():
    _df = pd.read_csv(_remove_csv)
    HYPERNYM_REMOVE_PAIRS = set(zip(_df['noun1'], _df['noun2']))

_keeps_csv = DATA_DIR / 'hypernym_conflict_keeps.csv'
if _keeps_csv.exists():
    _df = pd.read_csv(_keeps_csv)
    for _, row in _df.iterrows():
        HYPERNYM_CONFLICT_KEEPS[(row['noun1'], row['noun2'])] = row['keep_label'].strip().lower()

# Task family configs -- each defines how to find and parse files for one family.
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
# MODEL NAME DERIVATION
# =============================================================================

def derive_model_names(model_path, model_col, training_config, finetuned):
    """Derive local_model_name and hf_model_name.

    If model_path is available (new CSVs), use it directly.
    Otherwise, reconstruct from model + training_config columns (verified lossless).

    Returns (local_model_name, hf_model_name).
    """
    if not finetuned:
        # Base model: HF name is the model itself (e.g. "google/gemma-2-2b")
        if model_path:
            # model_path is the raw HF name like "google/gemma-2-2b"
            return model_path, model_path
        else:
            # Reconstruct: model_col is "v6-google_gemma-2-2b" → "google/gemma-2-2b"
            stripped = re.sub(r'^v\d+-', '', model_col)
            hf_name = stripped.replace('_', '/', 1)
            return hf_name, hf_name

    # Finetuned model
    if model_path:
        # model_path is a local path like "../models/v6-google--gemma-2-2b-delta..."
        local_name = Path(model_path).name
        # Strip _merged suffix if present
        if local_name.endswith('_merged'):
            local_name = local_name[:-len('_merged')]
    else:
        # Reconstruct local dir name from model + training_config (verified exact match)
        model_part = model_col.replace('_', '--', 1)
        local_name = f"{model_part}-{training_config.replace('_', '--')}"

    # Derive HF repo name using checkpoint_name_parser
    try:
        parsed = parse_checkpoint_name(local_name)
        hf_name = to_hf_repo_name(parsed)
    except (ValueError, KeyError):
        hf_name = local_name  # fallback: use local name as-is

    return local_name, hf_name


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

    Returns a dict with: model, task, split, self_tc, neg_tc, gpt2_tc, finetuned,
    metric_type, timestamp, training_config.  Returns None if unparseable.
    """
    filename = Path(csv_file).name
    stem = Path(csv_file).stem

    if not stem.startswith('scores_'):
        return None

    rest = stem[len('scores_'):]

    # --- TC prefix ---
    # Two orthogonal dimensions:
    #   conditioning: self (unconditional) vs neg (negated prompt)
    #   model: scoring model (default) vs base model (basetyp)
    # Prefix encoding: basetypneg- (base+neg), basetyp- (base+self),
    #                   neg- (scoring+neg), self- (scoring+self)
    if rest.startswith('basetypneg-'):
        basetyp_tc = True
        neg_tc = True
        self_tc = False
        rest = rest[len('basetypneg-'):]
    elif rest.startswith('basetyp-'):
        basetyp_tc = True
        neg_tc = False
        self_tc = True
        rest = rest[len('basetyp-'):]
    elif rest.startswith('self-'):
        basetyp_tc = False
        neg_tc = False
        self_tc = True
        rest = rest[len('self-'):]
    elif rest.startswith('neg-'):
        basetyp_tc = False
        neg_tc = True
        self_tc = False
        rest = rest[len('neg-'):]
    else:
        basetyp_tc = False
        neg_tc = False
        self_tc = False

    # --- Timestamp (shared helper) ---
    timestamp = _extract_timestamp(filename)
    # Strip timestamp from rest for further parsing
    ts_match = re.search(r'_(\d{8}(?:_\d{6})?)$', rest)
    if ts_match:
        rest = rest[:ts_match.start()]

    # --- Strip _eos suffix (EOS included in completion scoring) ---
    has_eos = rest.endswith('_eos')
    if has_eos:
        rest = rest[:-len('_eos')]

    # --- Strip _evallenorm suffix (eval-time lenorm, independent of TC) ---
    has_evallenorm = rest.endswith('_evallenorm')
    if has_evallenorm:
        rest = rest[:-len('_evallenorm')]

    # --- TC suffix (_evaltc from eval.py, _tc from eval_by_claude.py) ---
    # If any prefix (self-, neg-, basetyp-, basetypneg-) is present, the _tc suffix
    # is redundant (strip it but ignore).
    # If no prefix and _evaltc or _tc suffix is present, this is GPT-2 TC.
    has_evaltc_suffix = rest.endswith('_evaltc')
    has_tc_suffix = rest.endswith('_tc') and not has_evaltc_suffix
    if has_evaltc_suffix:
        rest = rest[:-len('_evaltc')]
    elif has_tc_suffix:
        rest = rest[:-len('_tc')]
    gpt2_tc = (has_evaltc_suffix or has_tc_suffix) and not self_tc and not neg_tc and not basetyp_tc

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
        'neg_tc': neg_tc,
        'gpt2_tc': gpt2_tc,
        'basetyp_tc': basetyp_tc,
        'include_eos': has_eos,
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
# HUGGINGFACE HELPERS
# =============================================================================

def pull_existing_summary(hf_org, hf_dataset):
    """Pull existing summary dataset from HuggingFace. Returns DataFrame or empty DataFrame."""
    repo_id = f"{hf_org}/{hf_dataset}"
    try:
        from datasets import load_dataset
        ds = load_dataset(repo_id, split='train')
        df = ds.to_pandas()
        print(f"Pulled existing summary from {repo_id}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"No existing summary on HF ({repo_id}): {e}")
        return pd.DataFrame()


def push_summary_to_hf(summary_df, hf_org, hf_dataset):
    """Push summary DataFrame to HuggingFace as a dataset."""
    repo_id = f"{hf_org}/{hf_dataset}"
    from datasets import Dataset
    from huggingface_hub import HfApi
    ds = Dataset.from_pandas(summary_df, preserve_index=False)
    ds.push_to_hub(repo_id, private=False)
    print(f"Pushed {len(summary_df)} rows to {repo_id}")


# =============================================================================
# FILE DISCOVERY, DEDUP, SUMMARY
# =============================================================================

def discover_and_summarize(outputs_dir, existing_filenames=None, file_pattern='scores_*.csv',
                           model_filter=None, epoch_filter=None):
    """Scan score CSV files, extract metadata, compute metrics, dedup.

    If existing_filenames is provided (set of filename strings), skip files
    already in the existing summary (incremental mode).
    model_filter: substring that must appear in the model name (e.g. 'v6').
    epoch_filter: if set, keep base models (no epoch) + finetuned matching this
                  epoch string (e.g. 'epoch2').
    """
    outputs_path = Path(outputs_dir)
    csv_files = sorted(outputs_path.glob(file_pattern))

    print(f"Found {len(csv_files)} scores_*.csv files in {outputs_path}")

    # Phase 1: parse all filenames
    parsed = []
    skipped = 0
    skipped_existing = 0
    skipped_filter = 0
    for csv_file in csv_files:
        if existing_filenames and csv_file.name in existing_filenames:
            skipped_existing += 1
            continue
        meta = extract_metadata(csv_file, TASK_FAMILY_CONFIGS)
        if meta is None:
            skipped += 1
            continue
        # Model filter
        if model_filter and model_filter not in meta['model']:
            skipped_filter += 1
            continue
        # Epoch filter: base models (not finetuned) always pass;
        # finetuned models must match the epoch string in training_config.
        # Exception: ifeval models use epoch1 instead of the default epoch filter.
        if epoch_filter and meta['finetuned']:
            if epoch_filter not in meta['training_config']:
                skipped_filter += 1
                continue
        meta['path'] = str(csv_file)
        meta['filename'] = csv_file.name
        parsed.append(meta)

    print(f"Parsed {len(parsed)} new files, skipped {skipped} (unrecognized pattern)")
    if skipped_filter:
        print(f"Skipped {skipped_filter} files (filtered out by model/epoch)")
    if skipped_existing:
        print(f"Skipped {skipped_existing} files already in existing summary")

    if not parsed:
        return pd.DataFrame()

    # Phase 2: dedup -- group by identity key, keep newest
    groups = {}
    for meta in parsed:
        key = (meta['model'], meta['task'], meta['split'], meta['self_tc'],
               meta['neg_tc'], meta['gpt2_tc'], meta['basetyp_tc'],
               meta['include_eos'], meta['training_config'])
        if key not in groups or meta['timestamp'] > groups[key]['timestamp']:
            groups[key] = meta

    deduped = list(groups.values())
    n_dropped = len(parsed) - len(deduped)
    if n_dropped > 0:
        print(f"Dedup: kept {len(deduped)}, dropped {n_dropped} older duplicates")

    # Phase 3: compute metrics for each file x eval_variant
    rows = []
    errors = 0
    total_deduped_rows = 0
    for meta in deduped:
        try:
            df = pd.read_csv(meta['path'])
        except Exception as e:
            print(f"  [ERROR] Loading {meta['filename']}: {e}", file=sys.stderr)
            errors += 1
            continue

        # Deduplicate rows with identical identity keys (same Q/A scored
        # under different generation strategies).
        #
        # For hypernym: also apply manually curated conflict resolution:
        #   - HYPERNYM_REMOVE_PAIRS: drop all rows for these (noun1, noun2)
        #   - HYPERNYM_CONFLICT_KEEPS: for label conflicts, keep the row
        #     matching the curated label
        #   - All other dups: keep first occurrence
        n_before = len(df)
        if 'noun1' in df.columns and 'noun2' in df.columns:
            # Remove pairs flagged for removal
            if HYPERNYM_REMOVE_PAIRS:
                mask = df.apply(lambda r: (r['noun1'], r['noun2']) in HYPERNYM_REMOVE_PAIRS, axis=1)
                df = df[~mask]
            # Resolve label conflicts using curated keeps
            if HYPERNYM_CONFLICT_KEEPS:
                resolved = []
                for (n1, n2), grp in df.groupby(['noun1', 'noun2']):
                    if len(grp) <= 1:
                        resolved.append(grp)
                        continue
                    keep_label = HYPERNYM_CONFLICT_KEEPS.get((n1, n2))
                    if keep_label is not None:
                        kept = grp[grp['gpt4_ground_truth'].str.strip().str.lower() == keep_label]
                        resolved.append(kept.head(1) if len(kept) > 0 else grp.head(1))
                    else:
                        resolved.append(grp.head(1))
                df = pd.concat(resolved, ignore_index=True)
            else:
                df = df.drop_duplicates(subset=['noun1', 'noun2'], keep='first')
        elif 'question' in df.columns and 'answer' in df.columns:
            df = df.drop_duplicates(subset=['question', 'answer'], keep='first')
        elif 'prompt' in df.columns and 'response' in df.columns:
            df = df.drop_duplicates(subset=['prompt', 'response'], keep='first')
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            total_deduped_rows += n_dropped

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

        # Read model_path from CSV if available (new format), else None
        model_path = None
        if 'model_path' in df.columns:
            # All rows have the same model_path; take the first non-empty one
            mp_vals = df['model_path'].dropna().unique()
            if len(mp_vals) > 0:
                model_path = str(mp_vals[0])

        # Derive canonical model names
        local_model_name, hf_model_name = derive_model_names(
            model_path, meta['model'], meta['training_config'], meta['finetuned']
        )

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
                'hf_model_name': hf_model_name,
                'local_model_name': local_model_name,
                'task': meta['task'],
                'split': meta['split'],
                'self_tc': meta['self_tc'],
                'neg_tc': meta['neg_tc'],
                'gpt2_tc': meta['gpt2_tc'],
                'basetyp_tc': meta['basetyp_tc'],
                'include_eos': meta['include_eos'],
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

    if total_deduped_rows:
        print(f"Dedup (within-file): dropped {total_deduped_rows} duplicate rows across all score files")
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
                        help='Output CSV path (default: output-metrics/summary_scores.csv)')
    parser.add_argument('--outputs-dir', default=str(DEFAULT_OUTPUTS_DIR),
                        help='Outputs directory (default: outputs/)')
    parser.add_argument('--file-pattern', default='scores_*.csv',
                        help='Glob pattern for score files (default: scores_*.csv)')
    parser.add_argument('--upload-hf', action='store_true', default=False,
                        help='Upload summary to HuggingFace after computing')
    parser.add_argument('--incremental', action='store_true', default=False,
                        help='Only process new files not already in HF summary (implies --upload-hf)')
    parser.add_argument('--hf-org', default='TAUR-dev',
                        help='HuggingFace org (default: TAUR-dev)')
    parser.add_argument('--hf-dataset', default=DEFAULT_HF_DATASET,
                        help=f'HuggingFace dataset name (default: {DEFAULT_HF_DATASET})')
    parser.add_argument('--model-filter', default=None,
                        help='Only include files whose model name contains this substring (e.g. v6)')
    parser.add_argument('--epoch-filter', default=None,
                        help='For finetuned models, only include this epoch (e.g. epoch2). Base models always included.')
    args = parser.parse_args()

    if args.incremental:
        args.upload_hf = True

    # Incremental mode: pull existing, skip already-processed files
    existing = pd.DataFrame()
    existing_filenames = None
    if args.incremental:
        existing = pull_existing_summary(args.hf_org, args.hf_dataset)
        if not existing.empty and 'filename' in existing.columns:
            existing_filenames = set(existing['filename'].unique())
            print(f"Will skip {len(existing_filenames)} already-summarized files")
        # Backfill hf_model_name / local_model_name if missing
        if not existing.empty and 'hf_model_name' not in existing.columns:
            print("Backfilling hf_model_name and local_model_name on existing rows...")
            names = existing.apply(
                lambda r: derive_model_names(
                    None, r['model'], r.get('training_config', ''), r.get('finetuned', False)
                ), axis=1, result_type='expand'
            )
            existing['local_model_name'] = names[0]
            existing['hf_model_name'] = names[1]

    summary = discover_and_summarize(args.outputs_dir, existing_filenames, args.file_pattern,
                                     args.model_filter, args.epoch_filter)

    # Merge with existing if incremental
    if args.incremental and not existing.empty and not summary.empty:
        summary = pd.concat([existing, summary], ignore_index=True)
        summary = summary.sort_values(
            ['model', 'task', 'split', 'eval_variant']
        ).reset_index(drop=True)
        print(f"Merged: {len(existing)} existing + {len(summary) - len(existing)} new = {len(summary)} total rows")
    elif args.incremental and not existing.empty and summary.empty:
        print("No new files to process. Summary is up to date.")
        return
    elif summary.empty:
        print("No scores files found or no metrics computed.", file=sys.stderr)
        sys.exit(1)

    # Save locally
    summaries_dir = DEFAULT_SUMMARIES_DIR
    summaries_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output or str(summaries_dir / 'summary_scores.csv')
    summary.to_csv(output_path, index=False)
    print(f"\nSummary: {len(summary)} rows written to {output_path}")

    # Upload to HF
    if args.upload_hf:
        push_summary_to_hf(summary, args.hf_org, args.hf_dataset)

    # Quick overview
    print(f"\nModels ({len(summary['model'].unique())}): {sorted(summary['model'].unique())}")
    print(f"Tasks ({len(summary['task'].unique())}): {sorted(summary['task'].unique())[:20]}...")
    print(f"Splits: {sorted(summary['split'].unique())}")
    print(f"Eval variants: {sorted(summary['eval_variant'].unique())}")
    print(f"Self-TC rows: {summary['self_tc'].sum()}")
    print(f"Neg-TC rows: {summary['neg_tc'].sum()}")
    print(f"GPT2-TC rows: {summary['gpt2_tc'].sum()}")
    print(f"BaseTyp-TC rows: {summary['basetyp_tc'].sum()}")
    print(f"Include-EOS rows: {summary['include_eos'].sum()}")
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
