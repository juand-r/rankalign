#!/usr/bin/env python3
"""
General-purpose visualization dashboard for evaluation scores.

Run with:
    cd /datastor1/jdr/gv-gap/rankalign/scripts
    source ~/venvs/venv_lexcons/bin/activate
    PORT=8889 python dashboard_viz.py

Then access via SSH port forwarding:
    ssh -L 8889:localhost:8889 <your-host>
    
Open in browser: http://localhost:8889
"""

import os
import re
import json
import fnmatch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =============================================================================
# CONFIGURATION
# =============================================================================

PORT = int(os.environ.get('PORT', 8889))
CONFIG_DIR = Path(__file__).parent.parent / 'config'
CONFIG_FILE = CONFIG_DIR / 'dashboard_config.json'
DEFAULT_OUTPUTS_DIR = Path(__file__).parent.parent / 'outputs'

# Default configuration template
DEFAULT_CONFIG = {
    'outputs_dir': str(DEFAULT_OUTPUTS_DIR),
    'file_pattern': 'scores_*.csv',
    'task_pattern': r'hypernym-([a-zA-Z]+)',  # regex with group for dataset name
    'split_patterns': {
        'train': '_train_',
        'test': '_test_'
    },
    'label_column': 'gpt4_ground_truth',
    'label_map': {'yes': 1, 'no': 0},  # None if already numeric
    'gen_score_col': 'gen_score',
    'val_score_col': 'val_score',
    'model_detection': {
        'base_pattern': r'^scores_v6-google_gemma-2-2b_',  # base model: scores_v6-google_gemma-2-2b_...
        'finetuned_pattern': r'^scores_v6-google_gemma-2-2b-delta',  # finetuned: scores_v6-google_gemma-2-2b-delta...
        'finetuned_marker': 'delta',  # legacy marker (used with finetuned_pattern now)
        'direction_patterns': {
            'd2g': '_d2g_',
            'g2d': '_g2d_'
        },
        # Display labels for directions (used in heatmap titles, bar plots, etc.)
        'direction_display': {
            'd2g': 'V2G',
            'g2d': 'G2V'
        },
        'variant_patterns': {
            'typcorr_lenorm': '_typcorr_lenorm_full-completion',
            'tc-online_lenorm': '_tc-online_lenorm_full-completion',
            'typcorr': '_typcorr_full-completion',
            'tc-online': '_tc-online_full-completion',
            'lenorm': '_lenorm_full-completion',
            'vanilla': '_full-completion_'
        }
    },
    # Union models: trained on combined/union task, evaluated on individual tasks
    # Set to empty dict {} to disable union model detection
    'union_models': {
        'training_task': 'hypernym-concat-bananas-to-dogs',  # The training task name for union models
        'eval_task_pattern': r'force-same-x_(hypernym-[a-zA-Z]+)_',  # Pattern to extract eval task
        'row_prefix': 'Union'  # Prefix for row labels (e.g., "Union", "Union+tc")
    },
    'aggregation_groups': {
        'All Hypernym': 'hypernym-*'  # pattern-based: aggregate all matching tasks
    },
    'metrics': ['Val Acc', 'Val ROC', 'Gen ROC', 'Correlation', 'Corr-Pos', 'Corr-Neg'],
    'eval_columns': {
        'raw': 'gen_score',
        'tc': 'gen_score_typcorr',
        'lenorm': 'gen_score_lenorm',
        'tc+lenorm': 'gen_score_typcorr_lenorm'
    }
}

# Internal defaults (not configurable)
DEFAULT_VARIANT_DISPLAY = {
    'typcorr_lenorm': '+tc+lenorm',
    'tc-online_lenorm': '+tco+lenorm',
    'typcorr': '+tc',
    'tc-online': '+tco',
    'lenorm': '+lenorm',
    'vanilla': ''  # Empty string = use direction label (V2G/G2V)
}

# Variants that support vallogodds suffix
VALLOGDODS_VARIANTS = {'tc-online', 'lenorm', 'vanilla', 'tc-online_lenorm'}

DEFAULT_ROW_ORDER = ['Base', 'SFT', 'SFT+vallogodds', 'Pref only', 'Pref only+vallogodds', 'Union+tc', 'Union', 'V2G', 'G2V', '+tc', '+tco', '+lenorm', '+tc+lenorm', '+tco+lenorm', '+vallogodds', '+tco+vallogodds', '+lenorm+vallogodds', '+tco+lenorm+vallogodds']

# Rows to hide from heatmaps and barplots
#HIDDEN_ROWS = {'+tco', 'V2G'}
HIDDEN_ROWS = set()

# Visualization colors for positive/negative classes
POS_CLASS_COLOR = 'orangered'
NEG_CLASS_COLOR = 'blue'
POS_OUTLIER_COLOR = 'red'      # For outlier X markers (positive class)
NEG_OUTLIER_COLOR = 'purple'   # For outlier X markers (negative class)


# =============================================================================
# CONFIG MANAGEMENT
# =============================================================================

def load_config_from_file():
    """Load configuration from JSON file."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            return config, None
        except Exception as e:
            return None, f"Error loading config: {e}"
    return None, f"Config file not found: {CONFIG_FILE}"


def save_config_to_file(config):
    """Save configuration to JSON file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def auto_detect_config(outputs_dir=None):
    """Auto-detect configuration from scores files in outputs directory."""
    if outputs_dir is None:
        outputs_dir = DEFAULT_OUTPUTS_DIR
    else:
        outputs_dir = Path(outputs_dir)
    
    config = DEFAULT_CONFIG.copy()
    config['outputs_dir'] = str(outputs_dir)
    
    # Find all scores CSV files
    csv_files = list(outputs_dir.glob('scores_*.csv'))
    if not csv_files:
        return config, "No scores_*.csv files found in outputs directory"
    
    # Analyze filenames to detect patterns
    filenames = [f.stem for f in csv_files]
    
    # Detect tasks (look for common patterns like hypernym-X, ifeval-Y, trivia-qa, etc.)
    tasks_found = set()
    task_patterns_found = {}
    
    # Try common task patterns
    common_patterns = [
        # Hypernym-style tasks with a dataset suffix (e.g., hypernym-bananas)
        (r'hypernym-([a-zA-Z]+)', 'hypernym'),
        # IFEval-style per-prompt tasks, e.g. ifeval-prompt_1, ifeval-prompt_2
        # We capture the portion after "ifeval-" so that the dataset name
        # (e.g., "prompt_1") can be reconstructed consistently.
        (r'ifeval-prompt_([0-9]+)', 'ifeval'),
        # Other single-name tasks without a dataset suffix
        (r'trivia-qa', 'trivia-qa'),
        (r'swords', 'swords'),
        (r'lambada', 'lambada'),
        (r'collie', 'collie'),
    ]
    
    for pattern, task_type in common_patterns:
        for fname in filenames:
            match = re.search(pattern, fname)
            if match:
                if match.groups():
                    tasks_found.add(f"{task_type}-{match.group(1)}")
                else:
                    tasks_found.add(task_type)
                task_patterns_found[task_type] = pattern
    
    # If hypernym or ifeval tasks are found, prefer a pattern with a capture group
    # so that we can extract the dataset / prompt identifier from filenames.
    if any(t.startswith('hypernym-') for t in tasks_found):
        config['task_pattern'] = r'hypernym-([a-zA-Z]+)'
    elif any(t.startswith('ifeval-') for t in tasks_found):
        # Matches e.g. "ifeval-prompt_1" in filenames like:
        # scores_gemma-2-9b-it_ifeval-prompt_1_train_log-odds_...
        config['task_pattern'] = r'ifeval-prompt_([0-9]+)'
    
    # Detect splits
    has_train = any('_train_' in f for f in filenames)
    has_test = any('_test_' in f for f in filenames)
    config['split_patterns'] = {}
    if has_train:
        config['split_patterns']['train'] = '_train_'
    if has_test:
        config['split_patterns']['test'] = '_test_'
    
    # Detect model variants
    variants_found = set()
    
    # Check for base models (not finetuned).
    base_candidates = [
        r'^scores_gemma-2-9b-it_',
        r'^scores_gemma-2-2b_',
    ]
    detected_base_pattern = next(
        (pat for pat in base_candidates if any(re.match(pat, f) and 'delta' not in f for f in filenames)),
        None
    )
    has_base = detected_base_pattern is not None
    
    finetuned_candidates = [
        r'^scores_v6-google_gemma-2-9b-it-delta',
        r'^scores_v6-google_gemma-2-2b-delta',
    ]
    detected_finetuned_pattern = next(
        (pat for pat in finetuned_candidates if any(re.match(pat, f) for f in filenames)),
        None
    )
    has_finetuned = detected_finetuned_pattern is not None
    
    # Check for directions
    has_d2g = any('_d2g_' in f for f in filenames)
    has_g2d = any('_g2d_' in f for f in filenames)
    
    # Check for training variants - look for all combinations
    has_vanilla = any('_full-completion_' in f and '_typcorr' not in f and '_tc-online' not in f and '_lenorm' not in f for f in filenames)
    has_typcorr_only = any('_typcorr_full-completion' in f and '_lenorm' not in f and '_tc-online' not in f for f in filenames)
    has_lenorm_only = any('_lenorm_full-completion' in f and '_typcorr' not in f and '_tc-online' not in f for f in filenames)
    has_typcorr_lenorm = any('_typcorr_lenorm_full-completion' in f for f in filenames)
    has_tc_online_only = any('_tc-online_full-completion' in f and '_lenorm' not in f for f in filenames)
    has_tc_online_lenorm = any('_tc-online_lenorm_full-completion' in f for f in filenames)
    
    # Update model detection config based on what we found
    config['model_detection'] = {
        'base_pattern': detected_base_pattern if has_base else None,
        'finetuned_pattern': detected_finetuned_pattern if has_finetuned else None,
        'finetuned_marker': 'delta' if not has_finetuned else None,
        'direction_patterns': {},
        'direction_display': {},
        'variant_patterns': {}
    }
    
    if has_d2g:
        config['model_detection']['direction_patterns']['d2g'] = '_d2g_'
        config['model_detection']['direction_display']['d2g'] = 'V2G'
    if has_g2d:
        config['model_detection']['direction_patterns']['g2d'] = '_g2d_'
        config['model_detection']['direction_display']['g2d'] = 'G2V'
    
    # Add all variant patterns that exist
    # Note: Order matters for detection - more specific patterns first
    if has_typcorr_lenorm:
        config['model_detection']['variant_patterns']['typcorr_lenorm'] = '_typcorr_lenorm_full-completion'
    if has_tc_online_lenorm:
        config['model_detection']['variant_patterns']['tc-online_lenorm'] = '_tc-online_lenorm_full-completion'
    if has_typcorr_only:
        config['model_detection']['variant_patterns']['typcorr'] = '_typcorr_full-completion'
    if has_tc_online_only:
        config['model_detection']['variant_patterns']['tc-online'] = '_tc-online_full-completion'
    if has_lenorm_only:
        config['model_detection']['variant_patterns']['lenorm'] = '_lenorm_full-completion'
    if has_vanilla:
        config['model_detection']['variant_patterns']['vanilla'] = '_full-completion_'
    
    # Detect union models (trained on combined task, evaluated on individual tasks)
    # Look for files with "concat" or combined task names and "force-same-x" eval pattern
    has_union = any('hypernym-concat-bananas-to-dogs' in f and 'force-same-x_' in f for f in filenames)
    if has_union:
        config['union_models'] = {
            'training_task': 'hypernym-concat-bananas-to-dogs',
            'eval_task_pattern': r'force-same-x_(hypernym-[a-zA-Z]+)_',
            'row_prefix': 'Union'
        }
    else:
        config['union_models'] = {}
    
    # Build aggregation groups based on tasks found
    config['aggregation_groups'] = {}
    
    # Group by task type prefix (e.g., "hypernym" from "hypernym-bananas")
    # Only create aggregation groups for task types that have subtasks (contain a hyphen)
    # and where there are multiple subtasks to aggregate
    task_prefixes = {}
    for task in tasks_found:
        if '-' in task:
            # Split only on first hyphen to handle cases like "trivia-qa" correctly
            prefix = task.split('-', 1)[0]
            if prefix not in task_prefixes:
                task_prefixes[prefix] = []
            task_prefixes[prefix].append(task)
    
    for prefix, matching_tasks in task_prefixes.items():
        if len(matching_tasks) > 1:
            config['aggregation_groups'][f'All {prefix.capitalize()}'] = f"{prefix}-*"
    
    # Read one CSV to detect label column
    label_detection_warning = None
    try:
        sample_df = pd.read_csv(csv_files[0], nrows=5)
        # Check for common label columns
        for col in ['gpt4_ground_truth', 'label', 'correct', 'taxonomic']:
            if col in sample_df.columns:
                config['label_column'] = col
                # Check if it's string or numeric
                if sample_df[col].dtype == 'object':
                    unique_vals = sample_df[col].str.lower().unique()
                    if 'yes' in unique_vals or 'no' in unique_vals:
                        config['label_map'] = {'yes': 1, 'no': 0}
                    else:
                        config['label_map'] = None
                else:
                    config['label_map'] = None
                break
    except Exception as e:
        label_detection_warning = f" (Warning: could not auto-detect label column from CSV: {e})"
    
    info_msg = f"Auto-detected from {len(csv_files)} files: {len(tasks_found)} tasks, splits: {list(config['split_patterns'].keys())}"
    if label_detection_warning:
        info_msg += label_detection_warning
    return config, info_msg


# =============================================================================
# DATA LOADING AND PROCESSING
# =============================================================================

def discover_scores_files(config):
    """Discover all scores CSV files based on config."""
    outputs_dir = Path(config['outputs_dir'])
    file_pattern = config.get('file_pattern', 'scores_*.csv')
    
    files = []
    for csv_file in sorted(outputs_dir.glob(file_pattern)):
        file_info = parse_filename(csv_file, config)
        if file_info:
            files.append(file_info)
    
    return files


def parse_filename(csv_file, config):
    """Parse filename to extract task, split, model info based on config.
    
    For Union models:
    scores_v5-google_..._hypernym-concat-bananas-to-dogs-v2-all_d2g_..._force-same-x_hypernym-diapers_test_...
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                          ^^^^^^^^^^^^^^^
                        training task (union)                                       eval task
    
    For regular models, we extract both training and eval tasks and only include
    if they match (to exclude cross-task evaluations).
    """
    name = csv_file.stem
    task_pattern = config.get('task_pattern', r'hypernym-([a-zA-Z]+)')
    
    # Check for union model first
    union_config = config.get('union_models', {})
    is_union = False
    
    if union_config:
        union_training_task = union_config.get('training_task', '')
        eval_pattern = union_config.get('eval_task_pattern', '')
        
        if union_training_task and union_training_task in name:
            # This is a union model - extract the evaluated task
            if eval_pattern:
                eval_match = re.search(eval_pattern, name)
                if eval_match:
                    is_union = True
                    if eval_match.groups():
                        # Pattern has capture group
                        task = eval_match.group(1)
                        # Try to get full task name (e.g., hypernym-diapers)
                        full_match = re.search(r'(hypernym-[a-zA-Z]+)', eval_match.group(0))
                        if full_match:
                            task = full_match.group(1)
                        dataset = task.split('-')[-1] if '-' in task else task
                    else:
                        task = eval_match.group(0)
                        dataset = task
    
    # If not union, extract both training and eval tasks and verify they match
    if not is_union:
        # Find all matches of the task pattern
        all_matches = list(re.finditer(task_pattern, name))
        
        if len(all_matches) >= 2:
            # Multiple matches - first is training task, last is eval task
            training_match = all_matches[0]
            eval_match = all_matches[-1]
            
            if training_match.groups() and eval_match.groups():
                training_task = training_match.group(1)
                eval_task = eval_match.group(1)
                
                if training_task == eval_task:
                    # Tasks match - use the eval task
                    base_task = task_pattern.split('(')[0].rstrip('-').rstrip('_')
                    if not base_task:
                        base_task = 'task'
                    dataset = eval_task
                    task = f"{base_task}-{dataset}" if base_task else dataset
                else:
                    # Training and eval tasks don't match - skip this file
                    return None
            else:
                # No capture group - use full match
                if training_match.group(0) == eval_match.group(0):
                    task = eval_match.group(0)
                    dataset = task
                else:
                    return None
        elif len(all_matches) == 1:
            # Only one match - use it (likely base model or simple case)
            match = all_matches[0]
            if match.groups():
                base_task = task_pattern.split('(')[0].rstrip('-').rstrip('_')
                if not base_task:
                    base_task = 'task'
                dataset = match.group(1)
                task = f"{base_task}-{dataset}" if base_task else dataset
            else:
                task = match.group(0)
                dataset = task
        else:
            task = 'unknown'
            dataset = 'unknown'
    
    # Extract split
    split = 'unknown'
    split_patterns = config.get('split_patterns', {})
    for split_name, pattern in split_patterns.items():
        if pattern in name:
            split = split_name
            break
    
    # Extract model info (direction, training variant, model type)
    model_detection = config.get('model_detection', {})
    direction, training_variant, model_type = determine_model_info(
        name, model_detection, is_union=is_union, union_config=union_config
    )
    
    # Skip files that don't match expected format
    if direction is None:
        return None
    
    display_name = f'{task} | {split} | {model_type}'
    
    return {
        'path': str(csv_file),
        'display': display_name,
        'task': task,
        'dataset': dataset,
        'split': split,
        'direction': direction,  # 'd2g', 'g2d', or 'base'
        'training_variant': training_variant,  # 'Base', 'V2G', '+tc', 'Union', etc.
        'model_type': model_type,  # Combined label for dropdown
        'is_union': is_union  # Flag for union models
    }


def determine_model_info(filename, model_detection, is_union=False, union_config=None):
    """Determine model direction and training variant from filename.
    
    Args:
        filename: The filename to parse
        model_detection: Config dict for model detection patterns
        is_union: Whether this is a union/combined model
        union_config: Config dict for union models (required if is_union=True)
    
    Returns:
        tuple: (direction, training_variant, model_type)
            - direction: 'd2g', 'g2d', or 'base'
            - training_variant: 'Base', 'V2G'/'G2V' (vanilla), '+tc', '+tco', '+lenorm',
                               or 'Union', 'Union+tc', etc. for union models
            - model_type: combined label like 'V2G+tc' for dropdown display
    """
    base_pattern = model_detection.get('base_pattern')
    finetuned_pattern = model_detection.get('finetuned_pattern')
    finetuned_marker = model_detection.get('finetuned_marker')  # legacy fallback
    direction_patterns = model_detection.get('direction_patterns', {})
    direction_display = model_detection.get('direction_display', {'d2g': 'V2G', 'g2d': 'G2V'})
    # Use internal default for variant_display (not configurable)
    variant_display = DEFAULT_VARIANT_DISPLAY

    # Check if base model vs finetuned
    # Base: scores_v6-google_gemma-2-2b_hypernym-bananas_... (no "delta")
    # Finetuned: scores_v6-google_gemma-2-2b-delta0.15-epoch2_hypernym-bananas-all_d2g_... (has "delta")
    
    # Finetuned models: use finetuned_pattern if available, else fall back to finetuned_marker
    if finetuned_pattern:
        is_finetuned = bool(re.match(finetuned_pattern, filename))
    else:
        is_finetuned = finetuned_marker and finetuned_marker in filename
    
    # Base model: matches pattern AND not finetuned
    is_base = base_pattern and re.match(base_pattern, filename) and not is_finetuned

    pref_match = re.search(r'(?:^|[_-])pref(?P<weight>\d+(?:\.\d+)?)', filename)
    pref_weight = None
    if pref_match:
        try:
            pref_weight = float(pref_match.group('weight'))
        except ValueError:
            pref_weight = None
    no_pref = pref_weight is not None and pref_weight == 0.0
    
    # Parse nll weights (nllv = validator, nllg = generator)
    nllv_match = re.search(r'nllv(?P<weight>\d+(?:\.\d+)?)', filename)
    nllg_match = re.search(r'nllg(?P<weight>\d+(?:\.\d+)?)', filename)
    nllv_weight = float(nllv_match.group('weight')) if nllv_match else None
    nllg_weight = float(nllg_match.group('weight')) if nllg_match else None
    
    # "Pref only" = pref weight is 1.0 and both nll weights are 0.0
    pref_only = (pref_weight is not None and pref_weight == 1.0 and
                 nllv_weight is not None and nllv_weight == 0.0 and
                 nllg_weight is not None and nllg_weight == 0.0)
    
    # Check for vallogodds suffix (used for SFT and Pref only variants)
    has_vallogodds = '_vallogodds' in filename
    
    if is_base:
        return 'base', 'Base', 'Base'

    if pref_only:
        if has_vallogodds:
            return 'base', 'Pref only+vallogodds', 'Pref only+vallogodds'
        return 'base', 'Pref only', 'Pref only'

    if no_pref:
        if has_vallogodds:
            return 'base', 'SFT+vallogodds', 'SFT+vallogodds'
        return 'base', 'SFT', 'SFT'
    
    if not is_finetuned:
        # Doesn't match our expected format - skip it
        return None, None, None
    
    # Determine direction using config
    direction = 'unknown'
    for dir_name, pattern in direction_patterns.items():
        if pattern in filename:
            direction = dir_name
            break
    
    dir_label = direction_display.get(direction, direction.upper())
    
    # First, detect what variant the file actually is (using complete pattern set)
    # Then check if that variant is in the config's variant_patterns
    variant_patterns = model_detection.get('variant_patterns', {})
    
    # Complete pattern set for detection (all known variants)
    all_variant_patterns = {
        'typcorr_lenorm': '_typcorr_lenorm_full-completion',
        'tc-online_lenorm': '_tc-online_lenorm_full-completion',
        'typcorr': '_typcorr_full-completion',
        'tc-online': '_tc-online_full-completion',
        'lenorm': '_lenorm_full-completion',
        'vanilla': '_full-completion_'
    }
    
    # Detect actual variant from filename (using complete pattern set)
    sorted_all_patterns = sorted(all_variant_patterns.items(), key=lambda x: -len(x[1]))
    detected_variant = None
    for variant_name, pattern in sorted_all_patterns:
        if pattern in filename:
            detected_variant = variant_name
            break  # Use first (most specific) match
    
    # If no variant detected, skip the file (don't assume it's vanilla)
    if not detected_variant:
        return None, None, None
    
    # Only proceed if detected variant is in config's variant_patterns
    # This ensures config controls what's shown
    if detected_variant not in variant_patterns:
        # File has a variant that's not in config - skip it
        return None, None, None
    
    # Use detected variant (which we know is in config)
    matched_variant = detected_variant
    
    # Check for vallogodds suffix (only for specific variants)
    has_vallogodds = False
    if matched_variant in VALLOGDODS_VARIANTS:
        # Check if _vallogodds appears in filename (after nllg1.0_)
        if '_vallogodds' in filename:
            has_vallogodds = True
    
    # Map detected variant to display label using variant_display
    suffix = ''
    if matched_variant:
        suffix = variant_display.get(matched_variant, '')
        if suffix:
            training_variant = suffix
        else:
            # Vanilla or variant not in variant_display - use direction label
            training_variant = dir_label
    else:
        # No variant detected - treat as vanilla, use direction label
        training_variant = dir_label
    
    # Append vallogodds suffix if detected
    if has_vallogodds:
        if not suffix:
            # Vanilla with vallogodds: use direction-agnostic '+vallogodds' row
            training_variant = '+vallogodds'
            suffix = '+vallogodds'
        else:
            # Variant with vallogodds: append to existing suffix
            training_variant = f'{training_variant}+vallogodds'
            suffix = f'{suffix}+vallogodds'
    
    # Handle union models - override training_variant with prefix
    if is_union and union_config:
        row_prefix = union_config.get('row_prefix', 'Union')
        if suffix:
            # Has variant (e.g., +tc) -> "Union+tc"
            training_variant = f'{row_prefix}{suffix}'
        else:
            # Vanilla union model -> "Union"
            training_variant = row_prefix
        # Model type for dropdown also uses prefix
        model_type = training_variant
    else:
        # Build model_type for dropdown display (includes direction)
        model_type = f'{dir_label}{suffix}' if suffix else dir_label
    
    return direction, training_variant, model_type


def determine_model_type(filename, model_detection):
    """Determine model type from filename (for backward compatibility)."""
    _, _, model_type = determine_model_info(filename, model_detection)
    return model_type


def load_scores_data(csv_path, config):
    """Load scores data from CSV file based on config."""
    df = pd.read_csv(csv_path)
    
    # Convert ground truth to binary label
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
            # Try common mappings
            df['label'] = gt_col.str.strip().str.lower().map({'yes': 1, 'no': 0, 'true': 1, 'false': 0})
            df['label'] = df['label'].fillna(0).astype(int)
    
    return df


def compute_metrics(gen_scores, val_scores, labels, metric_type='log-odds'):
    """Compute all metrics for a set of scores."""
    gen_scores_np = np.array(gen_scores)
    val_scores_np = np.array(val_scores)
    labels_np = np.array(labels)
    
    # Handle NaN values
    valid_mask = ~(np.isnan(gen_scores_np) | np.isnan(val_scores_np))
    if valid_mask.sum() < 2:
        return {'corr': np.nan, 'corr_pos': np.nan, 'corr_neg': np.nan, 
                'acc': np.nan, 'val_roc': np.nan, 'gen_roc': np.nan}
    
    gen_valid = gen_scores_np[valid_mask]
    val_valid = val_scores_np[valid_mask]
    labels_valid = labels_np[valid_mask]
    
    pos_mask = labels_valid == 1
    neg_mask = labels_valid == 0
    
    # Correlations
    try:
        corr_all, _ = pearsonr(gen_valid, val_valid)
    except:
        corr_all = np.nan
    
    try:
        if pos_mask.sum() > 1:
            corr_pos, _ = pearsonr(gen_valid[pos_mask], val_valid[pos_mask])
        else:
            corr_pos = np.nan
    except:
        corr_pos = np.nan
    
    try:
        if neg_mask.sum() > 1:
            corr_neg, _ = pearsonr(gen_valid[neg_mask], val_valid[neg_mask])
        else:
            corr_neg = np.nan
    except:
        corr_neg = np.nan
    
    # Threshold
    threshold = 0 if metric_type == 'log-odds' else np.log(0.5)
    
    # Accuracy
    preds = (val_valid > threshold).astype(int)
    acc = accuracy_score(labels_valid, preds)
    
    # ROC AUC
    try:
        val_roc = roc_auc_score(labels_valid, val_valid)
    except:
        val_roc = np.nan
    
    try:
        gen_roc = roc_auc_score(labels_valid, gen_valid)
    except:
        gen_roc = np.nan
    
    return {
        'corr': corr_all, 'corr_pos': corr_pos, 'corr_neg': corr_neg,
        'acc': acc, 'val_roc': val_roc, 'gen_roc': gen_roc
    }


def get_model_row_label(model_type):
    """Get display label for model row in heatmap."""
    return model_type


def get_model_rows(files_info):
    """Get unique model types for heatmap rows (for dropdown)."""
    model_types = set(f['model_type'] for f in files_info)
    
    # Sort with Base first, then alphabetically
    sorted_types = sorted(model_types, key=lambda x: (0 if x == 'Base' else 1, x))
    return sorted_types


def get_training_variant_rows(files_info, config=None):
    """Get unique training variants for heatmap rows (direction-agnostic).
    
    Args:
        files_info: List of file info dicts
        config: Optional config dict (unused, kept for compatibility)
    
    Returns rows like: ['Base', 'V2G', 'G2V', '+tc', '+tco', '+lenorm', '+tc+lenorm', '+tco+lenorm']
    where V2G/G2V represent vanilla finetuned models.
    """
    variants = set()
    for f in files_info:
        variants.add(f.get('training_variant', f['model_type']))
    
    # Use internal default for row order (not configurable)
    order = DEFAULT_ROW_ORDER
    
    # Sort: items in order list first (by their position), then others alphabetically
    def sort_key(x):
        if x in order:
            return (0, order.index(x))
        return (1, x)
    
    return sorted(variants, key=sort_key)


def expand_aggregation_pattern(pattern, all_tasks):
    """Expand aggregation pattern to list of tasks.
    
    Supports:
    - List: ["task1", "task2", ...]
    - Pattern: "prefix-*" matches all tasks starting with "prefix-"
    """
    if isinstance(pattern, list):
        return [t for t in pattern if t in all_tasks]
    elif isinstance(pattern, str) and '*' in pattern:
        # Wildcard pattern
        return [t for t in all_tasks if fnmatch.fnmatch(t, pattern)]
    elif isinstance(pattern, str):
        # Single task
        return [pattern] if pattern in all_tasks else []
    return []


# =============================================================================
# HEATMAP FUNCTIONS
# =============================================================================

def discover_heatmap_data_by_direction(task, split, files_info, config):
    """Discover and load heatmap data for a task and split, organized by direction.
    
    Returns:
        dict: {
            'd2g': {training_variant: {eval_col: metrics, ...}, ...},
            'g2d': {training_variant: {eval_col: metrics, ...}, ...},
            'base': {training_variant: {eval_col: metrics, ...}, ...}
        }
        list: training_variant_rows (shared across directions)
        list: eval_cols
    """
    eval_columns = config.get('eval_columns', DEFAULT_CONFIG['eval_columns'])
    
    training_variant_rows = get_training_variant_rows(files_info, config)
    eval_cols = list(eval_columns.keys())
    
    # Get direction keys from config
    direction_patterns = config.get('model_detection', {}).get('direction_patterns', {'d2g': '_d2g_', 'g2d': '_g2d_'})
    
    # Initialize data structure for each direction
    data = {dir_key: {row: {col: None for col in eval_cols} for row in training_variant_rows} 
            for dir_key in direction_patterns.keys()}
    data['base'] = {row: {col: None for col in eval_cols} for row in training_variant_rows}
    
    # Find matching files
    matching_files = [f for f in files_info if f['task'] == task and f['split'] == split]
    
    # Check which files are actual base models (match base_pattern)
    base_pattern = config.get('model_detection', {}).get('base_pattern')
    
    # Track which files are used for each row/column (for debugging)
    file_tracking = {}  # {(direction, training_variant, eval_col): filepath}
    
    for file_info in matching_files:
        direction = file_info.get('direction', 'base')
        training_variant = file_info.get('training_variant', file_info['model_type'])
        
        # Only store in 'base' direction if this is an actual base model file
        # (matches base_pattern or versioned base pattern), not a misclassified finetuned model
        filename = Path(file_info['path']).name
        finetuned_marker = config.get('model_detection', {}).get('finetuned_marker')
        
        is_actual_base = False
        if training_variant == 'Base':
            # Base model: matches pattern AND no delta
            if base_pattern and re.match(base_pattern, filename) and 'delta' not in filename:
                is_actual_base = True
        elif training_variant in ('SFT', 'SFT+vallogodds'):
            # No-pref base: allow pref0.0 runs even if they have delta
            if re.search(r'(?:^|[_-])pref0(?:\.0+)?', filename):
                is_actual_base = True
        elif training_variant in ('Pref only', 'Pref only+vallogodds'):
            # Pref-only: pref=1.0 and nll weights=0.0
            if re.search(r'pref1(?:\.0+)?', filename) and re.search(r'nllv0(?:\.0+)?', filename) and re.search(r'nllg0(?:\.0+)?', filename):
                is_actual_base = True
        
        try:
            df = load_scores_data(file_info['path'], config)
            metric_type = 'log-odds' if 'log-odds' in file_info['path'] else 'log-probs'
            
            for eval_col, gen_col in eval_columns.items():
                if gen_col in df.columns and 'val_score' in df.columns and 'label' in df.columns:
                    gen_scores = df[gen_col].values
                    val_scores = df['val_score'].values
                    labels = df['label'].values
                    metrics = compute_metrics(gen_scores, val_scores, labels, metric_type)
                    
                    # Store in the appropriate direction
                    if direction in data and training_variant in data[direction]:
                        # Only store in 'base' direction if this is an actual base model
                        if direction == 'base' and not is_actual_base:
                            # Skip storing misclassified finetuned models in base direction
                            continue
                        data[direction][training_variant][eval_col] = metrics
                        # Track which file was used
                        file_tracking[(direction, training_variant, eval_col)] = file_info['path']
        except Exception as e:
            print(f"Error loading {file_info['path']}: {e}")
            continue
    
    # Copy base model data to all directions (base model is shared)
    # Only copy from actual base models stored in data['base']['Base']
    for dir_key in direction_patterns.keys():
        for eval_col in eval_cols:
            if data['base'].get('Base', {}).get(eval_col) is not None:
                if dir_key in data:
                    data[dir_key]['Base'][eval_col] = data['base']['Base'][eval_col]
                    # Track that this was copied from base direction
                    if ('base', 'Base', eval_col) in file_tracking:
                        file_tracking[(dir_key, 'Base', eval_col)] = file_tracking[('base', 'Base', eval_col)]

            # Also copy "SFT" if present
            if data['base'].get('SFT', {}).get(eval_col) is not None:
                if dir_key in data:
                    # Ensure the SFT dict exists in this direction
                    if 'SFT' not in data[dir_key]:
                        data[dir_key]['SFT'] = {col: None for col in eval_cols}
                    data[dir_key]['SFT'][eval_col] = data['base']['SFT'][eval_col]
                    if ('base', 'SFT', eval_col) in file_tracking:
                        file_tracking[(dir_key, 'SFT', eval_col)] = file_tracking[('base', 'SFT', eval_col)]

            # Also copy "SFT+vallogodds" if present
            if data['base'].get('SFT+vallogodds', {}).get(eval_col) is not None:
                if dir_key in data:
                    if 'SFT+vallogodds' not in data[dir_key]:
                        data[dir_key]['SFT+vallogodds'] = {col: None for col in eval_cols}
                    data[dir_key]['SFT+vallogodds'][eval_col] = data['base']['SFT+vallogodds'][eval_col]
                    if ('base', 'SFT+vallogodds', eval_col) in file_tracking:
                        file_tracking[(dir_key, 'SFT+vallogodds', eval_col)] = file_tracking[('base', 'SFT+vallogodds', eval_col)]

            # Also copy "Pref only" if present
            if data['base'].get('Pref only', {}).get(eval_col) is not None:
                if dir_key in data:
                    # Ensure the Pref only dict exists in this direction
                    if 'Pref only' not in data[dir_key]:
                        data[dir_key]['Pref only'] = {col: None for col in eval_cols}
                    data[dir_key]['Pref only'][eval_col] = data['base']['Pref only'][eval_col]
                    if ('base', 'Pref only', eval_col) in file_tracking:
                        file_tracking[(dir_key, 'Pref only', eval_col)] = file_tracking[('base', 'Pref only', eval_col)]

            # Also copy "Pref only+vallogodds" if present
            if data['base'].get('Pref only+vallogodds', {}).get(eval_col) is not None:
                if dir_key in data:
                    if 'Pref only+vallogodds' not in data[dir_key]:
                        data[dir_key]['Pref only+vallogodds'] = {col: None for col in eval_cols}
                    data[dir_key]['Pref only+vallogodds'][eval_col] = data['base']['Pref only+vallogodds'][eval_col]
                    if ('base', 'Pref only+vallogodds', eval_col) in file_tracking:
                        file_tracking[(dir_key, 'Pref only+vallogodds', eval_col)] = file_tracking[('base', 'Pref only+vallogodds', eval_col)]
    
    # Write file tracking to log file for debugging
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = CONFIG_DIR / 'dashboard_file_tracking.log'
    from datetime import datetime
    with open(log_file, 'a') as f:
        f.write(f"\n=== File tracking for {task} / {split} ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===\n")
        for direction in sorted(set(d for d, _, _ in file_tracking.keys())):
            f.write(f"\n{direction.upper()} direction:\n")
            for training_variant in sorted(set(tv for d, tv, _ in file_tracking.keys() if d == direction)):
                f.write(f"  {training_variant}:\n")
                for eval_col in sorted(set(ec for d, tv, ec in file_tracking.keys() if d == direction and tv == training_variant)):
                    filepath = file_tracking.get((direction, training_variant, eval_col), 'NOT FOUND')
                    filename = Path(filepath).name if filepath != 'NOT FOUND' else 'NOT FOUND'
                    f.write(f"    {eval_col}: {filename}\n")
        f.write("\n")
    
    return data, training_variant_rows, eval_cols


def discover_heatmap_data(task, split, files_info, config):
    """Discover and load heatmap data for a task and split (legacy interface)."""
    eval_columns = config.get('eval_columns', DEFAULT_CONFIG['eval_columns'])
    
    model_rows = get_model_rows(files_info)
    eval_cols = list(eval_columns.keys())
    
    # Initialize data structure
    data = {model: {col: None for col in eval_cols} for model in model_rows}
    
    # Find matching files
    matching_files = [f for f in files_info if f['task'] == task and f['split'] == split]
    
    for file_info in matching_files:
        model_type = file_info['model_type']
        
        try:
            df = load_scores_data(file_info['path'], config)
            metric_type = 'log-odds' if 'log-odds' in file_info['path'] else 'log-probs'
            
            for eval_col, gen_col in eval_columns.items():
                if gen_col in df.columns and 'val_score' in df.columns and 'label' in df.columns:
                    gen_scores = df[gen_col].values
                    val_scores = df['val_score'].values
                    labels = df['label'].values
                    metrics = compute_metrics(gen_scores, val_scores, labels, metric_type)
                    data[model_type][eval_col] = metrics
        except Exception as e:
            print(f"Error loading {file_info['path']}: {e}")
            continue
    
    return data, model_rows, eval_cols


def create_heatmap_figure(heatmap_data, model_rows, eval_cols, title, metrics_list):
    """Create a heatmap figure."""
    fig = make_subplots(rows=1, cols=len(metrics_list), subplot_titles=metrics_list,
                        horizontal_spacing=0.03)
    
    metric_key_map = {
        'Val Acc': 'acc', 'Val ROC': 'val_roc', 'Gen ROC': 'gen_roc',
        'Correlation': 'corr', 'Corr-Pos': 'corr_pos', 'Corr-Neg': 'corr_neg'
    }
    
    for m_idx, metric in enumerate(metrics_list):
        metric_key = metric_key_map.get(metric, metric.lower())
        
        z = []
        text = []
        for row in model_rows:
            z_row = []
            text_row = []
            for col in eval_cols:
                metrics = heatmap_data.get(row, {}).get(col)
                if metrics is not None and not np.isnan(metrics.get(metric_key, np.nan)):
                    val = metrics[metric_key] * 100
                    z_row.append(val)
                    text_row.append(f'{val:.1f}')
                else:
                    z_row.append(None)
                    text_row.append('')
            z.append(z_row)
            text.append(text_row)
        
        fig.add_trace(
            go.Heatmap(
                z=z, x=eval_cols, y=model_rows,
                text=text, texttemplate='%{text}', textfont={'size': 10},
                colorscale='RdYlGn', zmin=0, zmax=100,
                showscale=(m_idx == len(metrics_list) - 1),
                hovertemplate='%{y} / %{x}: %{z:.1f}<extra></extra>'
            ),
            row=1, col=m_idx + 1
        )
    
    # Only show y-axis labels on the first heatmap
    for m_idx in range(len(metrics_list)):
        fig.update_yaxes(showticklabels=(m_idx == 0), row=1, col=m_idx + 1)
    
    fig.update_layout(
        title=title,
        height=250,  # Fixed height like hypernym_dashboard.py
        margin=dict(l=80, r=20, t=50, b=30),  # Same margin as hypernym_dashboard.py
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig


def create_direction_heatmap_figure(direction_data, training_rows, eval_cols, title, metrics_list, direction_label):
    """Create a heatmap figure for a specific direction (V2G or G2V).
    
    Args:
        direction_data: dict mapping training_variant -> {eval_col: metrics}
        training_rows: list of training variant labels (e.g., ['V2G', '+tc', '+lenorm', ...])
        eval_cols: list of eval column names
        title: figure title
        metrics_list: list of metrics to display
        direction_label: 'V2G' or 'G2V' for vanilla row labeling
    """
    fig = make_subplots(rows=1, cols=len(metrics_list), subplot_titles=metrics_list,
                        horizontal_spacing=0.03)
    
    metric_key_map = {
        'Val Acc': 'acc', 'Val ROC': 'val_roc', 'Gen ROC': 'gen_roc',
        'Correlation': 'corr', 'Corr-Pos': 'corr_pos', 'Corr-Neg': 'corr_neg'
    }
    
    # Include all relevant rows for this direction:
    # - Base model
    # - Vanilla for this direction (e.g., 'V2G' for d2g)
    # - Vanilla with vallogodds for this direction (e.g., 'V2G+vallogodds' for d2g)
    # - Training variants like +tc, +tco, +lenorm
    # - Training variants with vallogodds like +tco+vallogodds, +lenorm+vallogodds
    # - Union models (e.g., 'Union', 'Union+tc')
    # Show empty rows if no data - don't filter them out
    relevant_rows = []
    for row in training_rows:
        if row == 'Base':
            relevant_rows.append(row)
        elif row in ('SFT', 'SFT+vallogodds'):
            relevant_rows.append(row)
        elif row in ('Pref only', 'Pref only+vallogodds'):
            relevant_rows.append(row)
        elif row == direction_label:  # Vanilla for this direction (e.g., 'V2G' for d2g)
            relevant_rows.append(row)
        elif row.startswith('+'):  # Training variants like +tc, +tco, +lenorm, +vallogodds, +tco+vallogodds, etc.
            relevant_rows.append(row)
        elif row.startswith('Union'):  # Union models
            relevant_rows.append(row)
        # Skip other direction's vanilla (e.g., skip 'G2V' when direction_label is 'V2G')
    
    # Filter out hidden rows
    relevant_rows = [r for r in relevant_rows if r not in HIDDEN_ROWS]
    
    if not relevant_rows:
        # No relevant rows at all
        return None
    
    for m_idx, metric in enumerate(metrics_list):
        metric_key = metric_key_map.get(metric, metric.lower())
        
        z = []
        text = []
        for row in relevant_rows:
            z_row = []
            text_row = []
            for col in eval_cols:
                metrics = direction_data.get(row, {}).get(col)
                if metrics is not None and not np.isnan(metrics.get(metric_key, np.nan)):
                    val = metrics[metric_key] * 100
                    z_row.append(val)
                    text_row.append(f'{val:.1f}')
                else:
                    z_row.append(None)
                    text_row.append('')
            z.append(z_row)
            text.append(text_row)
        
        fig.add_trace(
            go.Heatmap(
                z=z, x=eval_cols, y=relevant_rows,
                text=text, texttemplate='%{text}', textfont={'size': 10},
                colorscale='RdYlGn', zmin=0, zmax=100,
                showscale=(m_idx == len(metrics_list) - 1),
                hovertemplate='%{y} / %{x}: %{z:.1f}<extra></extra>'
            ),
            row=1, col=m_idx + 1
        )
    
    # Only show y-axis labels on the first heatmap
    for m_idx in range(len(metrics_list)):
        fig.update_yaxes(showticklabels=(m_idx == 0), row=1, col=m_idx + 1)
    
    fig.update_layout(
        title=title,
        height=250,  # Fixed height like hypernym_dashboard.py
        margin=dict(l=80, r=20, t=50, b=30),  # Same margin as hypernym_dashboard.py
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig


def create_aggregated_heatmap(all_heatmap_data, tasks, model_rows, eval_cols, title, metrics_list):
    """Create aggregated heatmap showing mean across tasks."""
    fig = make_subplots(rows=1, cols=len(metrics_list), subplot_titles=metrics_list,
                        horizontal_spacing=0.03)
    
    metric_key_map = {
        'Val Acc': 'acc', 'Val ROC': 'val_roc', 'Gen ROC': 'gen_roc',
        'Correlation': 'corr', 'Corr-Pos': 'corr_pos', 'Corr-Neg': 'corr_neg'
    }
    
    for m_idx, metric in enumerate(metrics_list):
        metric_key = metric_key_map.get(metric, metric.lower())
        
        z = []
        text = []
        for row in model_rows:
            z_row = []
            text_row = []
            for col in eval_cols:
                values = []
                for task in tasks:
                    if task in all_heatmap_data:
                        metrics = all_heatmap_data[task].get(row, {}).get(col)
                        if metrics is not None and not np.isnan(metrics.get(metric_key, np.nan)):
                            values.append(metrics[metric_key] * 100)
                
                if values:
                    mean_val = np.mean(values)
                    z_row.append(mean_val)
                    text_row.append(f'{mean_val:.1f}')
                else:
                    z_row.append(None)
                    text_row.append('')
            z.append(z_row)
            text.append(text_row)
        
        fig.add_trace(
            go.Heatmap(
                z=z, x=eval_cols, y=model_rows,
                text=text, texttemplate='%{text}', textfont={'size': 10},
                colorscale='RdYlGn', zmin=0, zmax=100,
                showscale=(m_idx == len(metrics_list) - 1),
                hovertemplate='%{y} / %{x}: %{z:.1f}<extra></extra>'
            ),
            row=1, col=m_idx + 1
        )
    
    for m_idx in range(len(metrics_list)):
        fig.update_yaxes(showticklabels=(m_idx == 0), row=1, col=m_idx + 1)
    
    fig.update_layout(
        title=title,
        height=250,  # Fixed height like hypernym_dashboard.py
        margin=dict(l=80, r=20, t=50, b=30),  # Same margin as hypernym_dashboard.py
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig


def create_aggregated_direction_heatmap(all_heatmap_data_by_dir, tasks, training_rows, eval_cols, 
                                        title, metrics_list, direction, direction_label):
    """Create aggregated heatmap for a specific direction showing mean across tasks.
    
    Args:
        all_heatmap_data_by_dir: dict of {task: {direction: {training_variant: {eval_col: metrics}}}}
        tasks: list of tasks to aggregate
        training_rows: list of training variant labels
        eval_cols: list of eval column names
        title: figure title
        metrics_list: list of metrics to display
        direction: 'd2g' or 'g2d'
        direction_label: 'V2G' or 'G2V' for vanilla row labeling
    """
    fig = make_subplots(rows=1, cols=len(metrics_list), subplot_titles=metrics_list,
                        horizontal_spacing=0.03)
    
    metric_key_map = {
        'Val Acc': 'acc', 'Val ROC': 'val_roc', 'Gen ROC': 'gen_roc',
        'Correlation': 'corr', 'Corr-Pos': 'corr_pos', 'Corr-Neg': 'corr_neg'
    }
    
    # Include all relevant rows for this direction:
    # - Base model
    # - Vanilla for this direction (e.g., 'V2G' for d2g)
    # - Training variants like +tc, +tco, +lenorm
    # - Union models (e.g., 'Union', 'Union+tc')
    # Show empty rows if no data - don't filter them out
    relevant_rows = []
    for row in training_rows:
        if row == 'Base':
            relevant_rows.append(row)
        elif row in ('SFT', 'SFT+vallogodds'):
            relevant_rows.append(row)
        elif row in ('Pref only', 'Pref only+vallogodds'):
            relevant_rows.append(row)
        elif row == direction_label:  # Vanilla for this direction
            relevant_rows.append(row)
        elif row.startswith('+'):  # Training variants like +tc, +tco, +lenorm
            relevant_rows.append(row)
        elif row.startswith('Union'):  # Union models
            relevant_rows.append(row)
        # Skip other direction's vanilla (e.g., skip 'G2V' when direction_label is 'V2G')
    
    # Filter out hidden rows
    relevant_rows = [r for r in relevant_rows if r not in HIDDEN_ROWS]
    
    if not relevant_rows:
        return None
    
    for m_idx, metric in enumerate(metrics_list):
        metric_key = metric_key_map.get(metric, metric.lower())
        
        z = []
        text = []
        for row in relevant_rows:
            z_row = []
            text_row = []
            for col in eval_cols:
                values = []
                for task in tasks:
                    if task in all_heatmap_data_by_dir:
                        dir_data = all_heatmap_data_by_dir[task].get(direction, {})
                        metrics = dir_data.get(row, {}).get(col)
                        if metrics is not None and not np.isnan(metrics.get(metric_key, np.nan)):
                            values.append(metrics[metric_key] * 100)
                
                if values:
                    mean_val = np.mean(values)
                    z_row.append(mean_val)
                    text_row.append(f'{mean_val:.1f}')
                else:
                    z_row.append(None)
                    text_row.append('')
            z.append(z_row)
            text.append(text_row)
        
        fig.add_trace(
            go.Heatmap(
                z=z, x=eval_cols, y=relevant_rows,
                text=text, texttemplate='%{text}', textfont={'size': 10},
                colorscale='RdYlGn', zmin=0, zmax=100,
                showscale=(m_idx == len(metrics_list) - 1),
                hovertemplate='%{y} / %{x}: %{z:.1f}<extra></extra>'
            ),
            row=1, col=m_idx + 1
        )
    
    for m_idx in range(len(metrics_list)):
        fig.update_yaxes(showticklabels=(m_idx == 0), row=1, col=m_idx + 1)
    
    fig.update_layout(
        title=title,
        height=250,  # Fixed height like hypernym_dashboard.py
        margin=dict(l=80, r=20, t=50, b=30),  # Same margin as hypernym_dashboard.py
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig


def create_aggregated_bar_plot(all_heatmap_data, tasks, model_rows, eval_cols, metric, title):
    """Create bar plot with standard error for a single metric."""
    metric_key_map = {
        'Val Acc': 'acc', 'Val ROC': 'val_roc', 'Gen ROC': 'gen_roc',
        'Correlation': 'corr', 'Corr-Pos': 'corr_pos', 'Corr-Neg': 'corr_neg'
    }
    metric_key = metric_key_map.get(metric, metric.lower())
    
    fig = go.Figure()
    
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FECB52']
    
    x_positions = []
    x_labels = []
    current_x = 0
    
    for row_idx, row in enumerate(model_rows):
        for col_idx, col in enumerate(eval_cols):
            values = []
            for task in tasks:
                if task in all_heatmap_data:
                    metrics = all_heatmap_data[task].get(row, {}).get(col)
                    if metrics is not None and not np.isnan(metrics.get(metric_key, np.nan)):
                        values.append(metrics[metric_key] * 100)
            
            if values:
                mean_val = np.mean(values)
                std_err = np.std(values) / np.sqrt(len(values)) if len(values) > 1 else 0
            else:
                mean_val = 0
                std_err = 0
            
            fig.add_trace(go.Bar(
                x=[current_x],
                y=[mean_val],
                error_y=dict(type='data', array=[std_err], visible=True),
                marker_color=colors[row_idx % len(colors)],
                name=row if col_idx == 0 else None,
                showlegend=(col_idx == 0),
                legendgroup=row,
                hovertemplate=f'{row} / {col}: {mean_val:.1f} ± {std_err:.1f}<extra></extra>'
            ))
            
            x_positions.append(current_x)
            x_labels.append(col)
            current_x += 1
        
        current_x += 0.5
    
    is_correlation = metric in ['Correlation', 'Corr-Pos', 'Corr-Neg']
    if is_correlation:
        yaxis_config = dict(title=metric, showgrid=True, gridcolor='lightgray', gridwidth=1, dtick=10)
    else:
        yaxis_config = dict(title=metric, range=[40, 100], showgrid=True, gridcolor='lightgray', gridwidth=1, dtick=10)
    
    fig.update_layout(
        title=title,
        xaxis=dict(tickvals=x_positions, ticktext=x_labels, tickangle=45, showgrid=False),
        yaxis=yaxis_config,
        height=300,
        margin=dict(l=60, r=20, t=50, b=80),
        paper_bgcolor='white',
        plot_bgcolor='white',
        barmode='overlay',
        showlegend=True,
        legend=dict(orientation='h', y=1.15)
    )
    
    return fig


def create_direction_bar_plot(all_heatmap_data_by_dir, tasks, training_rows, eval_cols, metric, direction, direction_label):
    """Create bar plot for a specific direction (like hypernym_dashboard.py).
    
    Args:
        all_heatmap_data_by_dir: dict of {task: {direction: {training_variant: {eval_col: metrics}}}}
        tasks: list of tasks to aggregate
        training_rows: list of training variant labels (e.g., ['Base', 'V2G', '+tc', ...])
        eval_cols: list of eval column names
        metric: metric name
        direction: 'd2g' or 'g2d'
        direction_label: 'V2G' or 'G2V' for title
    """
    metric_key_map = {
        'Val Acc': 'acc', 'Val ROC': 'val_roc', 'Gen ROC': 'gen_roc',
        'Correlation': 'corr', 'Corr-Pos': 'corr_pos', 'Corr-Neg': 'corr_neg'
    }
    metric_key = metric_key_map.get(metric, metric.lower())
    
    fig = go.Figure()
    
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FECB52']
    
    # Filter rows relevant to this direction:
    # - Base model
    # - Vanilla for this direction (e.g., 'V2G' for d2g)
    # - Training variants like +tc, +tco, +lenorm
    # - Union models (e.g., 'Union', 'Union+tc')
    relevant_rows = []
    for row in training_rows:
        if row == 'Base':
            relevant_rows.append(row)
        elif row in ('SFT', 'SFT+vallogodds'):
            relevant_rows.append(row)
        elif row in ('Pref only', 'Pref only+vallogodds'):
            relevant_rows.append(row)
        elif row == direction_label:  # Vanilla for this direction
            relevant_rows.append(row)
        elif row.startswith('+'):  # Training variants
            relevant_rows.append(row)
        elif row.startswith('Union'):  # Union models
            relevant_rows.append(row)
    
    # Filter out hidden rows
    relevant_rows = [r for r in relevant_rows if r not in HIDDEN_ROWS]
    
    x_positions = []
    x_labels = []
    current_x = 0
    
    for row_idx, row in enumerate(relevant_rows):
        for col_idx, col in enumerate(eval_cols):
            values = []
            for task in tasks:
                if task in all_heatmap_data_by_dir:
                    dir_data = all_heatmap_data_by_dir[task].get(direction, {})
                    metrics = dir_data.get(row, {}).get(col)
                    if metrics is not None and not np.isnan(metrics.get(metric_key, np.nan)):
                        values.append(metrics[metric_key] * 100)
            
            if values:
                mean_val = np.mean(values)
                std_err = np.std(values) / np.sqrt(len(values)) if len(values) > 1 else 0
            else:
                mean_val = 0
                std_err = 0
            
            fig.add_trace(go.Bar(
                x=[current_x],
                y=[mean_val],
                error_y=dict(type='data', array=[std_err], visible=True),
                marker_color=colors[row_idx % len(colors)],
                name=row if col_idx == 0 else None,
                showlegend=(col_idx == 0),
                legendgroup=row,
                hovertemplate=f'{row} / {col}: {mean_val:.1f} ± {std_err:.1f}<extra></extra>'
            ))
            
            x_positions.append(current_x)
            x_labels.append(col)
            current_x += 1
        
        current_x += 0.5
    
    is_correlation = metric in ['Correlation', 'Corr-Pos', 'Corr-Neg']
    if is_correlation:
        yaxis_config = dict(title=metric, showgrid=True, gridcolor='lightgray', gridwidth=1, dtick=10)
    else:
        yaxis_config = dict(title=metric, range=[40, 100], showgrid=True, gridcolor='lightgray', gridwidth=1, dtick=10)
    
    fig.update_layout(
        title=f'{direction_label} - {metric}',
        xaxis=dict(tickvals=x_positions, ticktext=x_labels, tickangle=45, showgrid=False),
        yaxis=yaxis_config,
        height=300,
        margin=dict(l=60, r=20, t=50, b=80),
        paper_bgcolor='white',
        plot_bgcolor='white',
        barmode='overlay',
        showlegend=True,
        legend=dict(orientation='h', y=1.15)
    )
    
    return fig


# =============================================================================
# DASH APP
# =============================================================================

app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Layout with config page and viz page
app.layout = html.Div([
    # Store for config
    dcc.Store(id='config-store', data=None),
    dcc.Store(id='files-store', data=None),
    
    # Config Page
    html.Div(id='config-page', children=[
        html.H1('📊 Dashboard Configuration', style={'textAlign': 'center', 'color': '#333', 'marginBottom': '30px'}),
        
        html.Div([
            # Buttons row
            html.Div([
                html.Button('🔍 Auto-detect from outputs/ dir', id='auto-detect-btn', 
                           style={'marginRight': '10px', 'padding': '10px 20px', 'fontSize': '14px',
                                  'backgroundColor': '#4CAF50', 'color': 'white', 'border': 'none',
                                  'borderRadius': '5px', 'cursor': 'pointer'}),
                html.Button('📁 Load JSON Config', id='load-json-btn',
                           style={'marginRight': '10px', 'padding': '10px 20px', 'fontSize': '14px',
                                  'backgroundColor': '#2196F3', 'color': 'white', 'border': 'none',
                                  'borderRadius': '5px', 'cursor': 'pointer'}),
                html.Button('💾 Save Config to JSON', id='save-json-btn',
                           style={'marginRight': '10px', 'padding': '10px 20px', 'fontSize': '14px',
                                  'backgroundColor': '#FF9800', 'color': 'white', 'border': 'none',
                                  'borderRadius': '5px', 'cursor': 'pointer'}),
                html.Button('🚀 Load Dashboard', id='load-dashboard-btn-top',
                           style={'padding': '10px 20px', 'fontSize': '14px',
                                  'backgroundColor': '#673AB7', 'color': 'white', 'border': 'none',
                                  'borderRadius': '5px', 'cursor': 'pointer', 'fontWeight': 'bold'}),
            ], style={'marginBottom': '20px', 'textAlign': 'center'}),
            
            # Status message
            html.Div(id='config-status', style={
                'padding': '10px', 'marginBottom': '20px', 'borderRadius': '5px',
                'backgroundColor': '#e3f2fd', 'color': '#1565c0', 'textAlign': 'center'
            }),
            
            # Config form
            html.Div([
                # Outputs directory
                html.Div([
                    html.Label('Outputs Directory:', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                    dcc.Input(id='config-outputs-dir', type='text', value=str(DEFAULT_OUTPUTS_DIR),
                             style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'})
                ], style={'marginBottom': '15px'}),
                
                # Task pattern
                html.Div([
                    html.Label('Task Pattern (regex with capture group):', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                    dcc.Input(id='config-task-pattern', type='text', value=r'hypernym-([a-zA-Z]+)',
                             style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'}),
                    html.Small('Example: hypernym-([a-zA-Z]+) extracts "bananas" from "hypernym-bananas"', 
                              style={'color': '#666'})
                ], style={'marginBottom': '15px'}),
                
                # Split patterns
                html.Div([
                    html.Label('Split Patterns (JSON):', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                    dcc.Textarea(id='config-split-patterns', 
                                value='{"train": "_train_", "test": "_test_"}',
                                style={'width': '100%', 'height': '60px', 'padding': '8px', 'borderRadius': '4px', 
                                       'border': '1px solid #ccc', 'fontFamily': 'monospace'})
                ], style={'marginBottom': '15px'}),
                
                # Label column
                html.Div([
                    html.Label('Label Column:', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                    dcc.Input(id='config-label-col', type='text', value='gpt4_ground_truth',
                             style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'})
                ], style={'marginBottom': '15px'}),
                
                # Label map
                html.Div([
                    html.Label('Label Map (JSON, or empty if numeric):', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                    dcc.Input(id='config-label-map', type='text', value='{"yes": 1, "no": 0}',
                             style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'})
                ], style={'marginBottom': '15px'}),
                
                # Model detection
                html.Div([
                    html.Label('Model Detection (JSON):', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                    dcc.Textarea(id='config-model-detection',
                                value=json.dumps(DEFAULT_CONFIG['model_detection'], indent=2),
                                style={'width': '100%', 'height': '150px', 'padding': '8px', 'borderRadius': '4px',
                                       'border': '1px solid #ccc', 'fontFamily': 'monospace'})
                ], style={'marginBottom': '15px'}),
                
                # Aggregation groups
                html.Div([
                    html.Label('Aggregation Groups (JSON):', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                    dcc.Textarea(id='config-aggregation',
                                value='{"All Hypernym": "hypernym-*"}',
                                style={'width': '100%', 'height': '80px', 'padding': '8px', 'borderRadius': '4px',
                                       'border': '1px solid #ccc', 'fontFamily': 'monospace'}),
                    html.Small('Use "prefix-*" for patterns or ["task1", "task2"] for explicit lists', 
                              style={'color': '#666'})
                ], style={'marginBottom': '15px'}),
                
                # Eval columns
                html.Div([
                    html.Label('Eval Columns (JSON):', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                    dcc.Textarea(id='config-eval-cols',
                                value=json.dumps(DEFAULT_CONFIG['eval_columns'], indent=2),
                                style={'width': '100%', 'height': '100px', 'padding': '8px', 'borderRadius': '4px',
                                       'border': '1px solid #ccc', 'fontFamily': 'monospace'})
                ], style={'marginBottom': '15px'}),
                
                # Union models config
                html.Div([
                    html.Label('Union Models (JSON, or {} to disable):', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                    dcc.Textarea(id='config-union-models',
                                value=json.dumps(DEFAULT_CONFIG['union_models'], indent=2),
                                style={'width': '100%', 'height': '100px', 'padding': '8px', 'borderRadius': '4px',
                                       'border': '1px solid #ccc', 'fontFamily': 'monospace'}),
                    html.Small('Models trained on combined task, evaluated on individual tasks. Set to {} to disable.', 
                              style={'color': '#666'})
                ], style={'marginBottom': '25px'}),
                
            ], style={'maxWidth': '800px', 'margin': '0 auto', 'padding': '20px',
                     'backgroundColor': '#f9f9f9', 'borderRadius': '10px'}),
            
            # Load Dashboard button
            html.Div([
                html.Button('🚀 Load Dashboard', id='load-dashboard-btn',
                           style={'padding': '15px 40px', 'fontSize': '18px',
                                  'backgroundColor': '#673AB7', 'color': 'white', 'border': 'none',
                                  'borderRadius': '8px', 'cursor': 'pointer', 'fontWeight': 'bold'})
            ], style={'textAlign': 'center', 'marginTop': '30px'}),
            
        ], style={'maxWidth': '900px', 'margin': '0 auto'})
    ], style={'padding': '40px', 'backgroundColor': 'white', 'minHeight': '100vh'}),
    
    # Visualization Page (hidden initially)
    html.Div(id='viz-page', children=[
        html.Div([
            html.H1('⚖️ Generator-Validator Dashboard', style={'textAlign': 'center', 'color': '#333', 'display': 'inline-block'}),
            html.Button('⚙️ Back to Config', id='back-to-config-btn',
                       style={'marginLeft': '20px', 'padding': '8px 16px', 'fontSize': '12px',
                              'backgroundColor': '#9E9E9E', 'color': 'white', 'border': 'none',
                              'borderRadius': '4px', 'cursor': 'pointer', 'verticalAlign': 'middle'})
        ], style={'textAlign': 'center', 'marginBottom': '20px'}),
        
        # Three dropdowns for filtering
        html.Div([
            html.Div([
                html.Label('Task:', style={'fontWeight': 'bold'}),
                dcc.Dropdown(id='task-selector', style={'width': '100%'}, clearable=False)
            ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '2%'}),
            
            html.Div([
                html.Label('Split:', style={'fontWeight': 'bold'}),
                dcc.Dropdown(id='split-selector', style={'width': '100%'}, clearable=False)
            ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '2%'}),
            
            html.Div([
                html.Label('Model:', style={'fontWeight': 'bold'}),
                dcc.Dropdown(id='model-selector', style={'width': '100%'}, clearable=False)
            ], style={'width': '30%', 'display': 'inline-block'}),
        ], style={'width': '80%', 'margin': '20px auto'}),
        
        # Status message
        html.Div(id='file-status', style={
            'width': '80%', 'margin': '10px auto', 'textAlign': 'center',
            'color': '#666', 'fontStyle': 'italic'
        }),
        
        # Stats panel
        html.Div(id='stats-panel', style={
            'width': '60%', 'margin': '10px auto', 'padding': '15px',
            'backgroundColor': '#f5f5f5', 'borderRadius': '8px',
            'fontFamily': 'monospace', 'whiteSpace': 'pre-wrap'
        }),
        
        # Main scatter plot
        dcc.Graph(id='main-scatter', style={'height': '700px'}),
        
        # Click-to-view response panel for main scatter
        html.Div(id='response-panel-main', style={
            'width': '80%', 'margin': '8px auto 12px', 'padding': '10px 14px',
            'backgroundColor': '#fff8e1', 'borderRadius': '8px',
            'border': '1px solid #ffe0b2', 'fontFamily': 'monospace',
            'whiteSpace': 'pre-wrap'
        }, children="Click a point in the main scatter to view the full prompt/response here."),
        
        # Faceted by strategy
        html.Details([
            html.Summary('📊 Faceted View by Strategy', style={'cursor': 'pointer', 'fontWeight': 'bold'}),
            dcc.Graph(id='faceted-plot', style={'height': '600px'}),
            html.Div(id='response-panel-faceted', style={
                'width': '80%', 'margin': '8px auto 12px', 'padding': '10px 14px',
                'backgroundColor': '#fff8e1', 'borderRadius': '8px',
                'border': '1px solid #ffe0b2', 'fontFamily': 'monospace',
                'whiteSpace': 'pre-wrap'
            }, children="Click a point in the faceted plot to view the full prompt/response here.")
        ], style={'margin': '20px'}),
        
        # PCA/Standardized plot
        html.Details([
            html.Summary('🔬 Standardized Scores & PCA Analysis', style={'cursor': 'pointer', 'fontWeight': 'bold'}),
            dcc.Graph(id='pca-plot', style={'height': '500px'}),
            html.Div(id='response-panel-pca', style={
                'width': '80%', 'margin': '8px auto 12px', 'padding': '10px 14px',
                'backgroundColor': '#fff8e1', 'borderRadius': '8px',
                'border': '1px solid #ffe0b2', 'fontFamily': 'monospace',
                'whiteSpace': 'pre-wrap'
            }, children="Click a point in the PCA plots to view the full prompt/response here.")
        ], style={'margin': '20px'}),
        
        # Compare corrections 2x2
        html.Details([
            html.Summary('🔄 Compare Score Corrections (2x2)', style={'cursor': 'pointer', 'fontWeight': 'bold'}),
            dcc.Graph(id='compare-plot', style={'height': '1100px'}),
            html.Div(id='response-panel-compare', style={
                'width': '80%', 'margin': '8px auto 12px', 'padding': '10px 14px',
                'backgroundColor': '#fff8e1', 'borderRadius': '8px',
                'border': '1px solid #ffe0b2', 'fontFamily': 'monospace',
                'whiteSpace': 'pre-wrap'
            }, children="Click a point in the compare plot to view the full prompt/response here.")
        ], style={'margin': '20px'}),
        
        # Heatmaps section
        html.Details([
            html.Summary('🔥 Heatmaps (All Tasks)', style={'cursor': 'pointer', 'fontWeight': 'bold'}),
            dcc.Loading(
                id='heatmaps-loading',
                type='default',  # Options: 'graph', 'cube', 'circle', 'dot', 'default'
                children=html.Div(id='all-heatmaps-container'),
                style={'minHeight': '200px'}
            )
        ], open=True, style={'margin': '20px'}),
        
    ], style={'padding': '20px', 'backgroundColor': 'white', 'minHeight': '100vh', 'display': 'none'})
])


# =============================================================================
# CALLBACKS
# =============================================================================

@app.callback(
    [Output('config-outputs-dir', 'value'),
     Output('config-task-pattern', 'value'),
     Output('config-split-patterns', 'value'),
     Output('config-label-col', 'value'),
     Output('config-label-map', 'value'),
     Output('config-model-detection', 'value'),
     Output('config-aggregation', 'value'),
     Output('config-eval-cols', 'value'),
     Output('config-union-models', 'value'),
     Output('config-status', 'children'),
     Output('config-status', 'style')],
    [Input('auto-detect-btn', 'n_clicks'),
     Input('load-json-btn', 'n_clicks')],
    prevent_initial_call=True
)
def handle_config_buttons(auto_clicks, load_clicks):
    """Handle auto-detect and load JSON buttons."""
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    base_style = {'padding': '10px', 'marginBottom': '20px', 'borderRadius': '5px', 'textAlign': 'center'}
    
    if button_id == 'auto-detect-btn':
        config, msg = auto_detect_config()
        style = {**base_style, 'backgroundColor': '#e8f5e9', 'color': '#2e7d32'}
        
        return (
            config['outputs_dir'],
            config.get('task_pattern', ''),
            json.dumps(config.get('split_patterns', {})),
            config.get('label_column', 'gpt4_ground_truth'),
            json.dumps(config.get('label_map')) if config.get('label_map') else '',
            json.dumps(config.get('model_detection', {}), indent=2),
            json.dumps(config.get('aggregation_groups', {})),
            json.dumps(config.get('eval_columns', {}), indent=2),
            json.dumps(config.get('union_models', {}), indent=2),
            f"✅ {msg}",
            style
        )
    
    elif button_id == 'load-json-btn':
        config, error = load_config_from_file()
        if error:
            style = {**base_style, 'backgroundColor': '#ffebee', 'color': '#c62828'}
            return (
                dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update,
                f"⚠️ {error}",
                style
            )
        
        style = {**base_style, 'backgroundColor': '#e8f5e9', 'color': '#2e7d32'}
        return (
            config.get('outputs_dir', str(DEFAULT_OUTPUTS_DIR)),
            config.get('task_pattern', ''),
            json.dumps(config.get('split_patterns', {})),
            config.get('label_column', 'gpt4_ground_truth'),
            json.dumps(config.get('label_map')) if config.get('label_map') else '',
            json.dumps(config.get('model_detection', {}), indent=2),
            json.dumps(config.get('aggregation_groups', {})),
            json.dumps(config.get('eval_columns', {}), indent=2),
            json.dumps(config.get('union_models', {}), indent=2),
            f"✅ Loaded config from {CONFIG_FILE}",
            style
        )
    
    raise PreventUpdate


@app.callback(
    [Output('config-status', 'children', allow_duplicate=True),
     Output('config-status', 'style', allow_duplicate=True)],
    [Input('save-json-btn', 'n_clicks')],
    [State('config-outputs-dir', 'value'),
     State('config-task-pattern', 'value'),
     State('config-split-patterns', 'value'),
     State('config-label-col', 'value'),
     State('config-label-map', 'value'),
     State('config-model-detection', 'value'),
     State('config-aggregation', 'value'),
     State('config-eval-cols', 'value'),
     State('config-union-models', 'value')],
    prevent_initial_call=True
)
def save_config(n_clicks, outputs_dir, task_pattern, split_patterns, label_col, 
                label_map, model_detection, aggregation, eval_cols, union_models):
    """Save current config to JSON file."""
    if not n_clicks:
        raise PreventUpdate
    
    base_style = {'padding': '10px', 'marginBottom': '20px', 'borderRadius': '5px', 'textAlign': 'center'}
    
    try:
        config = {
            'outputs_dir': outputs_dir,
            'task_pattern': task_pattern,
            'split_patterns': json.loads(split_patterns) if split_patterns else {},
            'label_column': label_col,
            'label_map': json.loads(label_map) if label_map else None,
            'model_detection': json.loads(model_detection) if model_detection else {},
            'aggregation_groups': json.loads(aggregation) if aggregation else {},
            'eval_columns': json.loads(eval_cols) if eval_cols else {},
            'union_models': json.loads(union_models) if union_models else {}
        }
        
        save_config_to_file(config)
        style = {**base_style, 'backgroundColor': '#e8f5e9', 'color': '#2e7d32'}
        return f"✅ Saved config to {CONFIG_FILE}", style
    
    except Exception as e:
        style = {**base_style, 'backgroundColor': '#ffebee', 'color': '#c62828'}
        return f"⚠️ Error saving config: {e}", style


@app.callback(
    [Output('config-page', 'style'),
     Output('viz-page', 'style'),
     Output('config-store', 'data'),
     Output('files-store', 'data'),
     Output('task-selector', 'options'),
     Output('task-selector', 'value'),
     Output('split-selector', 'options'),
     Output('split-selector', 'value'),
     Output('model-selector', 'options'),
     Output('model-selector', 'value')],
    [Input('load-dashboard-btn', 'n_clicks'),
     Input('load-dashboard-btn-top', 'n_clicks'),
     Input('back-to-config-btn', 'n_clicks')],
    [State('config-outputs-dir', 'value'),
     State('config-task-pattern', 'value'),
     State('config-split-patterns', 'value'),
     State('config-label-col', 'value'),
     State('config-label-map', 'value'),
     State('config-model-detection', 'value'),
     State('config-aggregation', 'value'),
     State('config-eval-cols', 'value'),
     State('config-union-models', 'value')],
    prevent_initial_call=True
)
def toggle_pages(load_clicks, load_clicks_top, back_clicks, outputs_dir, task_pattern, split_patterns,
                 label_col, label_map, model_detection, aggregation, eval_cols, union_models):
    """Toggle between config page and viz page."""
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Both load buttons do the same thing
    if button_id == 'load-dashboard-btn-top':
        button_id = 'load-dashboard-btn'
    
    config_visible = {'padding': '40px', 'backgroundColor': 'white', 'minHeight': '100vh'}
    config_hidden = {'display': 'none'}
    viz_visible = {'padding': '20px', 'backgroundColor': 'white', 'minHeight': '100vh'}
    viz_hidden = {'display': 'none'}
    
    if button_id == 'back-to-config-btn':
        # Clear the stores when going back to config to force refresh on next load
        return (config_visible, viz_hidden, None, None,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update)
    
    # Load dashboard
    try:
        # Parse model_detection with better error handling
        try:
            model_detection_parsed = json.loads(model_detection) if model_detection else {}
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse model_detection JSON: {e}")
            model_detection_parsed = {}
        
        config = {
            'outputs_dir': outputs_dir,
            'task_pattern': task_pattern,
            'split_patterns': json.loads(split_patterns) if split_patterns else {},
            'label_column': label_col,
            'label_map': json.loads(label_map) if label_map else None,
            'model_detection': model_detection_parsed,
            'aggregation_groups': json.loads(aggregation) if aggregation else {},
            'eval_columns': json.loads(eval_cols) if eval_cols else {},
            'union_models': json.loads(union_models) if union_models else {},
            'metrics': DEFAULT_CONFIG['metrics']
        }
        
        # Discover files
        files_info = discover_scores_files(config)
        
        if not files_info:
            raise PreventUpdate
        
        # Build dropdown options
        all_tasks = sorted(set(f['task'] for f in files_info))
        all_splits = sorted(set(f['split'] for f in files_info))  # test before train alphabetically
        all_models = sorted(set(f['model_type'] for f in files_info), key=lambda x: (0 if x == 'Base' else 1, x))
        
        task_options = [{'label': t, 'value': t} for t in all_tasks]
        split_options = [{'label': s, 'value': s} for s in all_splits]
        model_options = [{'label': m, 'value': m} for m in all_models]
        
        return (
            config_hidden, viz_visible, config, files_info,
            task_options, all_tasks[0] if all_tasks else None,
            split_options, all_splits[0] if all_splits else None,
            model_options, all_models[0] if all_models else None
        )
    
    except Exception as e:
        print(f"Error loading dashboard: {e}")
        raise PreventUpdate


@app.callback(
    [Output('model-selector', 'options', allow_duplicate=True),
     Output('model-selector', 'value', allow_duplicate=True)],
    [Input('task-selector', 'value'),
     Input('split-selector', 'value')],
    [State('files-store', 'data')],
    prevent_initial_call=True
)
def update_model_options(task, split, files_info):
    """Update model dropdown based on task and split selection."""
    if not task or not split or not files_info:
        raise PreventUpdate
    
    available_models = sorted(
        set(f['model_type'] for f in files_info if f['task'] == task and f['split'] == split),
        key=lambda x: (0 if x == 'Base' else 1, x)
    )
    
    if not available_models:
        return [{'label': 'No models available', 'value': None}], None
    
    return [{'label': m, 'value': m} for m in available_models], available_models[0]


@app.callback(
    [Output('file-status', 'children'),
     Output('stats-panel', 'children'),
     Output('main-scatter', 'figure'),
     Output('faceted-plot', 'figure'),
     Output('pca-plot', 'figure'),
     Output('compare-plot', 'figure')],
    [Input('task-selector', 'value'),
     Input('split-selector', 'value'),
     Input('model-selector', 'value')],
    [State('config-store', 'data'),
     State('files-store', 'data')]
)
def update_visualizations(task, split, model_type, config, files_info):
    """Update main visualizations based on selections."""
    empty_fig = go.Figure()
    
    if not task or not split or not model_type or not config or not files_info:
        return "Please select all options", "No file selected", empty_fig, empty_fig, empty_fig, empty_fig
    
    # Find matching file
    matching = [f for f in files_info if f['task'] == task and f['split'] == split and f['model_type'] == model_type]
    
    if not matching:
        return f"No data for: {task} | {split} | {model_type}", "No data", empty_fig, empty_fig, empty_fig, empty_fig
    
    csv_path = matching[0]['path']
    
    # Load data
    df = load_scores_data(csv_path, config)
    
    # Determine metric type
    metric_type = 'log-odds' if 'log-odds' in csv_path else 'log-probs'
    metric_label = 'log-odds' if metric_type == 'log-odds' else 'log-probs'
    threshold = 0 if metric_type == 'log-odds' else np.log(0.5)
    
    gen_col = config.get('gen_score_col', 'gen_score')
    val_col = config.get('val_score_col', 'val_score')
    
    if gen_col not in df.columns or val_col not in df.columns or 'label' not in df.columns:
        return f"Missing required columns in {csv_path}", "Error", empty_fig, empty_fig, empty_fig, empty_fig
    
    gen_scores = df[gen_col].values
    val_scores = df[val_col].values
    labels = df['label'].values
    strategies = df['strategy'].values if 'strategy' in df.columns else np.array(['unknown'] * len(df))
    
    pos_mask = labels == 1
    neg_mask = labels == 0
    
    # Compute metrics
    metrics = compute_metrics(gen_scores, val_scores, labels, metric_type)
    
    # Stats text
    stats_text = (
        f"corr = {metrics['corr']*100:.1f}   corr-pos = {metrics['corr_pos']*100:.1f}   "
        f"corr-neg = {metrics['corr_neg']*100:.1f}\n"
        f"Accuracy = {metrics['acc']*100:.1f}   Val ROC = {metrics['val_roc']*100:.1f}   "
        f"Gen ROC = {metrics['gen_roc']*100:.1f}"
    )
    
    # === MAIN SCATTER PLOT WITH MARGINALS ===
    main_fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.8, 0.2],
        row_heights=[0.2, 0.8],
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
        specs=[[{"type": "histogram"}, None],
               [{"type": "scatter"}, {"type": "histogram"}]]
    )
    
    # Outliers: Compute outliers FIRST so we can exclude them from regular dot traces
    # outlier_method: 'identity' = distance from y=x line (after standardizing)
    #                 'kendall' = fraction of discordant pairs (Kendall tau violations)
    outlier_method = 'kendall'
    
    X = np.column_stack([gen_scores, val_scores])
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    if outlier_method == 'identity':
        # Distance from y=x line in standardized space: |y - x| / sqrt(2)
        outlier_scores = np.abs(X_std[:, 1] - X_std[:, 0]) / np.sqrt(2)
    elif outlier_method == 'kendall':
        # Fraction of discordant pairs for each point (Kendall tau violations)
        n = len(gen_scores)
        outlier_scores = np.zeros(n)
        for i in range(n):
            discordant = 0
            for j in range(n):
                if i != j:
                    # Discordant if x order doesn't match y order
                    x_diff = gen_scores[i] - gen_scores[j]
                    y_diff = val_scores[i] - val_scores[j]
                    if x_diff * y_diff < 0:  # Different signs = discordant
                        discordant += 1
            outlier_scores[i] = discordant / (n - 1)
    else:
        raise ValueError(f"Unknown outlier_method: {outlier_method}")
    
    outlier_indices = np.argsort(outlier_scores)[-40:]
    
    outlier_colors = [POS_OUTLIER_COLOR if labels[i] == 1 else NEG_OUTLIER_COLOR for i in outlier_indices]
    
    # Create mask for non-outlier points
    outlier_set = set(outlier_indices)
    non_outlier_mask = np.array([i not in outlier_set for i in range(len(labels))])
    
    # Create hover text for all points.
    # Always include (gen, val); optionally include noun2.
    has_noun2 = 'noun2' in df.columns
    has_prompt = 'prompt' in df.columns
    has_response = 'response' in df.columns
    
    full_prompts = df['prompt'].astype(str).fillna('') if has_prompt else pd.Series([''] * len(df))
    full_responses = df['response'].astype(str).fillna('') if has_response else pd.Series([''] * len(df))
    
    hover_texts = []
    for i in range(len(gen_scores)):
        parts = [f"Gen={gen_scores[i]:.2f}", f"Val={val_scores[i]:.2f}"]
        if has_noun2:
            parts.append(f"Item={df['noun2'].iloc[i]}")
        hover_text = " | ".join(parts)
        hover_texts.append(hover_text)
    hover_texts = np.array(hover_texts)
    
    # Customdata for click: full prompt/response
    customdata = np.column_stack([full_prompts.values, full_responses.values]) if (has_prompt or has_response) else None
    
    # Add scatter traces (excluding outliers)
    main_fig.add_trace(
        go.Scatter(
            x=gen_scores[pos_mask & non_outlier_mask], y=val_scores[pos_mask & non_outlier_mask],
            mode='markers', marker=dict(color=POS_CLASS_COLOR, size=8, opacity=0.6),
            name='Positive', legendgroup='pos',
            hovertext=hover_texts[pos_mask & non_outlier_mask], hoverinfo='text',
            customdata=customdata[pos_mask & non_outlier_mask] if customdata is not None else None
        ),
        row=2, col=1
    )
    main_fig.add_trace(
        go.Scatter(
            x=gen_scores[neg_mask & non_outlier_mask], y=val_scores[neg_mask & non_outlier_mask],
            mode='markers', marker=dict(color=NEG_CLASS_COLOR, size=8, opacity=0.6),
            name='Negative', legendgroup='neg',
            hovertext=hover_texts[neg_mask & non_outlier_mask], hoverinfo='text',
            customdata=customdata[neg_mask & non_outlier_mask] if customdata is not None else None
        ),
        row=2, col=1
    )
    
    main_fig.add_hline(y=threshold, line=dict(color='red', dash='dash', width=2), row=2, col=1)
    
    # Outlier display text (shown next to X markers) - just noun2
    outlier_display_texts = []
    for i in outlier_indices:
        if has_noun2:
            outlier_display_texts.append(df['noun2'].iloc[i][:8])
        else:
            outlier_display_texts.append('')
    
    main_fig.add_trace(
        go.Scatter(
            x=gen_scores[outlier_indices], y=val_scores[outlier_indices],
            mode='markers+text',
            marker=dict(symbol='x', size=9, color=outlier_colors),
            text=outlier_display_texts,
            textposition='top right', textfont=dict(size=8),
            name='Outliers', showlegend=False,
            hovertext=hover_texts[outlier_indices], hoverinfo='text',
            customdata=customdata[outlier_indices] if customdata is not None else None
        ),
        row=2, col=1
    )
    
    # Histograms
    main_fig.add_trace(go.Histogram(x=gen_scores[pos_mask], marker_color=POS_CLASS_COLOR, opacity=0.6, showlegend=False), row=1, col=1)
    main_fig.add_trace(go.Histogram(x=gen_scores[neg_mask], marker_color=NEG_CLASS_COLOR, opacity=0.6, showlegend=False), row=1, col=1)
    main_fig.add_trace(go.Histogram(y=val_scores[pos_mask], marker_color=POS_CLASS_COLOR, opacity=0.6, showlegend=False), row=2, col=2)
    main_fig.add_trace(go.Histogram(y=val_scores[neg_mask], marker_color=NEG_CLASS_COLOR, opacity=0.6, showlegend=False), row=2, col=2)
    
    main_fig.update_layout(title='Generator vs Validator Scores', paper_bgcolor='white', plot_bgcolor='white', showlegend=True)
    main_fig.update_xaxes(title_text='Generator log-probs', row=2, col=1, showgrid=True, gridcolor='lightgray')
    main_fig.update_yaxes(title_text=f'Validator {metric_label}', row=2, col=1, showgrid=True, gridcolor='lightgray')
    
    # === FACETED PLOT BY STRATEGY ===
    unique_strategies = sorted(set(strategies))
    n_strats = len(unique_strategies)
    n_cols = min(3, n_strats)
    n_rows = (n_strats + n_cols - 1) // n_cols
    
    faceted_fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=unique_strategies)
    
    x_min, x_max = gen_scores.min(), gen_scores.max()
    y_min, y_max = val_scores.min(), val_scores.max()
    
    for idx, strat in enumerate(unique_strategies):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        strat_mask = strategies == strat
        strat_pos = strat_mask & pos_mask
        strat_neg = strat_mask & neg_mask
        
        pos_below = ((labels == 1) & strat_mask & (val_scores < threshold)).sum()
        total_pos = strat_pos.sum()
        
        faceted_fig.add_trace(
            go.Scatter(
                x=gen_scores[strat_pos], y=val_scores[strat_pos],
                mode='markers', marker=dict(color=POS_CLASS_COLOR, size=6, opacity=0.6),
                name=f'Pos ({total_pos})', showlegend=(idx == 0),
                hovertext=hover_texts[strat_pos], hoverinfo='text',
                customdata=customdata[strat_pos] if customdata is not None else None
            ),
            row=row, col=col
        )
        faceted_fig.add_trace(
            go.Scatter(
                x=gen_scores[strat_neg], y=val_scores[strat_neg],
                mode='markers', marker=dict(color=NEG_CLASS_COLOR, size=6, opacity=0.6),
                name=f'Neg ({strat_neg.sum()})', showlegend=(idx == 0),
                hovertext=hover_texts[strat_neg], hoverinfo='text',
                customdata=customdata[strat_neg] if customdata is not None else None
            ),
            row=row, col=col
        )
        faceted_fig.add_hline(y=threshold, line=dict(color='red', dash='dash'), row=row, col=col)
        
        # Add annotation for misclassified
        faceted_fig.add_annotation(
            x=0.02, y=0.98, xref=f'x{idx+1 if idx > 0 else ""} domain', 
            yref=f'y{idx+1 if idx > 0 else ""} domain',
            text=f'Pos<thresh: {pos_below}/{total_pos}',
            showarrow=False, font=dict(size=9),
            bgcolor='white', bordercolor='gray', borderwidth=1
        )
    
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            faceted_fig.update_xaxes(range=[x_min - x_pad, x_max + x_pad], showgrid=True, gridcolor='lightgray', row=r, col=c)
            faceted_fig.update_yaxes(range=[y_min - y_pad, y_max + y_pad], showgrid=True, gridcolor='lightgray', row=r, col=c)
    
    faceted_fig.update_layout(title='Faceted by Strategy', paper_bgcolor='white', plot_bgcolor='white', height=400 * n_rows)
    
    # === PCA PLOT ===
    # Compute PCA for the right subplot
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)
    
    pca_fig = make_subplots(rows=1, cols=2, subplot_titles=['Standardized Scores', 'PCA'])
    
    # Left: Standardized scores (exclude outliers from regular dots)
    pca_fig.add_trace(
        go.Scatter(
            x=X_std[pos_mask & non_outlier_mask, 0], y=X_std[pos_mask & non_outlier_mask, 1], mode='markers',
            marker=dict(color=POS_CLASS_COLOR, size=6, opacity=0.5), name='Positive',
            hovertext=hover_texts[pos_mask & non_outlier_mask], hoverinfo='text',
            customdata=customdata[pos_mask & non_outlier_mask] if customdata is not None else None
        ), row=1, col=1
    )
    pca_fig.add_trace(
        go.Scatter(
            x=X_std[neg_mask & non_outlier_mask, 0], y=X_std[neg_mask & non_outlier_mask, 1], mode='markers',
            marker=dict(color=NEG_CLASS_COLOR, size=6, opacity=0.5), name='Negative',
            hovertext=hover_texts[neg_mask & non_outlier_mask], hoverinfo='text',
            customdata=customdata[neg_mask & non_outlier_mask] if customdata is not None else None
        ), row=1, col=1
    )
    pca_fig.add_trace(
        go.Scatter(
            x=X_std[outlier_indices, 0], y=X_std[outlier_indices, 1], mode='markers',
            marker=dict(symbol='x', size=8, color=outlier_colors),
            name='Outliers', showlegend=False,
            hovertext=hover_texts[outlier_indices], hoverinfo='text',
            customdata=customdata[outlier_indices] if customdata is not None else None
        ), row=1, col=1
    )
    
    # Add y=x identity line to standardized scores plot
    std_range = max(np.abs(X_std).max(), 3)
    pca_fig.add_trace(go.Scatter(x=[-std_range, std_range], y=[-std_range, std_range], mode='lines',
                                  line=dict(color='gray', dash='dot', width=2),
                                  name='y=x', showlegend=True), row=1, col=1)
    
    # Right: PCA (exclude outliers from regular dots)
    pca_fig.add_trace(
        go.Scatter(
            x=X_pca[pos_mask & non_outlier_mask, 0], y=X_pca[pos_mask & non_outlier_mask, 1], mode='markers',
            marker=dict(color=POS_CLASS_COLOR, size=6, opacity=0.5), showlegend=False,
            hovertext=hover_texts[pos_mask & non_outlier_mask], hoverinfo='text',
            customdata=customdata[pos_mask & non_outlier_mask] if customdata is not None else None
        ), row=1, col=2
    )
    pca_fig.add_trace(
        go.Scatter(
            x=X_pca[neg_mask & non_outlier_mask, 0], y=X_pca[neg_mask & non_outlier_mask, 1], mode='markers',
            marker=dict(color=NEG_CLASS_COLOR, size=6, opacity=0.5), showlegend=False,
            hovertext=hover_texts[neg_mask & non_outlier_mask], hoverinfo='text',
            customdata=customdata[neg_mask & non_outlier_mask] if customdata is not None else None
        ), row=1, col=2
    )
    pca_fig.add_trace(
        go.Scatter(
            x=X_pca[outlier_indices, 0], y=X_pca[outlier_indices, 1], mode='markers',
            marker=dict(symbol='x', size=8, color=outlier_colors),
            showlegend=False,
            hovertext=hover_texts[outlier_indices], hoverinfo='text',
            customdata=customdata[outlier_indices] if customdata is not None else None
        ), row=1, col=2
    )
    
    pca_fig.update_layout(paper_bgcolor='white', plot_bgcolor='white')
    # Add zeroline (axis lines) for both plots
    pca_fig.update_xaxes(title_text='Generator (std)', row=1, col=1, showgrid=True, gridcolor='lightgray',
                         zeroline=True, zerolinecolor='black', zerolinewidth=1)
    pca_fig.update_yaxes(title_text='Validator (std)', row=1, col=1, showgrid=True, gridcolor='lightgray',
                         zeroline=True, zerolinecolor='black', zerolinewidth=1)
    pca_fig.update_xaxes(title_text=f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', row=1, col=2, showgrid=True, gridcolor='lightgray',
                         zeroline=True, zerolinecolor='black', zerolinewidth=1)
    pca_fig.update_yaxes(title_text=f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', row=1, col=2, showgrid=True, gridcolor='lightgray',
                         zeroline=True, zerolinecolor='black', zerolinewidth=1)
    
    # === COMPARE CORRECTIONS 2x2 ===
    # Use eval_columns from config (maps display_name -> column_name)
    eval_columns = config.get('eval_columns', DEFAULT_CONFIG['eval_columns'])
    # Convert to (column_name, display_name) format, max 4 for 2x2 grid
    gen_variants = [(col, name.replace('_', ' ').title()) for name, col in list(eval_columns.items())[:4]]
    
    compare_fig = make_subplots(rows=2, cols=2, subplot_titles=[v[1] for v in gen_variants],
                                 vertical_spacing=0.25, horizontal_spacing=0.1)
    
    compare_outlier_info = []  # Collect outlier words for each subplot
    
    for idx, (gen_col_name, label) in enumerate(gen_variants):
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        if gen_col_name in df.columns:
            gen_vals = df[gen_col_name].values
            valid_mask = ~np.isnan(gen_vals)
            
            if valid_mask.sum() > 0:
                # Create hover text for this variant (uses gen_vals for x)
                compare_hover = []
                
                for i in range(len(gen_vals)):
                    parts = [f"Gen={gen_vals[i]:.2f}", f"Val={val_scores[i]:.2f}"]
                    if has_noun2:
                        parts.append(f"Item={df['noun2'].iloc[i]}")
                    compare_hover.append(" | ".join(parts))
                compare_hover = np.array(compare_hover)
                compare_customdata = customdata if customdata is not None else None
                
                # Compute outliers for this variant (reuse outlier_method from main plot)
                X_var = np.column_stack([gen_vals[valid_mask], val_scores[valid_mask]])
                scaler_var = StandardScaler()
                X_var_std = scaler_var.fit_transform(X_var)
                
                if outlier_method == 'identity':
                    var_outlier_scores = np.abs(X_var_std[:, 1] - X_var_std[:, 0]) / np.sqrt(2)
                    var_type1_scores = None
                    var_type2_scores = None
                elif outlier_method == 'kendall':
                    gen_valid = gen_vals[valid_mask]
                    val_valid = val_scores[valid_mask]
                    n_var = len(gen_valid)
                    var_type1_scores = np.zeros(n_var)  # bottom_right: x>a but y<b
                    var_type2_scores = np.zeros(n_var)  # top_left: x<a but y>b
                    for i in range(n_var):
                        type1_count = 0
                        type2_count = 0
                        for j in range(n_var):
                            if i != j:
                                x_diff = gen_valid[i] - gen_valid[j]
                                y_diff = val_valid[i] - val_valid[j]
                                if x_diff > 0 and y_diff < 0:
                                    type1_count += 1
                                elif x_diff < 0 and y_diff > 0:
                                    type2_count += 1
                        var_type1_scores[i] = type1_count / (n_var - 1)
                        var_type2_scores[i] = type2_count / (n_var - 1)
                    var_outlier_scores = var_type1_scores + var_type2_scores
                
                # Get indices within valid_mask subset, then map back to original indices
                valid_indices = np.where(valid_mask)[0]
                n_outliers = min(40, len(valid_indices))
                sorted_by_score = np.argsort(var_outlier_scores)[::-1][:n_outliers]
                var_outlier_indices = valid_indices[sorted_by_score]
                var_outlier_set = set(var_outlier_indices)
                var_non_outlier_mask = np.array([i not in var_outlier_set for i in range(len(labels))])
                var_outlier_colors = [POS_OUTLIER_COLOR if labels[i] == 1 else NEG_OUTLIER_COLOR for i in var_outlier_indices]
                
                # Collect outlier words split by type (for annotation below plot)
                if has_noun2 and outlier_method == 'kendall':
                    top_left_words = []
                    bottom_right_words = []
                    for local_idx in sorted_by_score:
                        orig_idx = valid_indices[local_idx]
                        word = df['noun2'].iloc[orig_idx]
                        is_pos = labels[orig_idx] == 1
                        if var_type1_scores[local_idx] >= var_type2_scores[local_idx]:
                            bottom_right_words.append((var_type1_scores[local_idx], word, is_pos))
                        else:
                            top_left_words.append((var_type2_scores[local_idx], word, is_pos))
                    top_left_words.sort(reverse=True, key=lambda x: x[0])
                    bottom_right_words.sort(reverse=True, key=lambda x: x[0])
                    compare_outlier_info.append((idx,
                        [(w, p) for _, w, p in top_left_words],
                        [(w, p) for _, w, p in bottom_right_words]))
                
                # Add scatter traces (excluding outliers)
                compare_fig.add_trace(
                    go.Scatter(
                        x=gen_vals[pos_mask & valid_mask & var_non_outlier_mask], 
                        y=val_scores[pos_mask & valid_mask & var_non_outlier_mask],
                        mode='markers', marker=dict(color=POS_CLASS_COLOR, size=6, opacity=0.5),
                        showlegend=(idx == 0), name='Positive',
                        hovertext=compare_hover[pos_mask & valid_mask & var_non_outlier_mask], hoverinfo='text',
                        customdata=compare_customdata[pos_mask & valid_mask & var_non_outlier_mask] if compare_customdata is not None else None
                    ),
                    row=row, col=col
                )
                compare_fig.add_trace(
                    go.Scatter(
                        x=gen_vals[neg_mask & valid_mask & var_non_outlier_mask], 
                        y=val_scores[neg_mask & valid_mask & var_non_outlier_mask],
                        mode='markers', marker=dict(color=NEG_CLASS_COLOR, size=6, opacity=0.5),
                        showlegend=(idx == 0), name='Negative',
                        hovertext=compare_hover[neg_mask & valid_mask & var_non_outlier_mask], hoverinfo='text',
                        customdata=compare_customdata[neg_mask & valid_mask & var_non_outlier_mask] if compare_customdata is not None else None
                    ),
                    row=row, col=col
                )
                
                # Add outliers as X markers
                compare_fig.add_trace(
                    go.Scatter(
                        x=gen_vals[var_outlier_indices], y=val_scores[var_outlier_indices],
                        mode='markers', marker=dict(symbol='x', size=8, color=var_outlier_colors),
                        showlegend=False,
                        hovertext=compare_hover[var_outlier_indices], hoverinfo='text',
                        customdata=compare_customdata[var_outlier_indices] if compare_customdata is not None else None
                    ),
                    row=row, col=col
                )
                
                # Add y=x line (in original space) - line through mean with slope = std_val/std_gen
                mean_gen = scaler_var.mean_[0]
                mean_val = scaler_var.mean_[1]
                std_gen = scaler_var.scale_[0]
                std_val = scaler_var.scale_[1]
                gen_min = gen_vals[valid_mask].min()
                gen_max = gen_vals[valid_mask].max()
                # y=x in standardized space means: (val - mean_val)/std_val = (gen - mean_gen)/std_gen
                # So: val = mean_val + std_val * (gen - mean_gen) / std_gen
                line_gen = np.array([gen_min, gen_max])
                line_val = mean_val + std_val * (line_gen - mean_gen) / std_gen
                compare_fig.add_trace(
                    go.Scatter(x=line_gen, y=line_val, mode='lines',
                               line=dict(color='gray', dash='dot', width=2),
                               showlegend=False, hoverinfo='skip'),
                    row=row, col=col
                )
                
                compare_fig.add_hline(y=threshold, line=dict(color='red', dash='dash'), row=row, col=col)
                
                # Compute metrics for this variant
                m = compute_metrics(gen_vals, val_scores, labels, metric_type)
                metrics_text = (f"corr={m['corr']*100:.1f}\ncorr-pos={m['corr_pos']*100:.1f}\n"
                               f"corr-neg={m['corr_neg']*100:.1f}\nAcc={m['acc']*100:.1f}\n"
                               f"Val ROC={m['val_roc']*100:.1f}\nGen ROC={m['gen_roc']*100:.1f}")
                
                compare_fig.add_annotation(
                    x=0.02, y=0.98,
                    xref=f'x{idx+1 if idx > 0 else ""} domain',
                    yref=f'y{idx+1 if idx > 0 else ""} domain',
                    text=metrics_text, showarrow=False, font=dict(size=9),
                    bgcolor='white', bordercolor='gray', align='left',
                    xanchor='left', yanchor='top'
                )
    
    compare_fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    compare_fig.update_yaxes(showgrid=True, gridcolor='lightgray')
    
    # Add outlier words annotation below each subplot (only for hypernym tasks)
    if task and task.startswith('hypernym-'):
        def format_colored_words(word_list, max_chars=120):
            # Build lines with max_chars characters each (by word length, not HTML)
            lines = []
            current_line = []
            current_len = 0
            for word, is_pos in word_list:
                word_len = len(word) + 2  # +2 for ", "
                if current_len + word_len > max_chars and current_line:
                    lines.append(', '.join(current_line))
                    current_line = []
                    current_len = 0
                color = 'red' if is_pos else 'blue'
                current_line.append(f'<span style="color:{color}">{word}</span>')
                current_len += word_len
            if current_line:
                lines.append(', '.join(current_line))
            return '<br>'.join(lines)
        
        for item in compare_outlier_info:
            idx, top_left_words, bottom_right_words = item
            parts = []
            if top_left_words:
                parts.append(f"<b>Top left:</b><br>{format_colored_words(top_left_words)}")
            if bottom_right_words:
                parts.append(f"<b>Bottom right:</b><br>{format_colored_words(bottom_right_words)}")
            if parts:
                words_text = '<br>'.join(parts)
                x_ref = 'x domain' if idx == 0 else f'x{idx+1} domain'
                y_ref = 'y domain' if idx == 0 else f'y{idx+1} domain'
                compare_fig.add_annotation(
                    x=0.5, y=-0.15, xref=x_ref, yref=y_ref,
                    text=words_text, showarrow=False, font=dict(size=10),
                    align='left', xanchor='center', yanchor='top'
                )
    
    compare_fig.update_layout(title='Compare Score Corrections', paper_bgcolor='white', plot_bgcolor='white', 
                               height=1100, margin=dict(b=150))
    
    file_status = f"Loaded: {Path(csv_path).name}"
    return file_status, stats_text, main_fig, faceted_fig, pca_fig, compare_fig


@app.callback(
    [Output('response-panel-main', 'children'),
     Output('response-panel-faceted', 'children'),
     Output('response-panel-pca', 'children'),
     Output('response-panel-compare', 'children')],
    [Input('main-scatter', 'clickData'),
     Input('faceted-plot', 'clickData'),
     Input('pca-plot', 'clickData'),
     Input('compare-plot', 'clickData')]
)
def update_response_panels(main_click, faceted_click, pca_click, compare_click):
    """Show full prompt/response text on click for each graph."""
    def render_panel(click_data, empty_text):
        if not click_data or 'points' not in click_data or not click_data['points']:
            return empty_text
        point = click_data['points'][0]
        customdata = point.get('customdata')
        if not customdata or len(customdata) < 2:
            return "No prompt/response text available for this point."
        prompt_text = customdata[0] or ""
        response_text = customdata[1] or ""
        return html.Div([
            html.Div("Prompt:", style={'fontWeight': 'bold', 'marginBottom': '4px'}),
            html.Pre(prompt_text, style={'whiteSpace': 'pre-wrap', 'marginTop': '0', 'marginBottom': '12px'}),
            html.Div("Response:", style={'fontWeight': 'bold', 'marginBottom': '4px'}),
            html.Pre(response_text, style={'whiteSpace': 'pre-wrap', 'marginTop': '0'})
        ])
    
    return (
        render_panel(main_click, "Click a point in the main scatter to view the full prompt/response here."),
        render_panel(faceted_click, "Click a point in the faceted plot to view the full prompt/response here."),
        render_panel(pca_click, "Click a point in the PCA plots to view the full prompt/response here."),
        render_panel(compare_click, "Click a point in the compare plot to view the full prompt/response here.")
    )


@app.callback(
    Output('all-heatmaps-container', 'children'),
    [Input('config-store', 'data'),
     Input('files-store', 'data')]
)
def generate_all_heatmaps(config, files_info):
    """Generate all heatmaps based on config.
    
    Creates two separate heatmaps per task/aggregate:
    - One for V2G (d2g direction): rows are training variants (V2G, +tc, +lenorm, etc.)
    - One for G2V (g2d direction): rows are training variants (G2V, +tc, +lenorm, etc.)
    """
    if not config or not files_info:
        return html.Div("No data loaded", style={'color': '#999', 'textAlign': 'center', 'padding': '20px'})
    
    children = []
    
    # Get all unique tasks and splits
    all_tasks = sorted(set(f['task'] for f in files_info))
    all_splits = sorted(set(f['split'] for f in files_info))
    
    training_rows = get_training_variant_rows(files_info, config)
    eval_columns = config.get('eval_columns', {})
    eval_cols = list(eval_columns.keys())
    metrics_list = config.get('metrics', ['Val Acc', 'Val ROC', 'Gen ROC', 'Correlation', 'Corr-Pos', 'Corr-Neg'])
    aggregation_groups = config.get('aggregation_groups', {})
    
    # Direction display mapping from config
    model_detection = config.get('model_detection', {})
    direction_patterns = model_detection.get('direction_patterns', {'d2g': '_d2g_', 'g2d': '_g2d_'})
    direction_display = model_detection.get('direction_display', {'d2g': 'V2G', 'g2d': 'G2V'})
    directions = [(dir_key, direction_display.get(dir_key, dir_key.upper())) 
                  for dir_key in direction_patterns.keys()]
    
    for split in all_splits:
        split_label = split.upper()
        split_color = '#2e7d32' if split == 'test' else '#c62828'
        split_bg = '#e8f5e9' if split == 'test' else '#ffebee'
        
        # Collect all content for this split in a separate list
        split_children = []
        
        split_children.append(html.H2(
            f'{"🧪" if split == "test" else "🏋️"} {split_label} SET RESULTS',
            style={'marginTop': '0px', 'marginBottom': '20px', 'color': split_color,
                   'borderBottom': f'3px solid {split_color}', 'paddingBottom': '10px',
                   'padding': '15px', 'borderRadius': '8px'}
        ))
        
        # Load heatmap data for all tasks in this split (by direction)
        all_heatmap_data_by_dir = {}
        for task in all_tasks:
            try:
                dir_data, _, _ = discover_heatmap_data_by_direction(task, split, files_info, config)
                all_heatmap_data_by_dir[task] = dir_data
            except Exception as e:
                print(f"Error loading heatmap data for {task}/{split}: {e}")
        
        # === AGGREGATED SECTION ===
        for group_name, pattern in aggregation_groups.items():
            tasks_in_group = expand_aggregation_pattern(pattern, all_tasks)
            tasks_with_data = [t for t in tasks_in_group if t in all_heatmap_data_by_dir]
            
            if len(tasks_with_data) > 1:
                try:
                    split_children.append(html.H3(
                        f'📊 {group_name} - {split_label} (Mean across {len(tasks_with_data)} tasks)',
                        style={'marginTop': '20px', 'marginBottom': '10px', 'color': '#1a5f7a',
                               'borderBottom': '2px solid #1a5f7a', 'paddingBottom': '10px'}
                    ))
                    
                    # Create two heatmaps: one for V2G, one for G2V
                    split_children.append(html.H4('Aggregated Heatmaps', style={'marginTop': '15px', 'color': '#333'}))
                    for direction, dir_label in directions:
                        fig_agg = create_aggregated_direction_heatmap(
                            all_heatmap_data_by_dir, tasks_with_data, training_rows, eval_cols,
                            f'{dir_label} Models (Mean across datasets)', metrics_list, direction, dir_label
                        )
                        if fig_agg is not None:
                            split_children.append(dcc.Graph(figure=fig_agg, style={'height': '280px'}))
                    
                    # Bar plots for ALL metrics - one per direction (like hypernym_dashboard.py)
                    split_children.append(html.H4('Bar Plots with Standard Error', style={'marginTop': '25px', 'color': '#333'}))
                    for metric in metrics_list:
                        split_children.append(html.H5(f'{metric}', style={'marginTop': '15px', 'color': '#555'}))
                        for direction, dir_label in directions:
                            fig_bar = create_direction_bar_plot(
                                all_heatmap_data_by_dir, tasks_with_data, training_rows, eval_cols,
                                metric, direction, dir_label
                            )
                            split_children.append(dcc.Graph(figure=fig_bar, style={'height': '320px'}))
                
                except Exception as e:
                    split_children.append(html.Div(
                        f"⚠️ Error generating aggregated view for {group_name}: {e}",
                        style={'color': '#c62828', 'padding': '10px', 'backgroundColor': '#ffebee', 'borderRadius': '5px'}
                    ))
        
        # === PER-TASK SECTION ===
        split_children.append(html.H3(
            f'📋 Per-Task Results - {split_label}',
            style={'marginTop': '30px', 'marginBottom': '10px', 'color': '#1a5f7a',
                   'borderBottom': '2px solid #1a5f7a', 'paddingBottom': '10px'}
        ))
        
        for task in all_tasks:
            if task not in all_heatmap_data_by_dir:
                continue
            
            split_children.append(html.H4(
                f'{task}',
                style={'marginTop': '15px', 'marginBottom': '5px', 'color': '#333',
                       'borderBottom': '1px solid #ddd', 'paddingBottom': '5px'}
            ))
            
            try:
                # Create two heatmaps: one for V2G, one for G2V
                dir_data = all_heatmap_data_by_dir[task]
                for direction, dir_label in directions:
                    fig = create_direction_heatmap_figure(
                        dir_data.get(direction, {}), training_rows, eval_cols,
                        f'{dir_label} Models', metrics_list, dir_label
                    )
                    if fig is not None:
                        split_children.append(dcc.Graph(figure=fig, style={'height': '280px', 'marginTop': '0px'}))
            except Exception as e:
                split_children.append(html.Div(
                    f"⚠️ Error generating heatmap for {task}: {e}",
                    style={'color': '#c62828', 'padding': '5px'}
                ))
        
        # Wrap split content - only TRAIN gets background color
        if split == 'train':
            children.append(html.Div(
                split_children,
                style={'backgroundColor': split_bg, 'padding': '20px', 'borderRadius': '10px',
                       'marginTop': '20px', 'marginBottom': '20px'}
            ))
        else:
            # TEST section - no background wrapper, just add children directly
            children.extend(split_children)
    
    return children


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print(f"Starting dashboard on port {PORT}")
    print(f"Access via: http://localhost:{PORT}")
    print(f"Config file location: {CONFIG_FILE}")
    print(f"Make sure to set up SSH port forwarding: ssh -L {PORT}:localhost:{PORT} <host>")
    app.run(host='0.0.0.0', port=PORT, debug=False)
