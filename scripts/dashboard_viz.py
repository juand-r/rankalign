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
        'base_pattern': r'^scores_gemma-2-2b_',  # base model (not finetuned)
        'finetuned_marker': 'v5-',  # marker for finetuned models
        'direction_patterns': {
            'd2g': '_d2g_',
            'g2d': '_g2d_'
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
    'aggregation_groups': {
        'All Hypernym': 'hypernym-*'  # pattern-based: aggregate all matching tasks
    },
    'metrics': ['Accuracy', 'Val ROC', 'Gen ROC', 'Correlation', 'Corr-Pos', 'Corr-Neg'],
    'eval_columns': {
        'raw': 'gen_score',
        'tc': 'gen_score_typcorr',
        'lenorm': 'gen_score_lenorm',
        'tc+lenorm': 'gen_score_typcorr_lenorm'
    }
}


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
    
    # Detect tasks (look for common patterns like hypernym-X, trivia-qa, etc.)
    tasks_found = set()
    task_patterns_found = {}
    
    # Try common task patterns
    common_patterns = [
        (r'hypernym-([a-zA-Z]+)', 'hypernym'),
        (r'trivia-qa', 'trivia-qa'),
        (r'swords', 'swords'),
        (r'lambada', 'lambada'),
        (r'ifeval', 'ifeval'),
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
    
    # If hypernym tasks found, use that pattern
    if any(t.startswith('hypernym-') for t in tasks_found):
        config['task_pattern'] = r'hypernym-([a-zA-Z]+)'
    
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
    
    # Check for base models (not finetuned)
    has_base = any(re.match(r'^scores_gemma-2-2b_', f) for f in filenames)
    
    # Check for finetuned models
    has_finetuned = any('v5-' in f for f in filenames)
    
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
        'base_pattern': r'^scores_gemma-2-2b_' if has_base else None,
        'finetuned_marker': 'v5-' if has_finetuned else None,
        'direction_patterns': {},
        'variant_patterns': {}
    }
    
    if has_d2g:
        config['model_detection']['direction_patterns']['d2g'] = '_d2g_'
    if has_g2d:
        config['model_detection']['direction_patterns']['g2d'] = '_g2d_'
    
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
    
    # Build aggregation groups based on tasks found
    config['aggregation_groups'] = {}
    
    # Group by task type prefix
    task_prefixes = set()
    for task in tasks_found:
        if '-' in task:
            prefix = task.split('-')[0]
            task_prefixes.add(prefix)
    
    for prefix in task_prefixes:
        matching_tasks = [t for t in tasks_found if t.startswith(f"{prefix}-")]
        if len(matching_tasks) > 1:
            config['aggregation_groups'][f'All {prefix.capitalize()}'] = f"{prefix}-*"
    
    # Read one CSV to detect label column
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
        pass  # Use defaults
    
    info_msg = f"Auto-detected from {len(csv_files)} files: {len(tasks_found)} tasks, splits: {list(config['split_patterns'].keys())}"
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
    """Parse filename to extract task, split, model info based on config."""
    name = csv_file.stem
    
    # Extract task/dataset
    task_pattern = config.get('task_pattern', r'hypernym-([a-zA-Z]+)')
    task_match = re.search(task_pattern, name)
    if task_match:
        if task_match.groups():
            # Pattern has capture group - use it as dataset name
            base_task = task_pattern.split('(')[0].rstrip('-').rstrip('_')
            if not base_task:
                base_task = 'task'
            dataset = task_match.group(1)
            task = f"{base_task}-{dataset}" if base_task else dataset
        else:
            task = task_match.group(0)
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
    
    # Extract model type
    model_detection = config.get('model_detection', {})
    model_type = determine_model_type(name, model_detection)
    
    display_name = f'{task} | {split} | {model_type}'
    
    return {
        'path': str(csv_file),
        'display': display_name,
        'task': task,
        'dataset': dataset,
        'split': split,
        'model_type': model_type
    }


def determine_model_type(filename, model_detection):
    """Determine model type from filename based on detection config."""
    base_pattern = model_detection.get('base_pattern')
    finetuned_marker = model_detection.get('finetuned_marker')
    direction_patterns = model_detection.get('direction_patterns', {})
    variant_patterns = model_detection.get('variant_patterns', {})
    
    # Check if base model
    is_base = False
    if base_pattern:
        is_base = bool(re.match(base_pattern, filename))
    
    # Check if finetuned
    is_finetuned = finetuned_marker and finetuned_marker in filename
    
    if is_base and not is_finetuned:
        return 'Base'
    
    if not is_finetuned:
        return 'Base'
    
    # Determine direction
    # Map internal names to display names: d2g -> V2G, g2d -> G2V
    direction_display = {'d2g': 'V2G', 'g2d': 'G2V'}
    direction = ''
    for dir_name, pattern in direction_patterns.items():
        if pattern in filename:
            direction = direction_display.get(dir_name, dir_name.upper())
            break
    
    # Determine variants by checking patterns
    # Check combined patterns first (more specific), then individual ones
    has_tc_online_lenorm = '_tc-online_lenorm_full-completion' in filename
    has_typcorr_lenorm = '_typcorr_lenorm_full-completion' in filename
    has_tc_online = '_tc-online_full-completion' in filename and not has_tc_online_lenorm
    has_typcorr = '_typcorr_full-completion' in filename and not has_typcorr_lenorm and '_tc-online' not in filename
    has_lenorm = '_lenorm_full-completion' in filename and not has_typcorr_lenorm and not has_tc_online_lenorm and '_typcorr' not in filename and '_tc-online' not in filename
    
    # Check if vanilla (no corrections at all)
    is_vanilla = '_full-completion_' in filename and not any([
        has_tc_online_lenorm, has_typcorr_lenorm, has_tc_online, has_typcorr, has_lenorm
    ])
    
    # Build suffix
    if has_tc_online_lenorm:
        suffix = '+tco+lenorm'
    elif has_tc_online:
        suffix = '+tco'
    elif has_typcorr_lenorm:
        suffix = '+tc+lenorm'
    elif has_typcorr:
        suffix = '+tc'
    elif has_lenorm:
        suffix = '+lenorm'
    elif is_vanilla:
        suffix = ''  # vanilla rankalign, no suffix
    else:
        suffix = ''
    
    if direction:
        return f'{direction}{suffix}' if suffix else direction
    else:
        return f'Rankalign{suffix}' if suffix else 'Rankalign'


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
    """Get unique model types for heatmap rows."""
    model_types = set(f['model_type'] for f in files_info)
    
    # Sort with Base first, then alphabetically
    sorted_types = sorted(model_types, key=lambda x: (0 if x == 'Base' else 1, x))
    return sorted_types


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

def discover_heatmap_data(task, split, files_info, config):
    """Discover and load heatmap data for a task and split."""
    eval_columns = config.get('eval_columns', {
        'raw': 'gen_score',
        'tc': 'gen_score_typcorr',
        'lenorm': 'gen_score_lenorm',
        'tc+lenorm': 'gen_score_typcorr_lenorm'
    })
    
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
        'Accuracy': 'acc', 'Val ROC': 'val_roc', 'Gen ROC': 'gen_roc',
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
        height=max(200, 50 + 25 * len(model_rows)),
        margin=dict(l=100, r=20, t=50, b=30),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig


def create_aggregated_heatmap(all_heatmap_data, tasks, model_rows, eval_cols, title, metrics_list):
    """Create aggregated heatmap showing mean across tasks."""
    fig = make_subplots(rows=1, cols=len(metrics_list), subplot_titles=metrics_list,
                        horizontal_spacing=0.03)
    
    metric_key_map = {
        'Accuracy': 'acc', 'Val ROC': 'val_roc', 'Gen ROC': 'gen_roc',
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
        height=max(200, 50 + 25 * len(model_rows)),
        margin=dict(l=100, r=20, t=50, b=30),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig


def create_aggregated_bar_plot(all_heatmap_data, tasks, model_rows, eval_cols, metric, title):
    """Create bar plot with standard error for a single metric."""
    metric_key_map = {
        'Accuracy': 'acc', 'Val ROC': 'val_roc', 'Gen ROC': 'gen_roc',
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
        yaxis_config = dict(title=metric, showgrid=True, gridcolor='lightgray', gridwidth=1, dtick=20)
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
                           style={'padding': '10px 20px', 'fontSize': '14px',
                                  'backgroundColor': '#FF9800', 'color': 'white', 'border': 'none',
                                  'borderRadius': '5px', 'cursor': 'pointer'}),
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
            html.H1('📊 Visualization Dashboard', style={'textAlign': 'center', 'color': '#333', 'display': 'inline-block'}),
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
        
        # Faceted by strategy
        html.Details([
            html.Summary('📊 Faceted View by Strategy', style={'cursor': 'pointer', 'fontWeight': 'bold'}),
            dcc.Graph(id='faceted-plot', style={'height': '600px'})
        ], style={'margin': '20px'}),
        
        # PCA/Standardized plot
        html.Details([
            html.Summary('🔬 Standardized Scores & PCA Analysis', style={'cursor': 'pointer', 'fontWeight': 'bold'}),
            dcc.Graph(id='pca-plot', style={'height': '500px'})
        ], style={'margin': '20px'}),
        
        # Compare corrections 2x2
        html.Details([
            html.Summary('🔄 Compare Score Corrections (2x2)', style={'cursor': 'pointer', 'fontWeight': 'bold'}),
            dcc.Graph(id='compare-plot', style={'height': '800px'})
        ], style={'margin': '20px'}),
        
        # Heatmaps section
        html.Details([
            html.Summary('🔥 Heatmaps (All Tasks)', style={'cursor': 'pointer', 'fontWeight': 'bold'}),
            html.Div(id='all-heatmaps-container')
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
     State('config-eval-cols', 'value')],
    prevent_initial_call=True
)
def save_config(n_clicks, outputs_dir, task_pattern, split_patterns, label_col, 
                label_map, model_detection, aggregation, eval_cols):
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
            'eval_columns': json.loads(eval_cols) if eval_cols else {}
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
     Input('back-to-config-btn', 'n_clicks')],
    [State('config-outputs-dir', 'value'),
     State('config-task-pattern', 'value'),
     State('config-split-patterns', 'value'),
     State('config-label-col', 'value'),
     State('config-label-map', 'value'),
     State('config-model-detection', 'value'),
     State('config-aggregation', 'value'),
     State('config-eval-cols', 'value')],
    prevent_initial_call=True
)
def toggle_pages(load_clicks, back_clicks, outputs_dir, task_pattern, split_patterns,
                 label_col, label_map, model_detection, aggregation, eval_cols):
    """Toggle between config page and viz page."""
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    config_visible = {'padding': '40px', 'backgroundColor': 'white', 'minHeight': '100vh'}
    config_hidden = {'display': 'none'}
    viz_visible = {'padding': '20px', 'backgroundColor': 'white', 'minHeight': '100vh'}
    viz_hidden = {'display': 'none'}
    
    if button_id == 'back-to-config-btn':
        return (config_visible, viz_hidden, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update)
    
    # Load dashboard
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
            'metrics': DEFAULT_CONFIG['metrics']
        }
        
        # Discover files
        files_info = discover_scores_files(config)
        
        if not files_info:
            raise PreventUpdate
        
        # Build dropdown options
        all_tasks = sorted(set(f['task'] for f in files_info))
        all_splits = sorted(set(f['split'] for f in files_info), reverse=True)
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
    
    main_fig.add_trace(
        go.Scatter(x=gen_scores[pos_mask], y=val_scores[pos_mask],
                   mode='markers', marker=dict(color='orange', size=8, opacity=0.6),
                   name='Positive', legendgroup='pos'),
        row=2, col=1
    )
    main_fig.add_trace(
        go.Scatter(x=gen_scores[neg_mask], y=val_scores[neg_mask],
                   mode='markers', marker=dict(color='blue', size=8, opacity=0.6),
                   name='Negative', legendgroup='neg'),
        row=2, col=1
    )
    
    main_fig.add_hline(y=threshold, line=dict(color='red', dash='dash', width=2), row=2, col=1)
    
    # Outliers
    X = np.column_stack([gen_scores, val_scores])
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    distances = np.sqrt(X_std[:, 0]**2 + X_std[:, 1]**2)
    outlier_indices = np.argsort(distances)[-40:]
    
    outlier_colors = ['orange' if labels[i] == 1 else 'purple' for i in outlier_indices]
    
    # Try to get display columns for outlier labels
    display_cols = ['noun1', 'noun2']  # Default
    outlier_texts = []
    for i in outlier_indices:
        if 'noun1' in df.columns and 'noun2' in df.columns:
            outlier_texts.append(f"{df['noun1'].iloc[i][:6]}/{df['noun2'].iloc[i][:6]}")
        else:
            outlier_texts.append('')
    
    main_fig.add_trace(
        go.Scatter(
            x=gen_scores[outlier_indices], y=val_scores[outlier_indices],
            mode='markers+text',
            marker=dict(symbol='x', size=12, color=outlier_colors, line=dict(width=2)),
            text=outlier_texts,
            textposition='top right', textfont=dict(size=8),
            name='Outliers', showlegend=False
        ),
        row=2, col=1
    )
    
    # Histograms
    main_fig.add_trace(go.Histogram(x=gen_scores[pos_mask], marker_color='orange', opacity=0.6, showlegend=False), row=1, col=1)
    main_fig.add_trace(go.Histogram(x=gen_scores[neg_mask], marker_color='blue', opacity=0.6, showlegend=False), row=1, col=1)
    main_fig.add_trace(go.Histogram(y=val_scores[pos_mask], marker_color='orange', opacity=0.6, showlegend=False), row=2, col=2)
    main_fig.add_trace(go.Histogram(y=val_scores[neg_mask], marker_color='blue', opacity=0.6, showlegend=False), row=2, col=2)
    
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
        
        faceted_fig.add_trace(
            go.Scatter(x=gen_scores[strat_pos], y=val_scores[strat_pos],
                       mode='markers', marker=dict(color='orange', size=6, opacity=0.6),
                       showlegend=(idx == 0), name='Positive'),
            row=row, col=col
        )
        faceted_fig.add_trace(
            go.Scatter(x=gen_scores[strat_neg], y=val_scores[strat_neg],
                       mode='markers', marker=dict(color='blue', size=6, opacity=0.6),
                       showlegend=(idx == 0), name='Negative'),
            row=row, col=col
        )
        faceted_fig.add_hline(y=threshold, line=dict(color='red', dash='dash'), row=row, col=col)
    
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            faceted_fig.update_xaxes(range=[x_min - x_pad, x_max + x_pad], showgrid=True, gridcolor='lightgray', row=r, col=c)
            faceted_fig.update_yaxes(range=[y_min - y_pad, y_max + y_pad], showgrid=True, gridcolor='lightgray', row=r, col=c)
    
    faceted_fig.update_layout(title='Faceted by Strategy', paper_bgcolor='white', plot_bgcolor='white', height=400 * n_rows)
    
    # === PCA PLOT ===
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)
    
    pca_fig = make_subplots(rows=1, cols=2, subplot_titles=['Standardized Scores', 'PCA'])
    
    pca_fig.add_trace(go.Scatter(x=X_std[pos_mask, 0], y=X_std[pos_mask, 1], mode='markers', 
                                  marker=dict(color='orange', size=6, opacity=0.5), name='Positive'), row=1, col=1)
    pca_fig.add_trace(go.Scatter(x=X_std[neg_mask, 0], y=X_std[neg_mask, 1], mode='markers',
                                  marker=dict(color='blue', size=6, opacity=0.5), name='Negative'), row=1, col=1)
    pca_fig.add_trace(go.Scatter(x=X_pca[pos_mask, 0], y=X_pca[pos_mask, 1], mode='markers',
                                  marker=dict(color='orange', size=6, opacity=0.5), showlegend=False), row=1, col=2)
    pca_fig.add_trace(go.Scatter(x=X_pca[neg_mask, 0], y=X_pca[neg_mask, 1], mode='markers',
                                  marker=dict(color='blue', size=6, opacity=0.5), showlegend=False), row=1, col=2)
    
    pca_fig.update_layout(paper_bgcolor='white', plot_bgcolor='white')
    pca_fig.update_xaxes(title_text='Generator (std)', row=1, col=1, showgrid=True, gridcolor='lightgray')
    pca_fig.update_yaxes(title_text='Validator (std)', row=1, col=1, showgrid=True, gridcolor='lightgray')
    pca_fig.update_xaxes(title_text=f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', row=1, col=2, showgrid=True, gridcolor='lightgray')
    pca_fig.update_yaxes(title_text=f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', row=1, col=2, showgrid=True, gridcolor='lightgray')
    
    # === COMPARE CORRECTIONS 2x2 ===
    eval_columns = config.get('eval_columns', {})
    gen_variants = [(col, name) for name, col in eval_columns.items()][:4]  # Max 4 for 2x2
    
    if len(gen_variants) < 4:
        gen_variants = [
            ('gen_score', 'Raw'),
            ('gen_score_typcorr', 'Typicality Corrected'),
            ('gen_score_lenorm', 'Length Normalized'),
            ('gen_score_typcorr_lenorm', 'Typcorr + Lenorm'),
        ]
    
    compare_fig = make_subplots(rows=2, cols=2, subplot_titles=[v[1] for v in gen_variants])
    
    for idx, (gen_col_name, label) in enumerate(gen_variants):
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        if gen_col_name in df.columns:
            gen_vals = df[gen_col_name].values
            valid_mask = ~np.isnan(gen_vals)
            
            if valid_mask.sum() > 0:
                compare_fig.add_trace(
                    go.Scatter(x=gen_vals[pos_mask & valid_mask], y=val_scores[pos_mask & valid_mask],
                               mode='markers', marker=dict(color='orange', size=6, opacity=0.5),
                               showlegend=(idx == 0), name='Positive'),
                    row=row, col=col
                )
                compare_fig.add_trace(
                    go.Scatter(x=gen_vals[neg_mask & valid_mask], y=val_scores[neg_mask & valid_mask],
                               mode='markers', marker=dict(color='blue', size=6, opacity=0.5),
                               showlegend=(idx == 0), name='Negative'),
                    row=row, col=col
                )
                compare_fig.add_hline(y=threshold, line=dict(color='red', dash='dash'), row=row, col=col)
    
    compare_fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    compare_fig.update_yaxes(showgrid=True, gridcolor='lightgray')
    compare_fig.update_layout(title='Compare Score Corrections', paper_bgcolor='white', plot_bgcolor='white')
    
    file_status = f"Loaded: {Path(csv_path).name}"
    return file_status, stats_text, main_fig, faceted_fig, pca_fig, compare_fig


@app.callback(
    Output('all-heatmaps-container', 'children'),
    [Input('config-store', 'data'),
     Input('files-store', 'data')]
)
def generate_all_heatmaps(config, files_info):
    """Generate all heatmaps based on config."""
    if not config or not files_info:
        return html.Div("No data loaded", style={'color': '#999', 'textAlign': 'center', 'padding': '20px'})
    
    children = []
    
    # Get all unique tasks and splits
    all_tasks = sorted(set(f['task'] for f in files_info))
    all_splits = sorted(set(f['split'] for f in files_info), reverse=True)
    
    model_rows = get_model_rows(files_info)
    eval_columns = config.get('eval_columns', {})
    eval_cols = list(eval_columns.keys())
    metrics_list = config.get('metrics', ['Accuracy', 'Val ROC', 'Gen ROC', 'Correlation', 'Corr-Pos', 'Corr-Neg'])
    aggregation_groups = config.get('aggregation_groups', {})
    
    for split in all_splits:
        split_label = split.upper()
        split_color = '#2e7d32' if split == 'test' else '#c62828'
        split_bg = '#e8f5e9' if split == 'test' else '#ffebee'
        
        children.append(html.H2(
            f'{"🧪" if split == "test" else "🏋️"} {split_label} SET RESULTS',
            style={'marginTop': '20px', 'marginBottom': '20px', 'color': split_color,
                   'borderBottom': f'3px solid {split_color}', 'paddingBottom': '10px',
                   'backgroundColor': split_bg, 'padding': '15px', 'borderRadius': '8px'}
        ))
        
        # Load heatmap data for all tasks in this split
        all_heatmap_data = {}
        for task in all_tasks:
            try:
                hm_data, _, _ = discover_heatmap_data(task, split, files_info, config)
                all_heatmap_data[task] = hm_data
            except Exception as e:
                print(f"Error loading heatmap data for {task}/{split}: {e}")
        
        # === AGGREGATED SECTION ===
        for group_name, pattern in aggregation_groups.items():
            tasks_in_group = expand_aggregation_pattern(pattern, all_tasks)
            tasks_with_data = [t for t in tasks_in_group if t in all_heatmap_data]
            
            if len(tasks_with_data) > 1:
                try:
                    children.append(html.H3(
                        f'📊 {group_name} - {split_label} (Mean across {len(tasks_with_data)} tasks)',
                        style={'marginTop': '20px', 'marginBottom': '10px', 'color': '#1a5f7a',
                               'borderBottom': '2px solid #1a5f7a', 'paddingBottom': '10px'}
                    ))
                    
                    # Aggregated heatmap
                    fig_agg = create_aggregated_heatmap(
                        all_heatmap_data, tasks_with_data, model_rows, eval_cols,
                        f'{group_name} (Mean)', metrics_list
                    )
                    children.append(dcc.Graph(figure=fig_agg, style={'height': '280px'}))
                    
                    # Bar plots for key metrics
                    for metric in ['Accuracy', 'Val ROC']:
                        if metric in metrics_list:
                            fig_bar = create_aggregated_bar_plot(
                                all_heatmap_data, tasks_with_data, model_rows, eval_cols,
                                metric, f'{group_name} - {metric}'
                            )
                            children.append(dcc.Graph(figure=fig_bar, style={'height': '320px'}))
                
                except Exception as e:
                    children.append(html.Div(
                        f"⚠️ Error generating aggregated view for {group_name}: {e}",
                        style={'color': '#c62828', 'padding': '10px', 'backgroundColor': '#ffebee', 'borderRadius': '5px'}
                    ))
        
        # === PER-TASK SECTION ===
        children.append(html.H3(
            f'📋 Per-Task Results - {split_label}',
            style={'marginTop': '30px', 'marginBottom': '10px', 'color': '#1a5f7a',
                   'borderBottom': '2px solid #1a5f7a', 'paddingBottom': '10px'}
        ))
        
        for task in all_tasks:
            if task not in all_heatmap_data:
                continue
            
            children.append(html.H4(
                f'{task}',
                style={'marginTop': '15px', 'marginBottom': '5px', 'color': '#333',
                       'borderBottom': '1px solid #ddd', 'paddingBottom': '5px'}
            ))
            
            try:
                fig = create_heatmap_figure(
                    all_heatmap_data[task], model_rows, eval_cols,
                    task, metrics_list
                )
                children.append(dcc.Graph(figure=fig, style={'height': '280px', 'marginTop': '0px'}))
            except Exception as e:
                children.append(html.Div(
                    f"⚠️ Error generating heatmap for {task}: {e}",
                    style={'color': '#c62828', 'padding': '5px'}
                ))
    
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
