#!/usr/bin/env python3
"""
Visualization dashboard for semi-supervised / label-only evaluation scores.

Row classification: each heatmap row comes from exactly ONE scores file.
Finetuned rows use a combinatorial scheme:
    {data_mode}-{training_mode}[-tcs|-tcsstep][-norm][-v]

  data_mode:     LO (label-only) or Semi (semi-supervised)
  training_mode: Pref, Comb, or SFT
  flags:         tcs (tc-self), tcsstep (tc-self-step), norm (lenorm), v (vallogodds)

Non-finetuned base model checkpoints add a single shared row label "Base" (optional, via base_pattern).

Supports hypernym, ambigqa, plausibleqa, and ifeval tasks.

Run with:
    cd /datastor1/jdr/gv-gap/rankalign/scripts
    source ~/venvs/venv_lexcons/bin/activate
    PORT=8890 python dashboard_viz_refactor-semi.py

Then access via SSH port forwarding:
    ssh -L 8890:localhost:8890 <your-host>

Open in browser: http://localhost:8890
"""

import os
import re
import json
import fnmatch
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, List, Tuple
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

from score_file_parsing import _extract_float, _extract_timestamp, _extract_split


# =============================================================================
# CONFIGURATION
# =============================================================================

PORT = int(os.environ.get('PORT', 8890))
CONFIG_DIR = Path(__file__).parent.parent / 'config'
CONFIG_FILE = CONFIG_DIR / 'dashboard_semi_config.json'
DEFAULT_OUTPUTS_DIR = Path(__file__).parent.parent / 'outputs'

DEFAULT_CONFIG = {
    'outputs_dir': str(DEFAULT_OUTPUTS_DIR),
    'file_pattern': 'scores_self-*.csv',
    'finetuned_pattern': r'^scores_self-v6-google_gemma-2-2b-delta',
    # Base (non-finetuned) eval files, e.g. scores_v6-google_gemma-2-2b_* (see base_scores_glob)
    'base_pattern': r'^scores_v6-google_gemma-2-2b_',
    'base_scores_glob': 'scores_v6-google_gemma-2-2b*.csv',
    'include_base_model_eval': True,
    'split_patterns': {
        'train': '_train_',
        'test': '_test_'
    },
    'label_column': 'gpt4_ground_truth',
    'label_map': {'yes': 1, 'no': 0},
    'gen_score_col': 'gen_score',
    'val_score_col': 'val_score',
    'aggregation_groups': {
        'All Hypernym': 'hypernym-*',
        'All AmbigQA': 'ambigqa-*',
        'All PlausibleQA': 'plausibleqa-*',
    },
    'metrics': ['Val Acc', 'Val ROC', 'Gen ROC', 'Correlation', 'Corr-Pos', 'Corr-Neg'],
    'eval_columns': {
        'raw': 'gen_score',
        'tc': 'gen_score_typcorr',
        'lenorm': 'gen_score_lenorm',
        'tc+lenorm': 'gen_score_typcorr_lenorm'
    },
    'visible_data_modes': ['LO', 'Semi'],
    'visible_modes': ['Comb', 'SFT', 'Pref'],
    'visible_flags': ['tcs', 'tcsstep', 'norm', 'v'],
    # Limit tasks in dropdowns / heatmaps (parsed task id, e.g. hypernym-foo). Empty = all.
    'task_filter_pattern': '',
}

# Visualization colors
POS_CLASS_COLOR = 'orangered'
NEG_CLASS_COLOR = 'blue'
POS_OUTLIER_COLOR = 'red'
NEG_OUTLIER_COLOR = 'purple'

METRIC_KEY_MAP = {
    'Val Acc': 'acc', 'Val ROC': 'val_roc', 'Gen ROC': 'gen_roc',
    'Correlation': 'corr', 'Corr-Pos': 'corr_pos', 'Corr-Neg': 'corr_neg'
}


# =============================================================================
# FILEINFO DATACLASS
# =============================================================================

@dataclass
class FileInfo:
    """Parsed representation of a semi-supervised scores CSV filename."""
    path: str
    filename: str
    task: str               # eval task, e.g. "hypernym-bananas", "plausibleqa-nq_1324"
    dataset: str            # part after family prefix, e.g. "bananas", "nq_1324"
    split: str              # "train" or "test"
    data_mode: str          # "LO" (label-only) or "Semi" (semi-supervised)
    # Training weights (None if absent):
    pref_weight: Optional[float]
    nllv_weight: Optional[float]
    nllg_weight: Optional[float]
    # Flags:
    has_tc_self: bool       # _tc-self_ in filename (not tc-self-step)
    has_tc_self_step: bool  # _tc-self-step_ in filename
    has_norm: bool          # _lenorm_ in filename
    has_vallogodds: bool    # _vallogodds in filename
    # Derived:
    training_mode: str      # "SFT", "Pref", or "Comb" ("" for base)
    row_label: str          # e.g. "LO-Comb-tcs-norm-v" or "Base"
    timestamp: str          # extracted from filename for dedup
    is_base: bool = False   # True for non-finetuned base model score files


# =============================================================================
# FILENAME PARSING
# =============================================================================

EVAL_TASK_RE = re.compile(
    r'(hypernym-[a-zA-Z]+|ambigqa-[a-zA-Z]+|'
    r'plausibleqa-(?:nq|trivia|webq)_\d+|'
    r'ifeval-prompt_\d+)'
    r'_(?:test|train)'
)


def parse_filename(csv_file, config):
    """Parse a semi-supervised scores CSV filename into a FileInfo.

    Pipeline:
      0. Optional: base model (base_pattern, non-finetuned) -> row "Base"
      1. Match finetuned pattern (scores_self-...-delta)
      2. Require _d2g_
      3. Extract data mode (semi / labelonly)
      4. Extract eval task via EVAL_TASK_RE
      5. Training mode (SFT / Pref / Comb)
      6. Flags (tc-self, tc-self-step, lenorm, vallogodds)

    Returns FileInfo or None (for files that should be skipped).
    """
    name = Path(csv_file).name
    stem = Path(csv_file).stem
    finetuned_pattern = config.get('finetuned_pattern', '') or ''
    base_pattern = (config.get('base_pattern') or '').strip()

    is_finetuned = bool(finetuned_pattern and re.match(finetuned_pattern, stem))

    # --- Base model (non-finetuned) ---
    if base_pattern and re.match(base_pattern, stem) and not is_finetuned:
        eval_match = EVAL_TASK_RE.search(stem)
        if not eval_match:
            return None
        task = eval_match.group(1)
        dataset = task.split('-', 1)[1] if '-' in task else task
        split = _extract_split(stem, config.get('split_patterns', {}))
        if split == 'unknown':
            return None
        timestamp = _extract_timestamp(name)
        return FileInfo(
            path=str(csv_file), filename=name,
            task=task, dataset=dataset, split=split,
            data_mode='', pref_weight=None, nllv_weight=None, nllg_weight=None,
            has_tc_self=False, has_tc_self_step=False, has_norm=False, has_vallogodds=False,
            training_mode='', row_label='Base', timestamp=timestamp,
            is_base=True,
        )

    if not finetuned_pattern or not is_finetuned:
        return None

    if '_d2g_' not in stem:
        return None

    # --- Data mode ---
    semi_match = re.search(r'_semi(\d+(?:\.\d+)?)_', stem)
    labelonly_match = re.search(r'_labelonly(\d+(?:\.\d+)?)_', stem)

    if semi_match:
        data_mode = 'Semi'
    elif labelonly_match:
        data_mode = 'LO'
    else:
        return None

    # --- Eval task ---
    eval_match = EVAL_TASK_RE.search(stem)
    if not eval_match:
        return None
    task = eval_match.group(1)
    dataset = task.split('-', 1)[1] if '-' in task else task

    # --- Split ---
    split = _extract_split(stem, config.get('split_patterns', {}))
    if split == 'unknown':
        return None

    timestamp = _extract_timestamp(name)

    # --- Training mode ---
    pref_weight = _extract_float(r'(?:^|[_-])pref(?P<weight>\d+(?:\.\d+)?)', stem)
    nllv_weight = _extract_float(r'nllv(?P<weight>\d+(?:\.\d+)?)', stem)
    nllg_weight = _extract_float(r'nllg(?P<weight>\d+(?:\.\d+)?)', stem)

    eff_pref = pref_weight if pref_weight is not None else 1.0
    eff_nllv = nllv_weight if nllv_weight is not None else 0.0
    eff_nllg = nllg_weight if nllg_weight is not None else 0.0

    is_sft = (eff_pref == 0.0 and eff_nllv == 1.0 and eff_nllg == 1.0)
    is_pref_only = (eff_nllv == 0.0 and eff_nllg == 0.0)
    is_comb = (eff_nllv == 1.0 and eff_nllg == 1.0 and eff_pref == 1.0)

    modes_matched = sum([is_sft, is_pref_only, is_comb])
    if modes_matched != 1:
        raise ValueError(
            f"Training mode ambiguous for: {name}\n"
            f"  pref={pref_weight}, nllv={nllv_weight}, nllg={nllg_weight}\n"
            f"  SFT={is_sft}, Pref={is_pref_only}, Comb={is_comb}"
        )

    if is_sft:
        training_mode = 'SFT'
    elif is_pref_only:
        training_mode = 'Pref'
    else:
        training_mode = 'Comb'

    # --- Flags ---
    has_tc_self_step = '_tc-self-step_' in stem
    has_tc_self = ('_tc-self_' in stem) and not has_tc_self_step
    has_norm = '_lenorm_' in stem
    has_vallogodds = '_vallogodds' in stem

    row_label = build_row_label(
        data_mode, training_mode, has_tc_self, has_tc_self_step, has_norm, has_vallogodds
    )

    return FileInfo(
        path=str(csv_file), filename=name,
        task=task, dataset=dataset, split=split,
        data_mode=data_mode,
        pref_weight=pref_weight, nllv_weight=nllv_weight, nllg_weight=nllg_weight,
        has_tc_self=has_tc_self, has_tc_self_step=has_tc_self_step,
        has_norm=has_norm, has_vallogodds=has_vallogodds,
        training_mode=training_mode, row_label=row_label,
        timestamp=timestamp,
        is_base=False,
    )


# =============================================================================
# ROW LABEL BUILDER
# =============================================================================

def build_row_label(
    data_mode, training_mode, has_tc_self, has_tc_self_step, has_norm, has_vallogodds
):
    """Build a row label from parsed fields.

    Format: {data_mode}-{training_mode}[-tcs|-tcsstep][-norm][-v]
    Examples: "LO-Pref", "Semi-Comb-tcs-norm-v", "Semi-Comb-tcsstep", "LO-SFT-tcs"
    """
    parts = [f"{data_mode}-{training_mode}"]
    if has_tc_self_step:
        parts.append('tcsstep')
    elif has_tc_self:
        parts.append('tcs')
    if has_norm:
        parts.append('norm')
    if has_vallogodds:
        parts.append('v')
    return '-'.join(parts)


def row_sort_key(row_label):
    """Sort key for row labels. LO before Semi, then by mode, then flags; Base last."""
    if row_label == 'Base':
        return (2, 99, '', row_label)
    parts = row_label.split('-', 2)
    data_mode = parts[0] if parts else ''

    dm_order = 0 if data_mode == 'LO' else 1

    mode = parts[1] if len(parts) > 1 else ''
    mode_order = {'Pref': 0, 'Comb': 1, 'SFT': 2}.get(mode, 3)

    flags = '-'.join(parts[2:]) if len(parts) > 2 else ''

    return (dm_order, mode_order, flags, row_label)


def is_row_visible(row_label, config):
    """Check if a row should be displayed based on config visibility settings."""
    if row_label == 'Base':
        return bool(config.get('include_base_model_eval', True))

    visible_data_modes = config.get('visible_data_modes', ['LO', 'Semi'])
    visible_modes = config.get('visible_modes', ['Comb', 'SFT', 'Pref'])
    visible_flags = config.get('visible_flags', ['tcs', 'tcsstep', 'norm', 'v'])

    parts = row_label.split('-', 2)
    data_mode = parts[0] if parts else ''
    mode = parts[1] if len(parts) > 1 else ''

    if data_mode not in visible_data_modes:
        return False
    if mode not in visible_modes:
        return False

    flags_part = parts[2] if len(parts) > 2 else ''
    if flags_part:
        row_flags = flags_part.split('-')
        for flag in row_flags:
            if flag not in visible_flags:
                return False

    return True


# =============================================================================
# FILE RESOLUTION
# =============================================================================

def _iter_score_csv_paths(outputs_dir: Path, config):
    """Yield unique CSV paths: primary file_pattern plus base_scores_glob when base eval is enabled."""
    patterns = [config.get('file_pattern', 'scores_self-*.csv')]
    base_pat = (config.get('base_pattern') or '').strip()
    if base_pat and config.get('include_base_model_eval', True):
        bg = (config.get('base_scores_glob') or '').strip() or 'scores_v6-google_gemma-2-2b*.csv'
        if bg not in patterns:
            patterns.append(bg)
    seen = set()
    for pat in patterns:
        for p in sorted(outputs_dir.glob(pat)):
            if p not in seen:
                seen.add(p)
                yield p


def discover_and_resolve_files(config):
    """Discover all scores files and parse them into FileInfo objects.

    Returns:
        list of FileInfo: All successfully parsed files.
    """
    outputs_dir = Path(config['outputs_dir'])

    all_files = []
    skipped = 0
    for csv_file in _iter_score_csv_paths(outputs_dir, config):
        try:
            info = parse_filename(csv_file, config)
            if info is not None:
                all_files.append(info)
            else:
                skipped += 1
        except ValueError as e:
            # Unexpected format - raise loudly
            raise ValueError(str(e))

    print(f"Discovered {len(all_files)} valid files, skipped {skipped}")
    return all_files


def resolve_files_for_task(file_infos, task, split):
    """Resolve files for a specific task/split to a 1:1 row_label -> FileInfo mapping.

    If multiple files map to the same row_label, picks the newest by timestamp.

    Returns:
        dict: {row_label: FileInfo}
        list: warnings
    """
    matching = [f for f in file_infos if f.task == task and f.split == split]

    # Group by row_label
    by_label = {}
    for f in matching:
        if f.row_label not in by_label:
            by_label[f.row_label] = []
        by_label[f.row_label].append(f)

    result = {}
    warnings = []
    for label, files in by_label.items():
        if len(files) == 1:
            result[label] = files[0]
        else:
            # Pick newest by timestamp
            files_sorted = sorted(files, key=lambda f: f.timestamp, reverse=True)
            result[label] = files_sorted[0]
            filenames = [f.filename for f in files_sorted]
            warnings.append(
                f"  [DEDUP] {task}/{split}/{label}: {len(files)} files found, "
                f"using newest: {filenames[0]} (dropped: {filenames[1:]})"
            )

    return result, warnings


def write_file_tracking_log(all_resolved, config):
    """Write file tracking log to a new file with datetime."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    log_filename = f"dashboard_file_tracking_{now.strftime('%Y%m%d_%H%M%S')}.log"
    log_file = CONFIG_DIR / log_filename

    with open(log_file, 'w') as f:
        f.write(f"=== Dashboard file tracking ({now.strftime('%Y-%m-%d %H:%M:%S')}) ===\n\n")

        for (task, split), resolved in sorted(all_resolved.items()):
            f.write(f"--- {task} / {split} ---\n")
            for label in sorted(resolved.keys(), key=row_sort_key):
                info = resolved[label]
                f.write(f"  {label}: {info.filename}\n")
            f.write("\n")

    print(f"File tracking log: {log_file}")
    return str(log_file)


# =============================================================================
# DATA LOADING AND METRICS
# =============================================================================

def load_scores_data(csv_path, config):
    """Load scores data from CSV file based on config."""
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
    """Compute all metrics for a set of scores."""
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


DASH_TASK_SOURCE_COL = '_dash_task_source'

# Plotly qualitative + Dark24 + Set2 (no runtime import of plotly.colors for env compatibility)
_QUALITATIVE_COLORS = (
    '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880',
    '#FF97FF', '#FECB52', '#2E91E5', '#E15F99', '#1CA71C', '#FB0D0D', '#DA16FF', '#222A2A',
    '#B68100', '#750D86', '#EB663B', '#511CFB', '#00A08B', '#FB00D1', '#FC0080', '#EBEEEB',
    '#AD9900', '#15EF4F', '#A1045A', '#785EF0', '#00FFC0', '#FA4B1B', '#FE00FA', '#F14D16',
    '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2', '#7F7F7F',
    '#BCBD22', '#17BECF', '#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854', '#FFD92F',
    '#E5C494', '#B3B3B3',
)


def _qualitative_palette(n: int) -> List[str]:
    return [_QUALITATIVE_COLORS[i % len(_QUALITATIVE_COLORS)] for i in range(n)]


def build_multi_prompt_gv_figure(
    combined_df: pd.DataFrame,
    gen_col: str,
    gen_axis_title: str,
    val_col: str,
    metric_label: str,
    metric_type: str,
    title: str,
    task_col: str = DASH_TASK_SOURCE_COL,
) -> go.Figure:
    """Generator vs validator scatter with marginals; points colored by source task (prompt), not label."""
    def _empty(msg: str) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            paper_bgcolor='white',
            plot_bgcolor='white',
            annotations=[dict(text=msg, xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False)],
        )
        return fig

    if combined_df is None or combined_df.empty:
        return _empty('Select one or more tasks and ensure split/model match files.')

    if gen_col not in combined_df.columns or val_col not in combined_df.columns:
        return _empty(f'Missing columns: need {gen_col!r} and {val_col!r} in loaded data.')

    df = combined_df.copy()
    valid_mask = np.isfinite(df[gen_col].astype(float)) & np.isfinite(df[val_col].astype(float))
    valid = df.loc[valid_mask]
    if valid.empty:
        return _empty('No finite gen/val scores for the selected tasks.')

    tasks = sorted(valid[task_col].astype(str).unique())
    palette = _qualitative_palette(len(tasks))
    colors = {t: palette[i] for i, t in enumerate(tasks)}

    threshold = 0 if metric_type == 'log-odds' else np.log(0.5)

    has_noun2 = 'noun2' in valid.columns
    prompt_col = 'prompt' if 'prompt' in valid.columns else ('val_prompt' if 'val_prompt' in valid.columns else None)
    response_col = 'response' if 'response' in valid.columns else ('answer' if 'answer' in valid.columns else None)
    main_fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.8, 0.2],
        row_heights=[0.2, 0.8],
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
        specs=[[{"type": "histogram"}, None],
               [{"type": "scatter"}, {"type": "histogram"}]],
    )

    for t in tasks:
        sub = valid[valid[task_col].astype(str) == t]
        gx = sub[gen_col].values.astype(float)
        vy = sub[val_col].values.astype(float)
        lab = sub['label'].values if 'label' in sub.columns else np.zeros(len(sub), dtype=int)
        hover_texts = []
        for i in range(len(gx)):
            parts = [f"Task={t}", f"Gen={gx[i]:.2f}", f"Val={vy[i]:.2f}",
                     'correct' if lab[i] == 1 else 'incorrect']
            if has_noun2:
                parts.append(f"Item={sub['noun2'].iloc[i]}")
            elif response_col == 'answer' and 'answer' in sub.columns:
                parts.append(f"Answer={sub['answer'].iloc[i]}")
            hover_texts.append(' | '.join(parts))
        if prompt_col or response_col:
            fp = sub[prompt_col].astype(str).fillna('').values if prompt_col else np.array([''] * len(sub), dtype=object)
            fr = sub[response_col].astype(str).fillna('').values if response_col else np.array([''] * len(sub), dtype=object)
            cd = np.column_stack([fp, fr])
        else:
            cd = None

        c = colors[t]
        main_fig.add_trace(
            go.Scatter(
                x=gx,
                y=vy,
                mode='markers',
                marker=dict(color=c, size=8, opacity=0.65),
                name=str(t),
                legendgroup=str(t),
                hovertext=np.array(hover_texts),
                hoverinfo='text',
                customdata=cd,
            ),
            row=2,
            col=1,
        )
        main_fig.add_trace(
            go.Histogram(x=gx, marker_color=c, opacity=0.45, showlegend=False, name=f'{t}-x'),
            row=1,
            col=1,
        )
        main_fig.add_trace(
            go.Histogram(y=vy, marker_color=c, opacity=0.45, showlegend=False, name=f'{t}-y'),
            row=2,
            col=2,
        )

    main_fig.add_hline(y=threshold, line=dict(color='red', dash='dash', width=2), row=2, col=1)

    main_fig.update_layout(
        title=title,
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=True,
        barmode='overlay',
        legend=dict(orientation='v', yanchor='top', y=1, xanchor='left', x=1.02),
    )
    main_fig.update_xaxes(title_text=gen_axis_title, row=2, col=1, showgrid=True, gridcolor='lightgray')
    main_fig.update_yaxes(title_text=f'Validator {metric_label}', row=2, col=1, showgrid=True, gridcolor='lightgray')
    return main_fig


# =============================================================================
# HEATMAP DATA LOADING
# =============================================================================

def load_heatmap_data_for_task(resolved_files, config):
    """Load metrics for all rows in a resolved task/split.

    Args:
        resolved_files: dict {row_label: FileInfo}
        config: dashboard config

    Returns:
        dict: {row_label: {eval_col: metrics_dict}}
    """
    eval_columns = config.get('eval_columns', DEFAULT_CONFIG['eval_columns'])
    data = {}

    for row_label, file_info in resolved_files.items():
        try:
            df = load_scores_data(file_info.path, config)
            metric_type = 'log-odds' if 'log-odds' in file_info.path else 'log-probs'

            row_data = {}
            for eval_col, gen_col in eval_columns.items():
                if gen_col in df.columns and 'val_score' in df.columns and 'label' in df.columns:
                    gen_scores = df[gen_col].values
                    val_scores = df['val_score'].values
                    labels = df['label'].values
                    metrics = compute_metrics(gen_scores, val_scores, labels, metric_type)
                    row_data[eval_col] = metrics
            data[row_label] = row_data
        except Exception as e:
            print(f"  [ERROR] Loading {file_info.filename}: {e}")
            continue

    return data


# =============================================================================
# HEATMAP AND BAR PLOT BUILDERS
# =============================================================================

def build_heatmap(data, row_labels, eval_cols, title, metrics_list):
    """Build a single heatmap figure.

    Args:
        data: {row_label: {eval_col: metrics_dict}}
        row_labels: ordered list of row labels (top to bottom)
        eval_cols: list of eval column names
        title: figure title
        metrics_list: list of metric display names

    Returns:
        plotly Figure or None if no rows
    """
    if not row_labels:
        return None

    fig = make_subplots(rows=1, cols=len(metrics_list), subplot_titles=metrics_list,
                        horizontal_spacing=0.03)

    for m_idx, metric in enumerate(metrics_list):
        metric_key = METRIC_KEY_MAP.get(metric, metric.lower())

        z = []
        text = []
        for row in row_labels:
            z_row = []
            text_row = []
            for col in eval_cols:
                metrics = data.get(row, {}).get(col)
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
                z=z, x=eval_cols, y=row_labels,
                text=text, texttemplate='%{text}', textfont={'size': 10},
                colorscale='RdYlGn', zmin=0, zmax=100,
                showscale=(m_idx == len(metrics_list) - 1),
                hovertemplate='%{y} / %{x}: %{z:.1f}<extra></extra>'
            ),
            row=1, col=m_idx + 1
        )

    for m_idx in range(len(metrics_list)):
        fig.update_yaxes(showticklabels=(m_idx == 0), row=1, col=m_idx + 1)

    # Force all y-axis labels to show (prevent Plotly auto-culling)
    for m_idx in range(len(metrics_list)):
        fig.update_yaxes(dtick=1, row=1, col=m_idx + 1)

    fig.update_layout(
        title=title,
        height=max(300, 45 * len(row_labels) + 120),
        margin=dict(l=180, r=20, t=50, b=30),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    return fig


def build_aggregated_heatmap(all_task_data, tasks, row_labels, eval_cols, title, metrics_list):
    """Build aggregated heatmap showing mean across tasks.

    Args:
        all_task_data: {task: {row_label: {eval_col: metrics_dict}}}
        tasks: list of tasks to aggregate
        row_labels: ordered list of row labels
        eval_cols: list of eval column names
        title: figure title
        metrics_list: list of metric display names

    Returns:
        plotly Figure or None
    """
    if not row_labels:
        return None

    # Average the metrics across tasks
    avg_data = {}
    for row in row_labels:
        avg_data[row] = {}
        for col in eval_cols:
            for metric_key in ['acc', 'val_roc', 'gen_roc', 'corr', 'corr_pos', 'corr_neg']:
                values = []
                for task in tasks:
                    task_data = all_task_data.get(task, {})
                    metrics = task_data.get(row, {}).get(col)
                    if metrics is not None and not np.isnan(metrics.get(metric_key, np.nan)):
                        values.append(metrics[metric_key])
                if col not in avg_data[row]:
                    avg_data[row][col] = {}
                avg_data[row][col][metric_key] = np.mean(values) if values else np.nan

    return build_heatmap(avg_data, row_labels, eval_cols, title, metrics_list)


def build_bar_plot(all_task_data, tasks, row_labels, eval_cols, metric, title):
    """Build bar plot with standard error for a single metric.

    Args:
        all_task_data: {task: {row_label: {eval_col: metrics_dict}}}
        tasks: list of tasks
        row_labels: ordered list of row labels
        eval_cols: list of eval column names
        metric: metric display name
        title: figure title

    Returns:
        plotly Figure
    """
    metric_key = METRIC_KEY_MAP.get(metric, metric.lower())
    fig = go.Figure()

    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
              '#19D3F3', '#FF6692', '#B6E880', '#FECB52']

    x_positions = []
    x_labels = []
    current_x = 0

    for row_idx, row in enumerate(row_labels):
        for col_idx, col in enumerate(eval_cols):
            values = []
            for task in tasks:
                metrics = all_task_data.get(task, {}).get(row, {}).get(col)
                if metrics is not None and not np.isnan(metrics.get(metric_key, np.nan)):
                    values.append(metrics[metric_key] * 100)

            mean_val = np.mean(values) if values else 0
            std_err = np.std(values) / np.sqrt(len(values)) if len(values) > 1 else 0

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


# =============================================================================
# AGGREGATION HELPERS
# =============================================================================

def expand_aggregation_pattern(pattern, all_tasks):
    """Expand aggregation pattern to list of tasks.

    Supports three pattern types:
      - list of task names:  ["task-a", "task-b"]
      - glob string:         "task-*"
      - regex string:        "regex:task-(foo|bar)_\\d+"
    """
    if isinstance(pattern, list):
        return [t for t in pattern if t in all_tasks]
    elif isinstance(pattern, str) and pattern.startswith('regex:'):
        rx = re.compile(pattern[len('regex:'):])
        return [t for t in all_tasks if rx.search(t)]
    elif isinstance(pattern, str) and '*' in pattern:
        return [t for t in all_tasks if fnmatch.fnmatch(t, pattern)]
    elif isinstance(pattern, str):
        return [pattern] if pattern in all_tasks else []
    return []


def task_name_matches_filter(task_name: str, pattern: str) -> bool:
    """Match a parsed task name against task_filter_pattern (empty string matches all).

    - Glob with '*': fnmatch
    - Prefix ``regex:``: re.search on task name
    - Otherwise: exact string equality
    """
    p = (pattern or '').strip()
    if not p:
        return True
    if p.startswith('regex:'):
        try:
            return bool(re.search(p[len('regex:'):], task_name))
        except re.error:
            return False
    if '*' in p:
        return fnmatch.fnmatch(task_name, p)
    return task_name == p


def filter_file_infos_by_task_pattern(file_infos, config):
    """Keep only FileInfo rows whose task matches config task_filter_pattern."""
    pat = (config.get('task_filter_pattern') or '').strip()
    if not pat:
        return file_infos
    return [f for f in file_infos if task_name_matches_filter(f.task, pat)]


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


# =============================================================================
# DASH APP
# =============================================================================

app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    # Stores
    dcc.Store(id='config-store', data=None),
    dcc.Store(id='files-store', data=None),

    # Config Page
    html.Div(id='config-page', children=[
        html.H1('📊 Dashboard Configuration',
                style={'textAlign': 'center', 'color': '#333', 'marginBottom': '30px'}),

        html.Div([
            # Buttons row
            html.Div([
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

                # Finetuned pattern
                html.Div([
                    html.Label('Finetuned Model Pattern (regex):', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                    dcc.Input(id='config-finetuned-pattern', type='text',
                             value=DEFAULT_CONFIG['finetuned_pattern'],
                             style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'}),
                    html.Small('Matches the start of score filenames, e.g. ^scores_self-v6-google_gemma-2-2b-delta',
                              style={'color': '#666'})
                ], style={'marginBottom': '15px'}),

                # Base model pattern + glob
                html.Div([
                    html.Label('Base Model Pattern (regex, optional):', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                    dcc.Input(id='config-base-pattern', type='text',
                             value=DEFAULT_CONFIG['base_pattern'],
                             style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'}),
                    html.Small('e.g. ^scores_v6-google_gemma-2-2b_ — leave empty to skip base model CSVs',
                              style={'color': '#666'})
                ], style={'marginBottom': '15px'}),

                html.Div([
                    html.Label('Base Model File Glob:', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                    dcc.Input(id='config-base-scores-glob', type='text',
                             value=DEFAULT_CONFIG['base_scores_glob'],
                             style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'}),
                    html.Small('Used only when Base Model Pattern is non-empty; widened to discover base score files',
                              style={'color': '#666'})
                ], style={'marginBottom': '15px'}),

                html.Div([
                    dcc.Checklist(
                        id='config-include-base',
                        options=[{'label': ' Include base model eval row (heatmaps + model dropdown)', 'value': 'yes'}],
                        value=['yes'] if DEFAULT_CONFIG['include_base_model_eval'] else [],
                        style={'fontSize': '14px'}
                    ),
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

                # Aggregation groups
                html.Div([
                    html.Label('Aggregation Groups (JSON):', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                    dcc.Textarea(id='config-aggregation',
                                value=json.dumps(DEFAULT_CONFIG['aggregation_groups'], indent=2),
                                style={'width': '100%', 'height': '80px', 'padding': '8px', 'borderRadius': '4px',
                                       'border': '1px solid #ccc', 'fontFamily': 'monospace'}),
                ], style={'marginBottom': '15px'}),

                # Task filter (parsed task ids)
                html.Div([
                    html.Label('Task filter (optional):', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                    dcc.Input(id='config-task-filter-pattern', type='text',
                             value=DEFAULT_CONFIG['task_filter_pattern'],
                             placeholder='e.g. hypernym-*  or  regex:ifeval-prompt_.*',
                             style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'}),
                    html.Small('Restrict dropdowns and heatmaps to matching parsed task names. '
                               'Empty = all tasks. Glob (*), exact match, or regex:... (same rules as aggregation groups).',
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

                # Visibility controls
                html.Div([
                    html.Label('Visible Data Modes (JSON list):', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                    dcc.Input(id='config-visible-data-modes', type='text',
                             value=json.dumps(DEFAULT_CONFIG['visible_data_modes']),
                             style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'}),
                    html.Small('Options: "LO" (label-only), "Semi" (semi-supervised)', style={'color': '#666'})
                ], style={'marginBottom': '15px'}),

                html.Div([
                    html.Label('Visible Modes (JSON list):', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                    dcc.Input(id='config-visible-modes', type='text',
                             value=json.dumps(DEFAULT_CONFIG['visible_modes']),
                             style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'}),
                    html.Small('Options: "Comb", "SFT", "Pref"', style={'color': '#666'})
                ], style={'marginBottom': '15px'}),

                html.Div([
                    html.Label('Visible Flags (JSON list):', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                    dcc.Input(id='config-visible-flags', type='text',
                             value=json.dumps(DEFAULT_CONFIG['visible_flags']),
                             style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'}),
                    html.Small(
                        'Options: "tcs" (tc-self), "tcsstep" (tc-self-step), '
                        '"norm" (lenorm), "v" (vallogodds)',
                        style={'color': '#666'},
                    )
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
            html.H1('⚖️ Semi-Supervised GV Dashboard',
                    style={'textAlign': 'center', 'color': '#333', 'display': 'inline-block'}),
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

        # Multi-prompt overlay (same layout as main GV scatter; color = task, not label)
        html.Details([
            html.Summary(
                '🧩 Multi-prompt generator vs validator (raw & FC)',
                style={'cursor': 'pointer', 'fontWeight': 'bold'}
            ),
            html.Div([
                html.Label('Tasks to overlay on one plot (same split & model):', style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='multi-prompt-task-selector',
                    multi=True,
                    placeholder='Select one or more tasks…',
                    style={'width': '100%'},
                ),
            ], style={'width': '80%', 'margin': '12px auto'}),
            html.P(
                'Points are colored by task (legend). Raw uses config eval column '
                '"raw"; FC uses "tc" (typo-corrected).',
                style={'width': '80%', 'margin': '0 auto 8px', 'color': '#555', 'fontSize': '13px'},
            ),
            dcc.Graph(id='multi-prompt-raw-scatter', style={'height': '700px'}),
            html.Div(id='response-panel-multi-raw', style={
                'width': '80%', 'margin': '8px auto 12px', 'padding': '10px 14px',
                'backgroundColor': '#fff8e1', 'borderRadius': '8px',
                'border': '1px solid #ffe0b2', 'fontFamily': 'monospace',
                'whiteSpace': 'pre-wrap',
            }, children='Click a point in the raw plot to view prompt/response here.'),
            dcc.Graph(id='multi-prompt-fc-scatter', style={'height': '700px'}),
            html.Div(id='response-panel-multi-fc', style={
                'width': '80%', 'margin': '8px auto 12px', 'padding': '10px 14px',
                'backgroundColor': '#fff8e1', 'borderRadius': '8px',
                'border': '1px solid #ffe0b2', 'fontFamily': 'monospace',
                'whiteSpace': 'pre-wrap',
            }, children='Click a point in the FC plot to view prompt/response here.'),
        ], style={'margin': '20px'}),

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
                type='default',
                children=html.Div(id='all-heatmaps-container'),
                style={'minHeight': '200px'}
            )
        ], open=True, style={'margin': '20px'}),

    ], style={'padding': '20px', 'backgroundColor': 'white', 'minHeight': '100vh', 'display': 'none'})
])


# =============================================================================
# CALLBACKS - CONFIG PAGE
# =============================================================================

def _build_config_from_form(outputs_dir, finetuned_pattern, base_pattern, base_scores_glob,
                            include_base, split_patterns, label_col,
                            label_map, aggregation, task_filter_pattern, eval_cols,
                            visible_data_modes, visible_modes, visible_flags):
    """Build config dict from form values."""
    return {
        'outputs_dir': outputs_dir,
        'file_pattern': 'scores_self-*.csv',
        'finetuned_pattern': finetuned_pattern,
        'base_pattern': (base_pattern or '').strip(),
        'base_scores_glob': (base_scores_glob or '').strip(),
        'include_base_model_eval': bool(include_base and 'yes' in include_base),
        'split_patterns': json.loads(split_patterns) if split_patterns else {},
        'label_column': label_col,
        'label_map': json.loads(label_map) if label_map else None,
        'gen_score_col': 'gen_score',
        'val_score_col': 'val_score',
        'aggregation_groups': json.loads(aggregation) if aggregation else {},
        'task_filter_pattern': (task_filter_pattern or '').strip(),
        'eval_columns': json.loads(eval_cols) if eval_cols else {},
        'metrics': DEFAULT_CONFIG['metrics'],
        'visible_data_modes': json.loads(visible_data_modes) if visible_data_modes else ['LO', 'Semi'],
        'visible_modes': json.loads(visible_modes) if visible_modes else ['Comb', 'SFT', 'Pref'],
        'visible_flags': json.loads(visible_flags) if visible_flags else ['tcs', 'tcsstep', 'norm', 'v'],
    }


@app.callback(
    [Output('config-outputs-dir', 'value'),
     Output('config-finetuned-pattern', 'value'),
     Output('config-base-pattern', 'value'),
     Output('config-base-scores-glob', 'value'),
     Output('config-include-base', 'value'),
     Output('config-split-patterns', 'value'),
     Output('config-label-col', 'value'),
     Output('config-label-map', 'value'),
     Output('config-aggregation', 'value'),
     Output('config-task-filter-pattern', 'value'),
     Output('config-eval-cols', 'value'),
     Output('config-visible-data-modes', 'value'),
     Output('config-visible-modes', 'value'),
     Output('config-visible-flags', 'value'),
     Output('config-status', 'children'),
     Output('config-status', 'style')],
    [Input('load-json-btn', 'n_clicks')],
    prevent_initial_call=True
)
def handle_load_config(n_clicks):
    """Handle Load JSON Config button."""
    if not n_clicks:
        raise PreventUpdate

    base_style = {'padding': '10px', 'marginBottom': '20px', 'borderRadius': '5px', 'textAlign': 'center'}

    config, error = load_config_from_file()
    if error:
        style = {**base_style, 'backgroundColor': '#ffebee', 'color': '#c62828'}
        return (
            dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update,
            f"⚠️ {error}",
            style
        )

    style = {**base_style, 'backgroundColor': '#e8f5e9', 'color': '#2e7d32'}
    inc = config.get('include_base_model_eval', True)
    return (
        config.get('outputs_dir', str(DEFAULT_OUTPUTS_DIR)),
        config.get('finetuned_pattern', ''),
        config.get('base_pattern', DEFAULT_CONFIG['base_pattern']),
        config.get('base_scores_glob', DEFAULT_CONFIG['base_scores_glob']),
        ['yes'] if inc else [],
        json.dumps(config.get('split_patterns', {})),
        config.get('label_column', 'gpt4_ground_truth'),
        json.dumps(config.get('label_map')) if config.get('label_map') else '',
        json.dumps(config.get('aggregation_groups', {}), indent=2),
        config.get('task_filter_pattern', ''),
        json.dumps(config.get('eval_columns', {}), indent=2),
        json.dumps(config.get('visible_data_modes', ['LO', 'Semi'])),
        json.dumps(config.get('visible_modes', ['Comb', 'SFT', 'Pref'])),
        json.dumps(config.get('visible_flags', ['tcs', 'tcsstep', 'norm', 'v'])),
        f"✅ Loaded config from {CONFIG_FILE}",
        style
    )


@app.callback(
    [Output('config-status', 'children', allow_duplicate=True),
     Output('config-status', 'style', allow_duplicate=True)],
    [Input('save-json-btn', 'n_clicks')],
    [State('config-outputs-dir', 'value'),
     State('config-finetuned-pattern', 'value'),
     State('config-base-pattern', 'value'),
     State('config-base-scores-glob', 'value'),
     State('config-include-base', 'value'),
     State('config-split-patterns', 'value'),
     State('config-label-col', 'value'),
     State('config-label-map', 'value'),
     State('config-aggregation', 'value'),
     State('config-task-filter-pattern', 'value'),
     State('config-eval-cols', 'value'),
     State('config-visible-data-modes', 'value'),
     State('config-visible-modes', 'value'),
     State('config-visible-flags', 'value')],
    prevent_initial_call=True
)
def save_config(n_clicks, outputs_dir, finetuned_pattern, base_pattern, base_scores_glob,
                include_base, split_patterns, label_col,
                label_map, aggregation, task_filter_pattern, eval_cols,
                visible_data_modes, visible_modes, visible_flags):
    """Save current config to JSON file."""
    if not n_clicks:
        raise PreventUpdate

    base_style = {'padding': '10px', 'marginBottom': '20px', 'borderRadius': '5px', 'textAlign': 'center'}

    try:
        config = _build_config_from_form(
            outputs_dir, finetuned_pattern, base_pattern, base_scores_glob, include_base,
            split_patterns, label_col,
            label_map, aggregation, task_filter_pattern, eval_cols,
            visible_data_modes, visible_modes, visible_flags
        )
        save_config_to_file(config)
        style = {**base_style, 'backgroundColor': '#e8f5e9', 'color': '#2e7d32'}
        return f"✅ Saved config to {CONFIG_FILE}", style

    except Exception as e:
        style = {**base_style, 'backgroundColor': '#ffebee', 'color': '#c62828'}
        return f"⚠️ Error saving config: {e}", style


# =============================================================================
# CALLBACKS - PAGE TOGGLE & DASHBOARD LOAD
# =============================================================================

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
     Output('model-selector', 'value'),
     Output('multi-prompt-task-selector', 'options'),
     Output('multi-prompt-task-selector', 'value')],
    [Input('load-dashboard-btn', 'n_clicks'),
     Input('load-dashboard-btn-top', 'n_clicks'),
     Input('back-to-config-btn', 'n_clicks')],
    [State('config-outputs-dir', 'value'),
     State('config-finetuned-pattern', 'value'),
     State('config-base-pattern', 'value'),
     State('config-base-scores-glob', 'value'),
     State('config-include-base', 'value'),
     State('config-split-patterns', 'value'),
     State('config-label-col', 'value'),
     State('config-label-map', 'value'),
     State('config-aggregation', 'value'),
     State('config-task-filter-pattern', 'value'),
     State('config-eval-cols', 'value'),
     State('config-visible-data-modes', 'value'),
     State('config-visible-modes', 'value'),
     State('config-visible-flags', 'value')],
    prevent_initial_call=True
)
def toggle_pages(load_clicks, load_clicks_top, back_clicks,
                 outputs_dir, finetuned_pattern, base_pattern, base_scores_glob, include_base,
                 split_patterns, label_col,
                 label_map, aggregation, task_filter_pattern, eval_cols,
                 visible_data_modes, visible_modes, visible_flags):
    """Toggle between config page and viz page."""
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'load-dashboard-btn-top':
        button_id = 'load-dashboard-btn'

    config_visible = {'padding': '40px', 'backgroundColor': 'white', 'minHeight': '100vh'}
    config_hidden = {'display': 'none'}
    viz_visible = {'padding': '20px', 'backgroundColor': 'white', 'minHeight': '100vh'}
    viz_hidden = {'display': 'none'}

    if button_id == 'back-to-config-btn':
        return (config_visible, viz_hidden, None, None,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update)

    # Load dashboard
    try:
        config = _build_config_from_form(
            outputs_dir, finetuned_pattern, base_pattern, base_scores_glob, include_base,
            split_patterns, label_col,
            label_map, aggregation, task_filter_pattern, eval_cols,
            visible_data_modes, visible_modes, visible_flags
        )

        # Discover and parse files
        file_infos = discover_and_resolve_files(config)
        file_infos = filter_file_infos_by_task_pattern(file_infos, config)

        if not file_infos:
            raise PreventUpdate

        # Convert FileInfo objects to dicts for JSON storage in dcc.Store
        files_data = [asdict(f) for f in file_infos]

        # Build dropdown options
        all_tasks = sorted(set(f.task for f in file_infos))
        all_splits = sorted(set(f.split for f in file_infos))
        all_row_labels = sorted(
            set(f.row_label for f in file_infos if is_row_visible(f.row_label, config)),
            key=row_sort_key
        )

        task_options = [{'label': t, 'value': t} for t in all_tasks]
        split_options = [{'label': s, 'value': s} for s in all_splits]
        model_options = [{'label': m, 'value': m} for m in all_row_labels]

        return (
            config_hidden, viz_visible, config, files_data,
            task_options, all_tasks[0] if all_tasks else None,
            split_options, all_splits[0] if all_splits else None,
            model_options, all_row_labels[0] if all_row_labels else None,
            task_options,
            [],
        )

    except Exception as e:
        print(f"Error loading dashboard: {e}")
        import traceback
        traceback.print_exc()
        raise PreventUpdate


@app.callback(
    [Output('model-selector', 'options', allow_duplicate=True),
     Output('model-selector', 'value', allow_duplicate=True)],
    [Input('task-selector', 'value'),
     Input('split-selector', 'value')],
    [State('files-store', 'data'),
     State('config-store', 'data')],
    prevent_initial_call=True
)
def update_model_options(task, split, files_data, config):
    """Update model dropdown based on task and split selection."""
    if not task or not split or not files_data:
        raise PreventUpdate

    available = sorted(
        set(f['row_label'] for f in files_data if f['task'] == task and f['split'] == split),
        key=row_sort_key
    )
    if config:
        available = [r for r in available if is_row_visible(r, config)]

    if not available:
        return [{'label': 'No models available', 'value': None}], None

    return [{'label': m, 'value': m} for m in available], available[0]


@app.callback(
    [Output('multi-prompt-raw-scatter', 'figure'),
     Output('multi-prompt-fc-scatter', 'figure')],
    [Input('multi-prompt-task-selector', 'value'),
     Input('split-selector', 'value'),
     Input('model-selector', 'value')],
    [State('config-store', 'data'),
     State('files-store', 'data')],
)
def update_multi_prompt_scatters(selected_tasks, split, row_label, config, files_data):
    """Overlay multiple tasks on one GV scatter each for raw gen score and FC (typcorr) gen score."""

    def _msg_fig(title: str, msg: str) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            paper_bgcolor='white',
            plot_bgcolor='white',
            annotations=[
                dict(
                    text=msg,
                    xref='paper',
                    yref='paper',
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=14, color='#888'),
                )
            ],
        )
        return fig

    empty = go.Figure()
    if not config or not files_data or not split or not row_label:
        return empty, empty

    if not selected_tasks:
        return (
            _msg_fig('Multi-prompt: raw', 'Select one or more tasks above.'),
            _msg_fig('Multi-prompt: FC', 'Select one or more tasks above.'),
        )

    eval_columns = config.get('eval_columns', DEFAULT_CONFIG['eval_columns'])
    raw_col = eval_columns.get('raw', 'gen_score')
    fc_col = eval_columns.get('tc', 'gen_score_typcorr')
    val_col = config.get('val_score_col', 'val_score')

    frames: List[pd.DataFrame] = []
    first_path: Optional[str] = None
    for task in selected_tasks:
        matching = [
            f for f in files_data
            if f['task'] == task and f['split'] == split and f['row_label'] == row_label
        ]
        if not matching:
            continue
        csv_path = matching[0]['path']
        if first_path is None:
            first_path = csv_path
        try:
            df = load_scores_data(csv_path, config)
        except Exception:
            continue
        df = df.copy()
        df[DASH_TASK_SOURCE_COL] = task
        frames.append(df)

    if not frames:
        empty_ann = go.Figure()
        empty_ann.update_layout(
            title='No matching CSVs for the selected tasks / split / model.',
            paper_bgcolor='white',
            plot_bgcolor='white',
            annotations=[
                dict(
                    text='Check that each selected task exists for this split and model.',
                    xref='paper',
                    yref='paper',
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=13, color='#888'),
                )
            ],
        )
        return empty_ann, empty_ann

    combined = pd.concat(frames, ignore_index=True)
    metric_type = 'log-odds' if first_path and 'log-odds' in first_path else 'log-probs'
    metric_label = 'log-odds' if metric_type == 'log-odds' else 'log-probs'
    gen_axis_base = f'Generator {metric_label}'

    raw_fig = build_multi_prompt_gv_figure(
        combined,
        raw_col,
        f'{gen_axis_base} (raw)',
        val_col,
        metric_label,
        metric_type,
        'Multi-prompt: raw generator vs validator',
    )
    if fc_col not in combined.columns:
        fc_fig = go.Figure()
        fc_fig.update_layout(
            title='Multi-prompt: FC generator vs validator',
            paper_bgcolor='white',
            plot_bgcolor='white',
            annotations=[
                dict(
                    text=f'Column {fc_col!r} not found in loaded data.',
                    xref='paper',
                    yref='paper',
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                )
            ],
        )
    else:
        fc_fig = build_multi_prompt_gv_figure(
            combined,
            fc_col,
            f'{gen_axis_base} (FC)',
            val_col,
            metric_label,
            metric_type,
            'Multi-prompt: FC generator vs validator',
        )

    return raw_fig, fc_fig


# =============================================================================
# CALLBACKS - SCATTER PLOTS (ported as-is from original)
# =============================================================================

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
def update_visualizations(task, split, row_label, config, files_data):
    """Update main visualizations based on selections."""
    empty_fig = go.Figure()

    if not task or not split or not row_label or not config or not files_data:
        return "Please select all options", "No file selected", empty_fig, empty_fig, empty_fig, empty_fig

    # Find matching file
    matching = [f for f in files_data if f['task'] == task and f['split'] == split and f['row_label'] == row_label]

    if not matching:
        return f"No data for: {task} | {split} | {row_label}", "No data", empty_fig, empty_fig, empty_fig, empty_fig

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

    # Outlier detection (Kendall method)
    outlier_method = 'kendall'

    X = np.column_stack([gen_scores, val_scores])
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    if outlier_method == 'identity':
        outlier_scores = np.abs(X_std[:, 1] - X_std[:, 0]) / np.sqrt(2)
    elif outlier_method == 'kendall':
        n = len(gen_scores)
        outlier_scores = np.zeros(n)
        for i in range(n):
            discordant = 0
            for j in range(n):
                if i != j:
                    x_diff = gen_scores[i] - gen_scores[j]
                    y_diff = val_scores[i] - val_scores[j]
                    if x_diff * y_diff < 0:
                        discordant += 1
            outlier_scores[i] = discordant / (n - 1)
    else:
        raise ValueError(f"Unknown outlier_method: {outlier_method}")

    outlier_indices = np.argsort(outlier_scores)[-40:]
    outlier_set = set(outlier_indices)
    non_outlier_mask = np.array([i not in outlier_set for i in range(len(labels))])

    # Hover text
    has_noun2 = 'noun2' in df.columns
    has_prompt = 'prompt' in df.columns or 'val_prompt' in df.columns
    has_response = 'response' in df.columns or 'answer' in df.columns

    prompt_col = 'prompt' if 'prompt' in df.columns else ('val_prompt' if 'val_prompt' in df.columns else None)
    response_col = 'response' if 'response' in df.columns else ('answer' if 'answer' in df.columns else None)

    full_prompts = df[prompt_col].astype(str).fillna('') if prompt_col else pd.Series([''] * len(df))
    full_responses = df[response_col].astype(str).fillna('') if response_col else pd.Series([''] * len(df))

    hover_texts = []
    for i in range(len(gen_scores)):
        parts = [f"Gen={gen_scores[i]:.2f}", f"Val={val_scores[i]:.2f}"]
        if has_noun2:
            parts.append(f"Item={df['noun2'].iloc[i]}")
        elif response_col == 'answer':
            parts.append(f"Answer={df['answer'].iloc[i]}")
        hover_texts.append(" | ".join(parts))
    hover_texts = np.array(hover_texts)

    customdata = np.column_stack([full_prompts.values, full_responses.values]) if (has_prompt or has_response) else None

    # Scatter traces (excluding outliers)
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

    # Outlier display
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
            marker=dict(
                color=[POS_CLASS_COLOR if labels[i] == 1 else NEG_CLASS_COLOR for i in outlier_indices],
                size=8,
                opacity=0.6
            ),
            text=outlier_display_texts,
            textposition='top right', textfont=dict(size=8),
            name='Outliers', showlegend=True,
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

    faceted_fig.update_layout(
        title='Faceted by Strategy',
        paper_bgcolor='white',
        plot_bgcolor='white',
        height=400 * n_rows,
        showlegend=True
    )

    # === PCA PLOT ===
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)

    pca_fig = make_subplots(rows=1, cols=2, subplot_titles=['Standardized Scores', 'PCA'])

    # Left: Standardized scores
    pca_fig.add_trace(
        go.Scatter(
            x=X_std[pos_mask & non_outlier_mask, 0], y=X_std[pos_mask & non_outlier_mask, 1], mode='markers',
            marker=dict(color=POS_CLASS_COLOR, size=6, opacity=0.5), name='Positive', legendgroup='pos',
            hovertext=hover_texts[pos_mask & non_outlier_mask], hoverinfo='text',
            customdata=customdata[pos_mask & non_outlier_mask] if customdata is not None else None
        ), row=1, col=1
    )
    pca_fig.add_trace(
        go.Scatter(
            x=X_std[neg_mask & non_outlier_mask, 0], y=X_std[neg_mask & non_outlier_mask, 1], mode='markers',
            marker=dict(color=NEG_CLASS_COLOR, size=6, opacity=0.5), name='Negative', legendgroup='neg',
            hovertext=hover_texts[neg_mask & non_outlier_mask], hoverinfo='text',
            customdata=customdata[neg_mask & non_outlier_mask] if customdata is not None else None
        ), row=1, col=1
    )
    pca_fig.add_trace(
        go.Scatter(
            x=X_std[outlier_indices, 0], y=X_std[outlier_indices, 1], mode='markers',
            marker=dict(
                color=[POS_CLASS_COLOR if labels[i] == 1 else NEG_CLASS_COLOR for i in outlier_indices],
                size=6,
                opacity=0.5
            ),
            name='Outliers', showlegend=True,
            hovertext=hover_texts[outlier_indices], hoverinfo='text',
            customdata=customdata[outlier_indices] if customdata is not None else None
        ), row=1, col=1
    )

    std_range = max(np.abs(X_std).max(), 3)
    pca_fig.add_trace(go.Scatter(x=[-std_range, std_range], y=[-std_range, std_range], mode='lines',
                                  line=dict(color='gray', dash='dot', width=2),
                                  name='y=x', showlegend=True), row=1, col=1)

    # Right: PCA
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
            marker=dict(
                color=[POS_CLASS_COLOR if labels[i] == 1 else NEG_CLASS_COLOR for i in outlier_indices],
                size=6,
                opacity=0.5
            ),
            showlegend=False,
            hovertext=hover_texts[outlier_indices], hoverinfo='text',
            customdata=customdata[outlier_indices] if customdata is not None else None
        ), row=1, col=2
    )

    pca_fig.update_layout(paper_bgcolor='white', plot_bgcolor='white', showlegend=True)
    pca_fig.update_xaxes(title_text='Generator (std)', row=1, col=1, showgrid=True, gridcolor='lightgray',
                         zeroline=True, zerolinecolor='black', zerolinewidth=1)
    pca_fig.update_yaxes(title_text='Validator (std)', row=1, col=1, showgrid=True, gridcolor='lightgray',
                         zeroline=True, zerolinecolor='black', zerolinewidth=1)
    pca_fig.update_xaxes(title_text=f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', row=1, col=2, showgrid=True, gridcolor='lightgray',
                         zeroline=True, zerolinecolor='black', zerolinewidth=1)
    pca_fig.update_yaxes(title_text=f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', row=1, col=2, showgrid=True, gridcolor='lightgray',
                         zeroline=True, zerolinecolor='black', zerolinewidth=1)

    # === COMPARE CORRECTIONS 2x2 ===
    eval_columns = config.get('eval_columns', DEFAULT_CONFIG['eval_columns'])
    gen_variants = []
    for name, col in list(eval_columns.items())[:4]:
        label = name.replace('_', ' ').title().replace('Tc', 'Fc')
        gen_variants.append((col, label))

    compare_fig = make_subplots(rows=2, cols=2, subplot_titles=[v[1] for v in gen_variants],
                                 vertical_spacing=0.25, horizontal_spacing=0.1)

    compare_outlier_info = []

    for idx, (gen_col_name, label) in enumerate(gen_variants):
        row = idx // 2 + 1
        col = idx % 2 + 1

        if gen_col_name in df.columns:
            gen_vals = df[gen_col_name].values
            valid_mask = ~np.isnan(gen_vals)

            if valid_mask.sum() > 0:
                compare_hover = []
                for i in range(len(gen_vals)):
                    parts = [f"Gen={gen_vals[i]:.2f}", f"Val={val_scores[i]:.2f}"]
                    if has_noun2:
                        parts.append(f"Item={df['noun2'].iloc[i]}")
                    compare_hover.append(" | ".join(parts))
                compare_hover = np.array(compare_hover)
                compare_customdata = customdata

                # Compute outliers for this variant
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
                    var_type1_scores = np.zeros(n_var)
                    var_type2_scores = np.zeros(n_var)
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

                valid_indices = np.where(valid_mask)[0]
                n_outliers = min(40, len(valid_indices))
                sorted_by_score = np.argsort(var_outlier_scores)[::-1][:n_outliers]
                var_outlier_indices = valid_indices[sorted_by_score]
                var_outlier_set = set(var_outlier_indices)
                var_non_outlier_mask = np.array([i not in var_outlier_set for i in range(len(labels))])
                # Collect outlier words by type
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

                # Scatter traces
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

                # Outlier X markers
                compare_fig.add_trace(
                    go.Scatter(
                        x=gen_vals[var_outlier_indices], y=val_scores[var_outlier_indices],
                        mode='markers',
                        marker=dict(
                            color=[POS_CLASS_COLOR if labels[i] == 1 else NEG_CLASS_COLOR for i in var_outlier_indices],
                            size=6,
                            opacity=0.5
                        ),
                        name='Outliers',
                        showlegend=(idx == 0),
                        hovertext=compare_hover[var_outlier_indices], hoverinfo='text',
                        customdata=compare_customdata[var_outlier_indices] if compare_customdata is not None else None
                    ),
                    row=row, col=col
                )

                # y=x line in original space
                mean_gen = scaler_var.mean_[0]
                mean_val_sc = scaler_var.mean_[1]
                std_gen = scaler_var.scale_[0]
                std_val_sc = scaler_var.scale_[1]
                gen_min = gen_vals[valid_mask].min()
                gen_max = gen_vals[valid_mask].max()
                line_gen = np.array([gen_min, gen_max])
                line_val = mean_val_sc + std_val_sc * (line_gen - mean_gen) / std_gen
                compare_fig.add_trace(
                    go.Scatter(x=line_gen, y=line_val, mode='lines',
                               line=dict(color='gray', dash='dot', width=2),
                               showlegend=False, hoverinfo='skip'),
                    row=row, col=col
                )

                compare_fig.add_hline(y=threshold, line=dict(color='red', dash='dash'), row=row, col=col)

                # Metrics annotation
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

    # Outlier words annotation (for hypernym tasks)
    if task and task.startswith('hypernym-'):
        def format_colored_words(word_list, max_chars=120):
            lines = []
            current_line = []
            current_len = 0
            for word, is_pos in word_list:
                word_len = len(word) + 2
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

    compare_fig.update_layout(
        title='Compare Score Corrections',
        paper_bgcolor='white',
        plot_bgcolor='white',
        height=1100,
        margin=dict(b=150),
        showlegend=True
    )

    file_status = f"Loaded: {Path(csv_path).name}"
    return file_status, stats_text, main_fig, faceted_fig, pca_fig, compare_fig


# =============================================================================
# CALLBACKS - RESPONSE PANELS (ported as-is)
# =============================================================================

@app.callback(
    [Output('response-panel-main', 'children'),
     Output('response-panel-faceted', 'children'),
     Output('response-panel-pca', 'children'),
     Output('response-panel-compare', 'children'),
     Output('response-panel-multi-raw', 'children'),
     Output('response-panel-multi-fc', 'children')],
    [Input('main-scatter', 'clickData'),
     Input('faceted-plot', 'clickData'),
     Input('pca-plot', 'clickData'),
     Input('compare-plot', 'clickData'),
     Input('multi-prompt-raw-scatter', 'clickData'),
     Input('multi-prompt-fc-scatter', 'clickData')]
)
def update_response_panels(main_click, faceted_click, pca_click, compare_click, multi_raw_click, multi_fc_click):
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
        render_panel(compare_click, "Click a point in the compare plot to view the full prompt/response here."),
        render_panel(multi_raw_click, "Click a point in the raw multi-prompt plot to view prompt/response here."),
        render_panel(multi_fc_click, "Click a point in the FC multi-prompt plot to view prompt/response here."),
    )


# =============================================================================
# CALLBACKS - HEATMAPS
# =============================================================================

@app.callback(
    Output('all-heatmaps-container', 'children'),
    [Input('config-store', 'data'),
     Input('files-store', 'data')]
)
def generate_all_heatmaps(config, files_data):
    """Generate all heatmaps based on config."""
    if not config or not files_data:
        return html.Div("No data loaded", style={'color': '#999', 'textAlign': 'center', 'padding': '20px'})

    children = []

    # Reconstruct FileInfo objects from dicts
    file_infos = []
    for d in files_data:
        d2 = dict(d)
        d2.setdefault('is_base', False)
        d2.setdefault('has_tc_self_step', False)
        file_infos.append(FileInfo(**d2))

    all_tasks = sorted(set(f.task for f in file_infos))
    all_splits = sorted(set(f.split for f in file_infos))
    eval_columns = config.get('eval_columns', {})
    eval_cols = list(eval_columns.keys())
    metrics_list = config.get('metrics', DEFAULT_CONFIG['metrics'])
    aggregation_groups = config.get('aggregation_groups', {})

    # Resolve files and load data for all tasks/splits
    all_resolved = {}  # (task, split) -> {row_label: FileInfo}
    all_task_data = {}  # keyed by split -> task -> {row_label: {eval_col: metrics}}
    all_warnings = []

    for split in all_splits:
        all_task_data[split] = {}
        for task in all_tasks:
            resolved, warnings = resolve_files_for_task(file_infos, task, split)
            all_resolved[(task, split)] = resolved
            all_warnings.extend(warnings)

            # Load metrics
            data = load_heatmap_data_for_task(resolved, config)
            all_task_data[split][task] = data

    # Write file tracking log
    write_file_tracking_log(all_resolved, config)

    # Print warnings
    for w in all_warnings:
        print(w)

    # Get all visible row labels across all tasks/splits
    all_row_labels = set()
    for resolved in all_resolved.values():
        all_row_labels.update(resolved.keys())

    # Filter by visibility config
    visible_rows = sorted(
        [r for r in all_row_labels if is_row_visible(r, config)],
        key=row_sort_key
    )

    for split in all_splits:
        split_label = split.upper()
        split_color = '#2e7d32' if split == 'test' else '#c62828'
        split_bg = '#e8f5e9' if split == 'test' else '#ffebee'

        split_children = []

        split_children.append(html.H2(
            f'{"🧪" if split == "test" else "🏋️"} {split_label} SET RESULTS',
            style={'marginTop': '0px', 'marginBottom': '20px', 'color': split_color,
                   'borderBottom': f'3px solid {split_color}', 'paddingBottom': '10px',
                   'padding': '15px', 'borderRadius': '8px'}
        ))

        # === AGGREGATED SECTION ===
        for group_name, pattern in aggregation_groups.items():
            tasks_in_group = expand_aggregation_pattern(pattern, all_tasks)
            tasks_with_data = [t for t in tasks_in_group if t in all_task_data.get(split, {})]

            if len(tasks_with_data) > 1:
                try:
                    split_children.append(html.H3(
                        f'📊 {group_name} - {split_label} (Mean across {len(tasks_with_data)} tasks)',
                        style={'marginTop': '20px', 'marginBottom': '10px', 'color': '#1a5f7a',
                               'borderBottom': '2px solid #1a5f7a', 'paddingBottom': '10px'}
                    ))

                    # Aggregated heatmap
                    split_children.append(html.H4('Aggregated Heatmap', style={'marginTop': '15px', 'color': '#333'}))
                    fig_agg = build_aggregated_heatmap(
                        all_task_data[split], tasks_with_data, visible_rows, eval_cols,
                        f'Mean across datasets', metrics_list
                    )
                    if fig_agg is not None:
                        split_children.append(dcc.Graph(figure=fig_agg))

                    # Bar plots for all metrics
                    split_children.append(html.H4('Bar Plots with Standard Error', style={'marginTop': '25px', 'color': '#333'}))
                    for metric in metrics_list:
                        split_children.append(html.H5(f'{metric}', style={'marginTop': '15px', 'color': '#555'}))
                        fig_bar = build_bar_plot(
                            all_task_data[split], tasks_with_data, visible_rows, eval_cols,
                            metric, f'{metric}'
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
            task_data = all_task_data.get(split, {}).get(task, {})
            if not task_data:
                continue

            split_children.append(html.H4(
                f'{task}',
                style={'marginTop': '15px', 'marginBottom': '5px', 'color': '#333',
                       'borderBottom': '1px solid #ddd', 'paddingBottom': '5px'}
            ))

            try:
                fig = build_heatmap(
                    task_data, visible_rows, eval_cols,
                    f'{task}', metrics_list
                )
                if fig is not None:
                    split_children.append(dcc.Graph(figure=fig, style={'marginTop': '0px'}))
            except Exception as e:
                split_children.append(html.Div(
                    f"⚠️ Error generating heatmap for {task}: {e}",
                    style={'color': '#c62828', 'padding': '5px'}
                ))

        # Wrap split content
        if split == 'train':
            children.append(html.Div(
                split_children,
                style={'backgroundColor': split_bg, 'padding': '20px', 'borderRadius': '10px',
                       'marginTop': '20px', 'marginBottom': '20px'}
            ))
        else:
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
