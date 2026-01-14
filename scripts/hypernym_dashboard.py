#!/usr/bin/env python3
"""
Interactive dashboard for hypernym visualization.

Run with:
    cd /datastor1/jdr/gv-gap/rankalign/scripts
    source ~/venvs/venv_lexcons/bin/activate
    PORT=8888 python hypernym_dashboard.py

Then access via SSH port forwarding:
    ssh -L 8888:localhost:8888 <your-host>
    
Open in browser: http://localhost:8888
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Configuration
OUTPUTS_DIR = Path(__file__).parent.parent / 'outputs'
DATA_DIR = Path(__file__).parent.parent / 'data'
PORT = int(os.environ.get('PORT', 8888))

# All datasets
ALL_DATASETS = ['bananas', 'bazookas', 'cabinets', 'cars', 'chairs', 'crows', 'diapers', 'dogs']

# Heatmap configuration
MODEL_ROWS = ['Base', 'Rankalign', '+tc', '+lenorm', '+tc+lenorm']
EVAL_COLS = ['raw', 'tc', 'lenorm', 'tc+lenorm']
METRICS = ['Accuracy', 'Val ROC', 'Gen ROC', 'Correlation', 'Corr-Pos', 'Corr-Neg']

EVAL_COL_MAP = {
    'raw': 'gen_score',
    'tc': 'gen_score_typcorr',
    'lenorm': 'gen_score_lenorm',
    'tc+lenorm': 'gen_score_typcorr_lenorm',
}


def discover_scores_files():
    """Discover all hypernym scores CSV files and create clean display names."""
    files = []
    for csv_file in sorted(OUTPUTS_DIR.glob('scores_*hypernym*.csv')):
        # Parse the filename to extract meaningful info
        name = csv_file.stem
        
        # Extract dataset name
        dataset_match = re.search(r'hypernym-([a-zA-Z]+)', name)
        dataset = dataset_match.group(1) if dataset_match else 'unknown'
        
        # Determine model type
        if 'v5-google' in name:
            # Finetuned model
            if '_g2d_' in name:
                direction = 'G2V'
            elif '_d2g_' in name:
                direction = 'V2G'
            else:
                direction = 'FT'
            
            # Check for training flags (before _full-completion)
            has_typcorr = '_typcorr_full-completion' in name or '_typcorr_lenorm_full-completion' in name
            has_lenorm = '_lenorm_full-completion' in name or '_typcorr_lenorm_full-completion' in name
            
            if has_typcorr and has_lenorm:
                suffix = '+tc+lenorm'
            elif has_typcorr:
                suffix = '+tc'
            elif has_lenorm:
                suffix = '+lenorm'
            else:
                suffix = ''
            
            model_type = f'{direction}{suffix}'
        else:
            model_type = 'Base'
        
        display_name = f'{dataset} | {model_type}'
        files.append({'path': str(csv_file), 'display': display_name, 'value': str(csv_file)})
    
    return files


def load_scores_data(csv_path):
    """Load scores data from CSV file."""
    df = pd.read_csv(csv_path)
    
    # Convert ground truth to binary label
    if 'gpt4_ground_truth' in df.columns:
        df['label'] = df['gpt4_ground_truth'].str.strip().str.lower().map({'yes': 1, 'no': 0})
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


def parse_model_config(filename):
    """Parse model configuration from filename."""
    name = filename if isinstance(filename, str) else filename.stem
    
    is_finetuned = 'v5-google' in name
    
    if not is_finetuned:
        dataset_match = re.search(r'hypernym-([a-zA-Z]+)', name)
        dataset_name = dataset_match.group(1) if dataset_match else 'unknown'
        return 'base', False, False, False, dataset_name
    
    if '_g2d_' in name:
        direction = 'g2d'
    elif '_d2g_' in name:
        direction = 'd2g'
    else:
        direction = 'unknown'
    
    # Check for typcorr and lenorm training flags (before _full-completion)
    has_typcorr = '_typcorr_full-completion' in name or '_typcorr_lenorm_full-completion' in name
    has_lenorm = '_lenorm_full-completion' in name or '_typcorr_lenorm_full-completion' in name
    
    dataset_match = re.search(r'hypernym-([a-zA-Z]+)', name)
    dataset_name = dataset_match.group(1) if dataset_match else 'unknown'
    
    return direction, is_finetuned, has_typcorr, has_lenorm, dataset_name


def get_model_row_label(is_finetuned, has_typcorr, has_lenorm):
    """Map model config to row label."""
    if not is_finetuned:
        return 'Base'
    elif has_typcorr and has_lenorm:
        return '+tc+lenorm'
    elif has_typcorr:
        return '+tc'
    elif has_lenorm:
        return '+lenorm'
    else:
        return 'Rankalign'


def compute_metrics_for_heatmap(df, gen_col, metric_type='log-odds'):
    """Compute metrics for heatmap cell."""
    if gen_col not in df.columns or 'val_score' not in df.columns:
        return None
    
    gen_scores = df[gen_col].values
    val_scores = df['val_score'].values
    labels = df['label'].values
    
    return compute_metrics(gen_scores, val_scores, labels, metric_type)


def discover_heatmap_data(dataset_name):
    """Discover and load heatmap data for a dataset."""
    data = {
        'base': {row: {col: None for col in EVAL_COLS} for row in MODEL_ROWS},
        'd2g': {row: {col: None for col in EVAL_COLS} for row in MODEL_ROWS},
        'g2d': {row: {col: None for col in EVAL_COLS} for row in MODEL_ROWS},
    }
    
    for csv_file in OUTPUTS_DIR.glob(f"scores_*hypernym-{dataset_name}*.csv"):
        direction, is_finetuned, has_typcorr, has_lenorm, ds_name = parse_model_config(csv_file.stem)
        model_row = get_model_row_label(is_finetuned, has_typcorr, has_lenorm)
        metric_type = 'log-odds' if 'log-odds' in csv_file.stem else 'log-probs'
        
        try:
            df = pd.read_csv(csv_file)
            if 'gpt4_ground_truth' in df.columns:
                df['label'] = df['gpt4_ground_truth'].str.strip().str.lower().map({'yes': 1, 'no': 0})
                df['label'] = df['label'].fillna(0).astype(int)
            
            for eval_col, gen_col in EVAL_COL_MAP.items():
                metrics = compute_metrics_for_heatmap(df, gen_col, metric_type=metric_type)
                
                if is_finetuned:
                    if direction == 'd2g':
                        data['d2g'][model_row][eval_col] = metrics
                    elif direction == 'g2d':
                        data['g2d'][model_row][eval_col] = metrics
                else:
                    data['base'][model_row][eval_col] = metrics
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue
    
    # Merge base model data into d2g and g2d
    for eval_col in EVAL_COLS:
        if data['base']['Base'][eval_col] is not None:
            data['d2g']['Base'][eval_col] = data['base']['Base'][eval_col]
            data['g2d']['Base'][eval_col] = data['base']['Base'][eval_col]
    
    return data


def create_heatmap_figure(heatmap_data, direction, metric_type):
    """Create a heatmap figure for one direction."""
    dir_label = 'V2G' if direction == 'd2g' else 'G2V'
    
    # Create subplots for each metric
    fig = make_subplots(rows=1, cols=6, subplot_titles=METRICS,
                        horizontal_spacing=0.03)
    
    for m_idx, metric in enumerate(METRICS):
        metric_key = {'Accuracy': 'acc', 'Val ROC': 'val_roc', 'Gen ROC': 'gen_roc',
                      'Correlation': 'corr', 'Corr-Pos': 'corr_pos', 'Corr-Neg': 'corr_neg'}[metric]
        
        z = []
        text = []
        for row in MODEL_ROWS:
            z_row = []
            text_row = []
            for col in EVAL_COLS:
                metrics = heatmap_data[direction][row][col]
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
                z=z, x=EVAL_COLS, y=MODEL_ROWS,
                text=text, texttemplate='%{text}', textfont={'size': 10},
                colorscale='RdYlGn', zmin=0, zmax=100,
                showscale=(m_idx == 5),
                hovertemplate='%{y} / %{x}: %{z:.1f}<extra></extra>'
            ),
            row=1, col=m_idx + 1
        )
    
    # Only show y-axis labels on the first (leftmost) heatmap
    for m_idx in range(6):
        if m_idx == 0:
            fig.update_yaxes(showticklabels=True, row=1, col=m_idx + 1)
        else:
            fig.update_yaxes(showticklabels=False, row=1, col=m_idx + 1)
    
    fig.update_layout(
        title=f'{dir_label} Models',
        height=250,
        margin=dict(l=80, r=20, t=50, b=30),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig


def create_aggregated_heatmap(all_heatmap_data, direction):
    """Create aggregated heatmap showing mean across all datasets."""
    dir_label = 'V2G' if direction == 'd2g' else 'G2V'
    
    fig = make_subplots(rows=1, cols=6, subplot_titles=METRICS,
                        horizontal_spacing=0.03)
    
    for m_idx, metric in enumerate(METRICS):
        metric_key = {'Accuracy': 'acc', 'Val ROC': 'val_roc', 'Gen ROC': 'gen_roc',
                      'Correlation': 'corr', 'Corr-Pos': 'corr_pos', 'Corr-Neg': 'corr_neg'}[metric]
        
        z = []
        text = []
        for row in MODEL_ROWS:
            z_row = []
            text_row = []
            for col in EVAL_COLS:
                values = []
                for dataset_name in ALL_DATASETS:
                    hm_data = all_heatmap_data[dataset_name]
                    metrics = hm_data[direction][row][col]
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
                z=z, x=EVAL_COLS, y=MODEL_ROWS,
                text=text, texttemplate='%{text}', textfont={'size': 10},
                colorscale='RdYlGn', zmin=0, zmax=100,
                showscale=(m_idx == 5),
                hovertemplate='%{y} / %{x}: %{z:.1f}<extra></extra>'
            ),
            row=1, col=m_idx + 1
        )
    
    # Only show y-axis labels on the first (leftmost) heatmap
    for m_idx in range(6):
        if m_idx == 0:
            fig.update_yaxes(showticklabels=True, row=1, col=m_idx + 1)
        else:
            fig.update_yaxes(showticklabels=False, row=1, col=m_idx + 1)
    
    fig.update_layout(
        title=f'{dir_label} Models (Mean across datasets)',
        height=250,
        margin=dict(l=80, r=20, t=50, b=30),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig


def create_aggregated_bar_plot(all_heatmap_data, metric, direction):
    """Create bar plot with standard error for a single metric."""
    dir_label = 'V2G' if direction == 'd2g' else 'G2V'
    metric_key = {'Accuracy': 'acc', 'Val ROC': 'val_roc', 'Gen ROC': 'gen_roc',
                  'Correlation': 'corr', 'Corr-Pos': 'corr_pos', 'Corr-Neg': 'corr_neg'}[metric]
    
    fig = go.Figure()
    
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
    
    x_positions = []
    x_labels = []
    current_x = 0
    
    for row_idx, row in enumerate(MODEL_ROWS):
        for col_idx, col in enumerate(EVAL_COLS):
            values = []
            for dataset_name in ALL_DATASETS:
                hm_data = all_heatmap_data[dataset_name]
                metrics = hm_data[direction][row][col]
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
                marker_color=colors[row_idx],
                name=row if col_idx == 0 else None,
                showlegend=(col_idx == 0),
                legendgroup=row,
                hovertemplate=f'{row} / {col}: {mean_val:.1f} ± {std_err:.1f}<extra></extra>'
            ))
            
            x_positions.append(current_x)
            x_labels.append(col)
            current_x += 1
        
        current_x += 0.5  # Gap between model groups
    
    fig.update_layout(
        title=f'{dir_label} - {metric}',
        xaxis=dict(
            tickvals=x_positions,
            ticktext=x_labels,
            tickangle=45
        ),
        yaxis=dict(title=metric),
        height=300,
        margin=dict(l=60, r=20, t=50, b=80),
        paper_bgcolor='white',
        plot_bgcolor='white',
        barmode='overlay',
        showlegend=True,
        legend=dict(orientation='h', y=1.15)
    )
    
    return fig


# Initialize Dash app
app = dash.Dash(__name__)

# Discover files
scores_files = discover_scores_files()

app.layout = html.Div([
    html.H1('Hypernym Visualization Dashboard', style={'textAlign': 'center', 'color': '#333'}),
    
    # File selector
    html.Div([
        html.Label('Select Scores File:', style={'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='file-selector',
            options=[{'label': f['display'], 'value': f['value']} for f in scores_files],
            value=scores_files[0]['value'] if scores_files else None,
            style={'width': '100%'}
        )
    ], style={'width': '60%', 'margin': '20px auto'}),
    
    # Stats panel
    html.Div(id='stats-panel', style={
        'width': '60%', 'margin': '10px auto', 'padding': '15px',
        'backgroundColor': '#f5f5f5', 'borderRadius': '8px',
        'fontFamily': 'monospace', 'whiteSpace': 'pre-wrap'
    }),
    
    # Main scatter plot with marginals
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
        html.Summary('🔥 Heatmaps (All Datasets)', style={'cursor': 'pointer', 'fontWeight': 'bold'}),
        html.Div(id='all-heatmaps-container')
    ], open=True, style={'margin': '20px'}),
    
], style={'padding': '20px', 'backgroundColor': 'white', 'minHeight': '100vh'})


@app.callback(
    [Output('stats-panel', 'children'),
     Output('main-scatter', 'figure'),
     Output('faceted-plot', 'figure'),
     Output('pca-plot', 'figure'),
     Output('compare-plot', 'figure')],
    [Input('file-selector', 'value')]
)
def update_visualizations(csv_path):
    if not csv_path:
        empty_fig = go.Figure()
        return "No file selected", empty_fig, empty_fig, empty_fig, empty_fig
    
    # Load data
    df = load_scores_data(csv_path)
    
    # Determine metric type
    metric_type = 'log-odds' if 'log-odds' in csv_path else 'log-probs'
    metric_label = 'log-odds' if metric_type == 'log-odds' else 'log-probs'
    threshold = 0 if metric_type == 'log-odds' else np.log(0.5)
    
    gen_scores = df['gen_score'].values
    val_scores = df['val_score'].values
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
    
    # Main scatter
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
    
    # Threshold line
    main_fig.add_hline(y=threshold, line=dict(color='red', dash='dash', width=2),
                       row=2, col=1)
    
    # Outliers
    X = np.column_stack([gen_scores, val_scores])
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    distances = np.sqrt(X_std[:, 0]**2 + X_std[:, 1]**2)
    outlier_indices = np.argsort(distances)[-40:]
    
    # Add outlier markers
    outlier_colors = ['orange' if labels[i] == 1 else 'purple' for i in outlier_indices]
    main_fig.add_trace(
        go.Scatter(
            x=gen_scores[outlier_indices], y=val_scores[outlier_indices],
            mode='markers+text',
            marker=dict(symbol='x', size=12, color=outlier_colors, line=dict(width=2)),
            text=[f"{df['noun1'].iloc[i][:6]}/{df['noun2'].iloc[i][:6]}" 
                  if 'noun1' in df.columns else '' for i in outlier_indices],
            textposition='top right', textfont=dict(size=8),
            name='Outliers', showlegend=False
        ),
        row=2, col=1
    )
    
    # X histogram (top)
    main_fig.add_trace(
        go.Histogram(x=gen_scores[pos_mask], marker_color='orange', opacity=0.6, name='Pos', showlegend=False),
        row=1, col=1
    )
    main_fig.add_trace(
        go.Histogram(x=gen_scores[neg_mask], marker_color='blue', opacity=0.6, name='Neg', showlegend=False),
        row=1, col=1
    )
    
    # Y histogram (right)
    main_fig.add_trace(
        go.Histogram(y=val_scores[pos_mask], marker_color='orange', opacity=0.6, showlegend=False),
        row=2, col=2
    )
    main_fig.add_trace(
        go.Histogram(y=val_scores[neg_mask], marker_color='blue', opacity=0.6, showlegend=False),
        row=2, col=2
    )
    
    main_fig.update_layout(
        title='Generator vs Validator Scores',
        paper_bgcolor='white', plot_bgcolor='white',
        showlegend=True
    )
    main_fig.update_xaxes(title_text='Generator log-probs', row=2, col=1, showgrid=True, gridcolor='lightgray')
    main_fig.update_yaxes(title_text=f'Validator {metric_label}', row=2, col=1, showgrid=True, gridcolor='lightgray')
    
    # === FACETED PLOT BY STRATEGY ===
    unique_strategies = sorted(set(strategies))
    n_strats = len(unique_strategies)
    n_cols = min(3, n_strats)
    n_rows = (n_strats + n_cols - 1) // n_cols
    
    faceted_fig = make_subplots(rows=n_rows, cols=n_cols, 
                                 subplot_titles=unique_strategies)
    
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
            go.Scatter(x=gen_scores[strat_pos], y=val_scores[strat_pos],
                       mode='markers', marker=dict(color='orange', size=6, opacity=0.6),
                       name=f'Pos ({total_pos})', showlegend=(idx == 0)),
            row=row, col=col
        )
        faceted_fig.add_trace(
            go.Scatter(x=gen_scores[strat_neg], y=val_scores[strat_neg],
                       mode='markers', marker=dict(color='blue', size=6, opacity=0.6),
                       name=f'Neg ({strat_neg.sum()})', showlegend=(idx == 0)),
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
    
    # Set shared axis limits with padding - explicitly set for each subplot
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    x_range = [x_min - x_pad, x_max + x_pad]
    y_range = [y_min - y_pad, y_max + y_pad]
    
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            faceted_fig.update_xaxes(range=x_range, showgrid=True, gridcolor='lightgray', row=r, col=c)
            faceted_fig.update_yaxes(range=y_range, showgrid=True, gridcolor='lightgray', row=r, col=c)
    
    faceted_fig.update_layout(
        title='Faceted by Strategy',
        paper_bgcolor='white', plot_bgcolor='white',
        height=400 * n_rows
    )
    
    # === PCA PLOT ===
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)
    
    pca_fig = make_subplots(rows=1, cols=2,
                             subplot_titles=['Standardized Scores', 'PCA'])
    
    # Left: standardized
    pca_fig.add_trace(
        go.Scatter(x=X_std[pos_mask, 0], y=X_std[pos_mask, 1],
                   mode='markers', marker=dict(color='orange', size=6, opacity=0.5),
                   name='Positive'),
        row=1, col=1
    )
    pca_fig.add_trace(
        go.Scatter(x=X_std[neg_mask, 0], y=X_std[neg_mask, 1],
                   mode='markers', marker=dict(color='blue', size=6, opacity=0.5),
                   name='Negative'),
        row=1, col=1
    )
    pca_fig.add_trace(
        go.Scatter(x=X_std[outlier_indices, 0], y=X_std[outlier_indices, 1],
                   mode='markers', marker=dict(symbol='x', size=10, color=outlier_colors),
                   name='Outliers', showlegend=False),
        row=1, col=1
    )
    
    # Right: PCA
    pca_fig.add_trace(
        go.Scatter(x=X_pca[pos_mask, 0], y=X_pca[pos_mask, 1],
                   mode='markers', marker=dict(color='orange', size=6, opacity=0.5),
                   showlegend=False),
        row=1, col=2
    )
    pca_fig.add_trace(
        go.Scatter(x=X_pca[neg_mask, 0], y=X_pca[neg_mask, 1],
                   mode='markers', marker=dict(color='blue', size=6, opacity=0.5),
                   showlegend=False),
        row=1, col=2
    )
    pca_fig.add_trace(
        go.Scatter(x=X_pca[outlier_indices, 0], y=X_pca[outlier_indices, 1],
                   mode='markers', marker=dict(symbol='x', size=10, color=outlier_colors),
                   showlegend=False),
        row=1, col=2
    )
    
    pca_fig.update_layout(
        paper_bgcolor='white', plot_bgcolor='white'
    )
    pca_fig.update_xaxes(title_text='Generator (std)', row=1, col=1, showgrid=True, gridcolor='lightgray')
    pca_fig.update_yaxes(title_text='Validator (std)', row=1, col=1, showgrid=True, gridcolor='lightgray')
    pca_fig.update_xaxes(title_text=f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', row=1, col=2, showgrid=True, gridcolor='lightgray')
    pca_fig.update_yaxes(title_text=f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', row=1, col=2, showgrid=True, gridcolor='lightgray')
    
    # === COMPARE CORRECTIONS 2x2 ===
    gen_variants = [
        ('gen_score', 'Raw'),
        ('gen_score_typcorr', 'Typicality Corrected'),
        ('gen_score_lenorm', 'Length Normalized'),
        ('gen_score_typcorr_lenorm', 'Typcorr + Lenorm'),
    ]
    
    compare_fig = make_subplots(rows=2, cols=2, subplot_titles=[v[1] for v in gen_variants])
    
    for idx, (gen_col, label) in enumerate(gen_variants):
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        if gen_col in df.columns:
            gen_vals = df[gen_col].values
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
    compare_fig.update_layout(
        title='Compare Score Corrections',
        paper_bgcolor='white', plot_bgcolor='white'
    )
    
    return stats_text, main_fig, faceted_fig, pca_fig, compare_fig


@app.callback(
    Output('all-heatmaps-container', 'children'),
    [Input('file-selector', 'options')]
)
def generate_all_heatmaps(_options):
    children = []
    
    # Load all heatmap data
    all_heatmap_data = {}
    for dataset_name in ALL_DATASETS:
        all_heatmap_data[dataset_name] = discover_heatmap_data(dataset_name)
    
    # === AGGREGATED SECTION ===
    children.append(html.H3(
        '📊 Aggregated Results (Mean across all datasets)',
        style={'marginTop': '10px', 'marginBottom': '10px', 'color': '#1a5f7a',
               'borderBottom': '2px solid #1a5f7a', 'paddingBottom': '10px'}
    ))
    
    # Aggregated heatmaps
    children.append(html.H4('Aggregated Heatmaps', style={'marginTop': '15px', 'color': '#333'}))
    fig_agg_d2g = create_aggregated_heatmap(all_heatmap_data, 'd2g')
    fig_agg_g2d = create_aggregated_heatmap(all_heatmap_data, 'g2d')
    children.append(dcc.Graph(figure=fig_agg_d2g, style={'height': '280px'}))
    children.append(dcc.Graph(figure=fig_agg_g2d, style={'height': '280px'}))
    
    # Bar plots
    children.append(html.H4('Bar Plots with Standard Error', style={'marginTop': '25px', 'color': '#333'}))
    
    for metric in METRICS:
        children.append(html.H5(f'{metric}', style={'marginTop': '15px', 'color': '#555'}))
        fig_bar_d2g = create_aggregated_bar_plot(all_heatmap_data, metric, 'd2g')
        fig_bar_g2d = create_aggregated_bar_plot(all_heatmap_data, metric, 'g2d')
        children.append(dcc.Graph(figure=fig_bar_d2g, style={'height': '320px'}))
        children.append(dcc.Graph(figure=fig_bar_g2d, style={'height': '320px'}))
    
    # === PER-DATASET SECTION ===
    children.append(html.H3(
        '📋 Per-Dataset Results',
        style={'marginTop': '30px', 'marginBottom': '10px', 'color': '#1a5f7a',
               'borderBottom': '2px solid #1a5f7a', 'paddingBottom': '10px'}
    ))
    
    for dataset_name in ALL_DATASETS:
        children.append(html.H4(
            f'{dataset_name.capitalize()}',
            style={'marginTop': '15px', 'marginBottom': '5px', 'color': '#333',
                   'borderBottom': '1px solid #ddd', 'paddingBottom': '5px'}
        ))
        
        heatmap_data = all_heatmap_data[dataset_name]
        
        fig_d2g = create_heatmap_figure(heatmap_data, 'd2g', 'log-odds')
        fig_g2d = create_heatmap_figure(heatmap_data, 'g2d', 'log-odds')
        
        children.append(dcc.Graph(figure=fig_d2g, style={'height': '280px', 'marginTop': '0px'}))
        children.append(dcc.Graph(figure=fig_g2d, style={'height': '280px', 'marginTop': '0px'}))
    
    return children


if __name__ == '__main__':
    print(f"Starting dashboard on port {PORT}")
    print(f"Access via: http://localhost:{PORT}")
    print(f"Make sure to set up SSH port forwarding: ssh -L {PORT}:localhost:{PORT} <host>")
    app.run(host='0.0.0.0', port=PORT, debug=False)
