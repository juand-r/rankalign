#!/usr/bin/env python3
"""
Plot comparison of correlation metrics across base, base+typcorr, and rankaligned models.

Creates a 3-row figure showing corr_all, corr_pos, and corr_neg for 8 nouns (bananas to dogs).
Creates a 4-row figure showing disc_acc, disc_roc, gen_mrr_pos_dataset, gen_roc.
Each noun has 6 bars: base, base+typcorr, G, G+typcorr, V, V+typcorr.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
NOUNS = ['bananas', 'bazookas', 'cabinets', 'cars', 'chairs', 'crows', 'diapers', 'dogs', 'concat-bananas-to-dogs']

# Correlation metrics (first plot)
CORR_METRICS = ['corr_all', 'corr_pos', 'corr_neg']
CORR_METRIC_LABELS = ['Correlation (All)', 'Correlation (Positive)', 'Correlation (Negative)']

# Accuracy/ROC metrics (second plot)
ACC_METRICS = ['disc_acc', 'disc_roc', 'gen_mrr_pos_dataset', 'gen_roc']
ACC_METRIC_LABELS = ['Discriminator Accuracy', 'Discriminator ROC', 'Generator MRR (Pos, Dataset)', 'Generator ROC']

# Column indices (0-indexed) based on the CSV structure
COL_MODEL = 0
COL_TASK = 1
COL_CORR_ALL = 2
COL_CORR_POS = 3
COL_CORR_NEG = 4
COL_DISC_ACC = 5
COL_DISC_ROC = 6
COL_GEN_MRR_POS_DATASET = 27
COL_GEN_ROC = 29

# Bar colors
COLORS = {
    'base': '#87CEEB',           # light blue
    'base_typcorr': '#00008B',   # dark blue
    'G': '#FFB6C1',              # light red/pink
    'G_typcorr': '#8B0000',      # dark red
    'V': '#90EE90',              # light green
    'V_typcorr': '#006400',      # dark green
}

BAR_LABELS = ['Base', 'Base + Typ', 'G', 'G + Typ', 'V', 'V + Typ']


def load_csv_no_header(filepath):
    """Load CSV without headers."""
    return pd.read_csv(filepath, header=None)


def extract_noun_from_task(task):
    """Extract noun from task name like 'hypernym-bananas'."""
    if task.startswith('hypernym-'):
        return task.replace('hypernym-', '')
    return task


def get_metric_values(df, noun, metric_col):
    """Get metric value for a noun from a dataframe."""
    task_name = f'hypernym-{noun}'
    mask = df[COL_TASK] == task_name
    if mask.any():
        return df.loc[mask, metric_col].values[0]
    return np.nan


def get_rankaligned_value(df, noun, alignment_type, typcorr, metric_col):
    """
    Get metric value from rankaligned dataframe.
    
    alignment_type: 'd2g' for G, 'g2d' for V
    typcorr: True/False for typicality correction
    """
    task_name = f'hypernym-{noun}'
    
    for idx, row in df.iterrows():
        model = str(row[COL_MODEL])
        task = str(row[COL_TASK])
        
        if task != task_name:
            continue
            
        has_d2g = '--d2g--' in model
        has_g2d = '--g2d--' in model
        has_typcorr = '--typcorr--' in model
        
        if alignment_type == 'd2g' and has_d2g:
            if typcorr == has_typcorr:
                return row[metric_col]
        elif alignment_type == 'g2d' and has_g2d:
            if typcorr == has_typcorr:
                return row[metric_col]
    
    return np.nan


def collect_data(df_base, df_base_typcorr, df_rankaligned, metrics, metric_cols):
    """Collect data for all metrics and nouns."""
    data = {metric: {noun: [] for noun in NOUNS} for metric in metrics}
    
    for metric in metrics:
        col = metric_cols[metric]
        for noun in NOUNS:
            values = []
            
            # Base
            values.append(get_metric_values(df_base, noun, col))
            
            # Base + typcorr
            values.append(get_metric_values(df_base_typcorr, noun, col))
            
            # G (d2g without typcorr)
            values.append(get_rankaligned_value(df_rankaligned, noun, 'd2g', False, col))
            
            # G + typcorr
            values.append(get_rankaligned_value(df_rankaligned, noun, 'd2g', True, col))
            
            # V (g2d without typcorr)
            values.append(get_rankaligned_value(df_rankaligned, noun, 'g2d', False, col))
            
            # V + typcorr
            values.append(get_rankaligned_value(df_rankaligned, noun, 'g2d', True, col))
            
            data[metric][noun] = values
    
    return data


def create_bar_plot(data, metrics, metric_labels, title, output_basename, base_dir, ylims=None):
    """Create and save a bar plot figure.
    
    ylims: optional dict mapping metric name to (ymin, ymax) tuple
    """
    n_rows = len(metrics)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3.5 * n_rows), sharex=True)
    
    # Handle single row case
    if n_rows == 1:
        axes = [axes]
    
    x = np.arange(len(NOUNS))
    bar_width = 0.12
    offsets = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]) * bar_width
    
    color_list = [
        COLORS['base'],
        COLORS['base_typcorr'],
        COLORS['G'],
        COLORS['G_typcorr'],
        COLORS['V'],
        COLORS['V_typcorr'],
    ]
    
    for ax_idx, (ax, metric, label) in enumerate(zip(axes, metrics, metric_labels)):
        for i, (offset, color, bar_label) in enumerate(zip(offsets, color_list, BAR_LABELS)):
            values = [data[metric][noun][i] for noun in NOUNS]
            ax.bar(x + offset, values, bar_width, label=bar_label, color=color, edgecolor='black', linewidth=0.5)
        
        ax.set_ylabel(label, fontsize=11)
        
        # Set y-limits (use custom if provided, else default to 0-1)
        if ylims and metric in ylims:
            ymin, ymax = ylims[metric]
            ax.set_ylim(ymin, ymax)
            ax.axhline(y=(ymin + ymax) / 2, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        else:
            ax.set_ylim(0, 1.0)
            ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Only show legend on first subplot
        if ax_idx == 0:
            ax.legend(loc='upper right', ncol=6, fontsize=9, framealpha=0.9)
    
    # Set x-axis labels on bottom subplot only
    axes[-1].set_xticks(x)
    # Format labels nicely
    labels = []
    for n in NOUNS:
        if n == 'concat-bananas-to-dogs':
            labels.append('Concat\n(Ban-Dogs)')
        else:
            labels.append(n.capitalize())
    axes[-1].set_xticklabels(labels, fontsize=10)
    axes[-1].set_xlabel('Noun Category', fontsize=11)
    
    # Title
    fig.suptitle(title, fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = base_dir / f'{output_basename}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to: {output_path}")
    
    # Also save as PDF
    output_path_pdf = base_dir / f'{output_basename}.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Saved figure to: {output_path_pdf}")
    
    plt.close()


def main():
    # File paths
    base_dir = Path('/datastor1/jdr/gv-gap/rankalign/outputs/tmp')
    base_file = base_dir / 'base.csv'
    base_typcorr_file = base_dir / 'base-typcorr.csv'
    rankaligned_file = base_dir / 'rankaligned.csv'
    
    # Load data
    df_base = load_csv_no_header(base_file)
    df_base_typcorr = load_csv_no_header(base_typcorr_file)
    df_rankaligned = load_csv_no_header(rankaligned_file)
    
    # ===== Plot 1: Correlation metrics =====
    corr_metric_cols = {
        'corr_all': COL_CORR_ALL,
        'corr_pos': COL_CORR_POS,
        'corr_neg': COL_CORR_NEG,
    }
    
    corr_data = collect_data(df_base, df_base_typcorr, df_rankaligned, CORR_METRICS, corr_metric_cols)
    create_bar_plot(
        corr_data, 
        CORR_METRICS, 
        CORR_METRIC_LABELS,
        'Hypernym Task: Correlation Metrics by Model Configuration',
        'hypernym_comparison_corr',
        base_dir
    )
    
    # ===== Plot 2: Accuracy/ROC metrics =====
    acc_metric_cols = {
        'disc_acc': COL_DISC_ACC,
        'disc_roc': COL_DISC_ROC,
        'gen_mrr_pos_dataset': COL_GEN_MRR_POS_DATASET,
        'gen_roc': COL_GEN_ROC,
    }
    
    acc_data = collect_data(df_base, df_base_typcorr, df_rankaligned, ACC_METRICS, acc_metric_cols)
    acc_ylims = {
        'gen_mrr_pos_dataset': (0, 0.2),
    }
    create_bar_plot(
        acc_data,
        ACC_METRICS,
        ACC_METRIC_LABELS,
        'Hypernym Task: Accuracy & ROC Metrics by Model Configuration',
        'hypernym_comparison_acc',
        base_dir,
        ylims=acc_ylims
    )


if __name__ == '__main__':
    main()

