#!/usr/bin/env python3
"""
Standalone visualization script for scores CSV files.

Usage:
    python viz_scores.py <scores_csv_file> [--output <output_file>]

Example:
    python viz_scores.py ../outputs/scores_gemma-2-2b_hypernym-diapers_test_v2_20260112_200603.csv
"""

import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Marker styles for different strategies
STRATEGY_MARKERS = {
    'single_token': 'o',      # circle
    '2token': 's',            # square
    'beam_search': '^',       # triangle up
    'gpt4_positive': 'D',     # diamond
    'gpt4_negative': 'v',     # triangle down
    'existing_hypernyms': 'p', # pentagon
    'unknown': 'x',           # x
}

# Default marker for unknown strategies
DEFAULT_MARKER = '.'


def create_visualization_with_marginals(gen_scores, val_scores, labels, strategies, 
                                         metric_type='log-probs', output_file=None, title=None,
                                         show_histograms=False):
    """
    Create scatter plot with optional marginal histograms, colored by label.
    """
    gen_scores_np = np.array([float(x) for x in gen_scores])
    val_scores_np = np.array([float(x) for x in val_scores])
    labels_np = np.array(labels)
    
    if show_histograms:
        # Create figure with marginal histograms
        fig = plt.figure(figsize=(12, 10))
        ax_main = fig.add_axes([0.1, 0.1, 0.65, 0.65])
        ax_hist_x = fig.add_axes([0.1, 0.77, 0.65, 0.15], sharex=ax_main)
        ax_hist_y = fig.add_axes([0.77, 0.1, 0.15, 0.65], sharey=ax_main)
    else:
        # Create simple figure without histograms
        fig, ax_main = plt.subplots(figsize=(10, 8))
        ax_hist_x = None
        ax_hist_y = None
    
    # Masks for labels
    pos_mask = labels_np == 1
    neg_mask = labels_np == 0
    
    # Plot all points with simple circles, colored by label
    ax_main.scatter(gen_scores_np[pos_mask], val_scores_np[pos_mask],
                   c='orange', marker='o', alpha=0.6, s=30, edgecolors='none', label='Positive')
    ax_main.scatter(gen_scores_np[neg_mask], val_scores_np[neg_mask],
                   c='blue', marker='o', alpha=0.6, s=30, edgecolors='none', label='Negative')
    
    # Add threshold line
    metric_label = 'log-odds' if metric_type == 'log-odds' else 'log-probs'
    if metric_type == 'log-odds':
        threshold = 0
        ax_main.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    else:
        threshold = np.log(0.5)
        ax_main.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Generator is always log-probs, only validator changes based on metric_type
    ax_main.set_xlabel('Generator log-probs', fontsize=12)
    ax_main.set_ylabel(f'Validator {metric_label}', fontsize=12)
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(loc='upper left', title='Label')
    
    # Calculate and display metrics
    # Pearson correlation (all, pos, neg)
    corr_all, _ = pearsonr(gen_scores_np, val_scores_np)
    
    if pos_mask.sum() > 1:
        corr_pos, _ = pearsonr(gen_scores_np[pos_mask], val_scores_np[pos_mask])
    else:
        corr_pos = np.nan
    
    if neg_mask.sum() > 1:
        corr_neg, _ = pearsonr(gen_scores_np[neg_mask], val_scores_np[neg_mask])
    else:
        corr_neg = np.nan
    
    # Accuracy (predict positive if val_score > threshold)
    preds = (val_scores_np > threshold).astype(int)
    acc = accuracy_score(labels_np, preds)
    
    # ROC AUC for validator
    try:
        val_roc = roc_auc_score(labels_np, val_scores_np)
    except ValueError:
        val_roc = np.nan  # If only one class present
    
    # ROC AUC for generator
    try:
        gen_roc = roc_auc_score(labels_np, gen_scores_np)
    except ValueError:
        gen_roc = np.nan  # If only one class present
    
    # Add metrics text box (values as percentages)
    metrics_text = f'corr = {corr_all*100:.1f}\ncorr-pos = {corr_pos*100:.1f}\ncorr-neg = {corr_neg*100:.1f}\nAccuracy = {acc*100:.1f}\nVal ROC = {val_roc*100:.1f}\nGen ROC = {gen_roc*100:.1f}'
    ax_main.text(0.98, 0.02, metrics_text, transform=ax_main.transAxes, fontsize=10,
                 verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Marginal histograms - colored by label (only if show_histograms)
    if show_histograms and ax_hist_x is not None and ax_hist_y is not None:
        bins_x = 50
        bins_y = 50
        
        # X marginal (generator scores)
        ax_hist_x.hist(gen_scores_np[pos_mask], bins=bins_x, alpha=0.6, color='orange', label='Positive')
        ax_hist_x.hist(gen_scores_np[neg_mask], bins=bins_x, alpha=0.6, color='blue', label='Negative')
        ax_hist_x.set_ylabel('Count')
        ax_hist_x.tick_params(labelbottom=False)
        
        # Y marginal (validator scores)
        ax_hist_y.hist(val_scores_np[pos_mask], bins=bins_y, alpha=0.6, color='orange', orientation='horizontal')
        ax_hist_y.hist(val_scores_np[neg_mask], bins=bins_y, alpha=0.6, color='blue', orientation='horizontal')
        ax_hist_y.set_xlabel('Count')
        ax_hist_y.tick_params(labelleft=False)
        
        # Add threshold line to y marginal
        ax_hist_y.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    if title:
        if show_histograms:
            fig.suptitle(title, fontsize=14, y=0.98)
        else:
            ax_main.set_title(title, fontsize=14)
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    plt.close()


def create_faceted_by_strategy(gen_scores, val_scores, labels, strategies,
                                metric_type='log-probs', output_file=None, title=None):
    """
    Create faceted plot - one subplot per strategy.
    """
    gen_scores_np = np.array([float(x) for x in gen_scores])
    val_scores_np = np.array([float(x) for x in val_scores])
    labels_np = np.array(labels)
    strategies_np = np.array(strategies)
    
    unique_strategies = sorted(set(strategies))
    n_strategies = len(unique_strategies)
    
    # Determine grid size
    n_cols = min(3, n_strategies)
    n_rows = (n_strategies + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)
    
    # Compute global axis limits
    x_min, x_max = gen_scores_np.min(), gen_scores_np.max()
    y_min, y_max = val_scores_np.min(), val_scores_np.max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    
    metric_label = 'log-odds' if metric_type == 'log-odds' else 'log-probs'
    threshold = 0 if metric_type == 'log-odds' else np.log(0.5)
    
    for idx, strategy in enumerate(unique_strategies):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        strat_mask = strategies_np == strategy
        pos_mask = (labels_np == 1) & strat_mask
        neg_mask = (labels_np == 0) & strat_mask
        
        # Count misclassified positives (below threshold)
        pos_below_threshold = ((labels_np == 1) & strat_mask & (val_scores_np < threshold)).sum()
        total_pos = pos_mask.sum()
        
        ax.scatter(gen_scores_np[pos_mask], val_scores_np[pos_mask],
                   c='orange', alpha=0.6, s=30, label=f'Pos ({total_pos})')
        ax.scatter(gen_scores_np[neg_mask], val_scores_np[neg_mask],
                   c='blue', alpha=0.6, s=30, label=f'Neg ({neg_mask.sum()})')
        
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_xlabel(f'Generator {metric_label}', fontsize=10)
        ax.set_ylabel(f'Validator {metric_label}', fontsize=10)
        ax.set_title(f'{strategy}\n(Pos below thresh: {pos_below_threshold}/{total_pos})', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # Hide empty subplots
    for idx in range(n_strategies, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    # Modify output filename for faceted version
    output_path = Path(output_file)
    faceted_output = output_path.parent / f"{output_path.stem}_faceted{output_path.suffix}"
    
    plt.savefig(faceted_output, dpi=150, bbox_inches='tight')
    print(f"Faceted visualization saved to: {faceted_output}")
    plt.close()


def load_scores_csv(csv_path):
    """
    Load scores from CSV file.
    
    Returns:
        gen_scores, val_scores, labels, strategies, metric_type
    """
    gen_scores = []
    val_scores = []
    labels = []
    strategies = []
    metric_type = 'log-probs'  # default
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gen_scores.append(float(row['gen_score']))
            val_scores.append(float(row['val_score']))
            
            # Convert ground truth to binary label
            gt = row['gpt4_ground_truth'].strip().lower()
            labels.append(1 if gt == 'yes' else 0)
            
            # Get strategy
            strategies.append(row.get('strategy', 'unknown'))
            
            # Detect metric type from filename or strategy column (legacy)
            # This is now detected from filename in main()
    
    return gen_scores, val_scores, labels, strategies, metric_type


def create_pca_visualization(gen_scores, val_scores, labels, strategies, noun1_list, noun2_list,
                              output_file=None, title=None, n_outliers=10):
    """
    Create PCA visualization with standardized scores and highlight outliers.
    
    Standardization: subtract mean, divide by std for each dimension.
    PCA: project to 2D (though with 2 features, PCA just rotates).
    Outliers: points furthest from origin in standardized space.
    """
    gen_scores_np = np.array([float(x) for x in gen_scores])
    val_scores_np = np.array([float(x) for x in val_scores])
    labels_np = np.array(labels)
    
    # Stack into 2D array and standardize
    X = np.column_stack([gen_scores_np, val_scores_np])
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    # PCA (with 2 features, this is essentially a rotation)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)
    
    # Find outliers (furthest from origin in standardized space)
    distances = np.sqrt(X_std[:, 0]**2 + X_std[:, 1]**2)
    outlier_indices = np.argsort(distances)[-n_outliers:]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left plot: Standardized scores
    ax1 = axes[0]
    pos_mask = labels_np == 1
    neg_mask = labels_np == 0
    
    ax1.scatter(X_std[pos_mask, 0], X_std[pos_mask, 1], c='orange', alpha=0.5, s=30, label='Positive')
    ax1.scatter(X_std[neg_mask, 0], X_std[neg_mask, 1], c='blue', alpha=0.5, s=30, label='Negative')
    
    # Highlight outliers (colored by label: orange=pos, purple=neg)
    outlier_colors = ['orange' if labels_np[i] == 1 else 'purple' for i in outlier_indices]
    ax1.scatter(X_std[outlier_indices, 0], X_std[outlier_indices, 1], 
                c=outlier_colors, s=100, marker='x', linewidths=2, label=f'Top {n_outliers} outliers')
    
    # Annotate outliers
    for idx in outlier_indices:
        label_text = f"{noun1_list[idx][:8]}/{noun2_list[idx][:8]}"
        ax1.annotate(label_text, (X_std[idx, 0], X_std[idx, 1]), fontsize=7,
                     xytext=(5, 5), textcoords='offset points', alpha=0.8)
    
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Generator (standardized)', fontsize=12)
    ax1.set_ylabel('Validator (standardized)', fontsize=12)
    ax1.set_title('Standardized Scores', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Right plot: PCA
    ax2 = axes[1]
    ax2.scatter(X_pca[pos_mask, 0], X_pca[pos_mask, 1], c='orange', alpha=0.5, s=30, label='Positive')
    ax2.scatter(X_pca[neg_mask, 0], X_pca[neg_mask, 1], c='blue', alpha=0.5, s=30, label='Negative')
    
    # Highlight outliers (colored by label: orange=pos, purple=neg)
    ax2.scatter(X_pca[outlier_indices, 0], X_pca[outlier_indices, 1], 
                c=outlier_colors, s=100, marker='x', linewidths=2, label=f'Top {n_outliers} outliers')
    
    # Annotate outliers
    for idx in outlier_indices:
        label_text = f"{noun1_list[idx][:8]}/{noun2_list[idx][:8]}"
        ax2.annotate(label_text, (X_pca[idx, 0], X_pca[idx, 1]), fontsize=7,
                     xytext=(5, 5), textcoords='offset points', alpha=0.8)
    
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax2.set_title('PCA of Standardized Scores', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    # Modify output filename for PCA version
    output_path = Path(output_file)
    pca_output = output_path.parent / f"{output_path.stem}_pca{output_path.suffix}"
    
    plt.savefig(pca_output, dpi=150, bbox_inches='tight')
    print(f"PCA visualization saved to: {pca_output}")
    plt.close()
    
    # Print outlier details
    print(f"\nTop {n_outliers} outliers (furthest from mean in standardized space):")
    for i, idx in enumerate(reversed(outlier_indices)):
        label = "POS" if labels_np[idx] == 1 else "NEG"
        print(f"  {i+1}. [{label}] {noun1_list[idx]} / {noun2_list[idx]} | "
              f"gen={gen_scores_np[idx]:.2f}, val={val_scores_np[idx]:.2f}, dist={distances[idx]:.2f}")


def load_scores_csv_full(csv_path):
    """
    Load scores from CSV file with all fields including correction columns.
    """
    gen_scores = []
    gen_scores_typcorr = []
    gen_scores_lenorm = []
    gen_scores_typcorr_lenorm = []
    val_scores = []
    labels = []
    strategies = []
    noun1_list = []
    noun2_list = []
    num_tokens_list = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gen_scores.append(float(row['gen_score']))
            val_scores.append(float(row['val_score']))
            
            # Load correction columns (may be NaN or missing in old CSVs)
            try:
                gen_scores_typcorr.append(float(row.get('gen_score_typcorr', 'nan')))
            except (ValueError, TypeError):
                gen_scores_typcorr.append(float('nan'))
            try:
                gen_scores_lenorm.append(float(row.get('gen_score_lenorm', 'nan')))
            except (ValueError, TypeError):
                gen_scores_lenorm.append(float('nan'))
            try:
                gen_scores_typcorr_lenorm.append(float(row.get('gen_score_typcorr_lenorm', 'nan')))
            except (ValueError, TypeError):
                gen_scores_typcorr_lenorm.append(float('nan'))
            
            gt = row['gpt4_ground_truth'].strip().lower()
            labels.append(1 if gt == 'yes' else 0)
            
            strategies.append(row.get('strategy', 'unknown'))
            noun1_list.append(row.get('noun1', ''))
            noun2_list.append(row.get('noun2', ''))
            num_tokens_list.append(int(row.get('num_tokens', 1)))
    
    return {
        'gen_score': gen_scores,
        'gen_score_typcorr': gen_scores_typcorr,
        'gen_score_lenorm': gen_scores_lenorm,
        'gen_score_typcorr_lenorm': gen_scores_typcorr_lenorm,
        'val_score': val_scores,
        'labels': labels,
        'strategies': strategies,
        'noun1': noun1_list,
        'noun2': noun2_list,
        'num_tokens': num_tokens_list,
    }


def create_compare_corrections(data, metric_type='log-probs', output_file=None, title=None):
    """
    Create 2x2 grid comparing different generator score corrections.
    
    Grid layout:
    [raw gen_score]              [typcorr gen_score]
    [lenorm gen_score]           [typcorr + lenorm gen_score]
    
    Y-axis (validator) is shared across all plots.
    """
    val_scores_np = np.array(data['val_score'])
    labels_np = np.array(data['labels'])
    
    gen_variants = [
        ('gen_score', 'Raw'),
        ('gen_score_typcorr', 'Typicality Corrected'),
        ('gen_score_lenorm', 'Length Normalized'),
        ('gen_score_typcorr_lenorm', 'Typcorr + Lenorm'),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharey=True)
    
    metric_label = 'log-odds' if metric_type == 'log-odds' else 'log-probs'
    threshold = 0 if metric_type == 'log-odds' else np.log(0.5)
    
    pos_mask = labels_np == 1
    neg_mask = labels_np == 0
    
    for idx, (key, label) in enumerate(gen_variants):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        gen_scores_np = np.array(data[key])
        
        # Check if data is available (not all NaN)
        if np.all(np.isnan(gen_scores_np)):
            ax.text(0.5, 0.5, 'No data\n(run eval with\n--typicality-correction)', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12, color='gray')
            ax.set_title(label, fontsize=12)
            ax.set_xlabel('Generator log-probs', fontsize=10)
            if col == 0:
                ax.set_ylabel(f'Validator {metric_label}', fontsize=10)
            continue
        
        # Plot points
        ax.scatter(gen_scores_np[pos_mask], val_scores_np[pos_mask],
                   c='orange', marker='o', alpha=0.5, s=20, edgecolors='none', label='Positive')
        ax.scatter(gen_scores_np[neg_mask], val_scores_np[neg_mask],
                   c='blue', marker='o', alpha=0.5, s=20, edgecolors='none', label='Negative')
        
        # Threshold line
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1, alpha=0.7)
        
        # Compute metrics
        valid_mask = ~np.isnan(gen_scores_np)
        if valid_mask.sum() > 1:
            try:
                corr_all, _ = pearsonr(gen_scores_np[valid_mask], val_scores_np[valid_mask])
            except:
                corr_all = np.nan
            
            # Correlation among positives
            pos_valid = valid_mask & pos_mask
            if pos_valid.sum() > 1:
                try:
                    corr_pos, _ = pearsonr(gen_scores_np[pos_valid], val_scores_np[pos_valid])
                except:
                    corr_pos = np.nan
            else:
                corr_pos = np.nan
            
            # Correlation among negatives
            neg_valid = valid_mask & neg_mask
            if neg_valid.sum() > 1:
                try:
                    corr_neg, _ = pearsonr(gen_scores_np[neg_valid], val_scores_np[neg_valid])
                except:
                    corr_neg = np.nan
            else:
                corr_neg = np.nan
            
            try:
                gen_roc = roc_auc_score(labels_np[valid_mask], gen_scores_np[valid_mask])
            except:
                gen_roc = np.nan
            try:
                val_roc = roc_auc_score(labels_np[valid_mask], val_scores_np[valid_mask])
            except:
                val_roc = np.nan
            preds = (val_scores_np[valid_mask] > threshold).astype(int)
            acc = accuracy_score(labels_np[valid_mask], preds)
            
            metrics_text = f'corr={corr_all*100:.1f}\ncorr-pos={corr_pos*100:.1f}\ncorr-neg={corr_neg*100:.1f}\nAcc={acc*100:.1f}\nVal ROC={val_roc*100:.1f}\nGen ROC={gen_roc*100:.1f}'
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(label, fontsize=12)
        ax.set_xlabel('Generator log-probs', fontsize=10)
        if col == 0:
            ax.set_ylabel(f'Validator {metric_label}', fontsize=10)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc='lower right', fontsize=8)
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    # Modify output filename
    output_path = Path(output_file)
    compare_output = output_path.parent / f"{output_path.stem}_compare{output_path.suffix}"
    
    plt.savefig(compare_output, dpi=150, bbox_inches='tight')
    print(f"Compare corrections visualization saved to: {compare_output}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize scores from CSV file")
    parser.add_argument("csv_file", type=str, help="Path to scores CSV file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output filename (default: auto-generate)")
    parser.add_argument("--title", "-t", type=str, default=None, help="Plot title")
    parser.add_argument("--metric-type", type=str, default=None, choices=['log-odds', 'log-probs'],
                        help="Override metric type detection (default: auto-detect from filename)")
    parser.add_argument("--faceted", "-f", action="store_true", help="Also create faceted plot by strategy")
    parser.add_argument("--pca", action="store_true", help="Create PCA plot with standardized scores and outliers")
    parser.add_argument("--n-outliers", type=int, default=10, help="Number of outliers to highlight in PCA plot (default: 10)")
    parser.add_argument("--histograms", action="store_true", help="Show marginal histograms on main plot (off by default)")
    parser.add_argument("--compare-corrections", action="store_true", help="Create 2x2 grid comparing raw/typcorr/lenorm/both gen scores")
    
    args = parser.parse_args()
    
    # Load data
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        return 1
    
    print(f"Loading scores from: {csv_path}")
    
    # Use full loader if PCA or compare-corrections is requested
    if args.pca or args.compare_corrections:
        data = load_scores_csv_full(csv_path)
        gen_scores = data['gen_score']
        val_scores = data['val_score']
        labels = data['labels']
        strategies = data['strategies']
        noun1_list = data['noun1']
        noun2_list = data['noun2']
    else:
        gen_scores, val_scores, labels, strategies, _ = load_scores_csv(csv_path)
        noun1_list, noun2_list = None, None
        data = None
    
    # Detect metric type from filename
    if args.metric_type:
        metric_type = args.metric_type
    elif 'log-odds' in str(csv_path):
        metric_type = 'log-odds'
    else:
        metric_type = 'log-probs'
    
    print(f"  Loaded {len(gen_scores)} samples")
    print(f"  Positive: {sum(labels)}, Negative: {len(labels) - sum(labels)}")
    print(f"  Metric type: {metric_type}")
    print(f"  Strategies: {sorted(set(strategies))}")
    
    # Generate output filename if not provided
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = csv_path.stem.replace('scores_', 'viz_')
        output_file = csv_path.parent / f"{stem}_{timestamp}.png"
    
    # Generate title if not provided
    title = args.title
    if not title:
        title = csv_path.stem.replace('scores_', '').replace('_', ' ')
    
    # Create main visualization (with optional histograms)
    create_visualization_with_marginals(gen_scores, val_scores, labels, strategies, 
                                         metric_type, output_file, title,
                                         show_histograms=args.histograms)
    
    # Create faceted plot if requested
    if args.faceted:
        create_faceted_by_strategy(gen_scores, val_scores, labels, strategies,
                                    metric_type, output_file, title)
    
    # Create compare corrections plot if requested
    if args.compare_corrections:
        create_compare_corrections(data, metric_type, output_file, title)
    
    # Create PCA plot if requested
    if args.pca:
        create_pca_visualization(gen_scores, val_scores, labels, strategies,
                                  noun1_list, noun2_list, output_file, title, 
                                  n_outliers=args.n_outliers)
    
    return 0


if __name__ == "__main__":
    exit(main())
