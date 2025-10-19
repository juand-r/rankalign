#!/usr/bin/env python3
"""
Analyze debug values to verify log-odds vs log-probs relationship.
Compare ROC curves, find equivalent thresholds, and visualize relationships.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from scipy.optimize import minimize_scalar
import sys

def load_data(task='collie'):
    """Load both logodds and logprobs CSV files."""
    logodds_file = f"../outputs/debug_values_{task}_logodds.csv"
    logprobs_file = f"../outputs/debug_values_{task}_logprobs.csv"
    
    df_logodds = pd.read_csv(logodds_file)
    df_logprobs = pd.read_csv(logprobs_file)
    
    print(f"Loaded {len(df_logodds)} examples from logodds file")
    print(f"Loaded {len(df_logprobs)} examples from logprobs file")
    
    return df_logodds, df_logprobs


def compute_roc_curves(df_logodds, df_logprobs):
    """Compute ROC curves for both metrics."""
    # For log-odds (higher = more positive)
    fpr_odds, tpr_odds, thresholds_odds = roc_curve(
        df_logodds['ground_truth'], 
        df_logodds['disc_score']
    )
    auc_odds = auc(fpr_odds, tpr_odds)
    
    # For log-probs (higher = more positive) 
    fpr_probs, tpr_probs, thresholds_probs = roc_curve(
        df_logprobs['ground_truth'],
        df_logprobs['disc_score']
    )
    auc_probs = auc(fpr_probs, tpr_probs)
    
    return (fpr_odds, tpr_odds, thresholds_odds, auc_odds), \
           (fpr_probs, tpr_probs, thresholds_probs, auc_probs)


def plot_roc_curves(roc_odds, roc_probs, save_path='../outputs/roc_comparison.png'):
    """Plot both ROC curves on the same plot."""
    fpr_odds, tpr_odds, _, auc_odds = roc_odds
    fpr_probs, tpr_probs, _, auc_probs = roc_probs
    
    plt.figure(figsize=(10, 8))
    
    # Plot log-odds ROC in blue
    plt.plot(fpr_odds, tpr_odds, 'b-', linewidth=2, 
             label=f'Log-odds (AUC = {auc_odds:.4f})')
    
    # Plot log-probs ROC in red
    plt.plot(fpr_probs, tpr_probs, 'r-', linewidth=2,
             label=f'Log-probs (AUC = {auc_probs:.4f})')
    
    # Diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison: Log-odds vs Log-probs', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to: {save_path}")
    plt.close()


def find_equivalent_thresholds(df_logodds, df_logprobs, target_threshold_odds=0.0):
    """
    Find the log-probs threshold that gives the same confusion matrix
    as a given log-odds threshold.
    """
    # Get confusion matrix for log-odds at target threshold
    y_true = df_logodds['ground_truth'].values
    pred_odds = (df_logodds['disc_score'].values > target_threshold_odds).astype(int)
    cm_odds = confusion_matrix(y_true, pred_odds)
    
    print(f"\nConfusion matrix for log-odds (threshold = {target_threshold_odds}):")
    print(cm_odds)
    
    # Search for equivalent threshold in log-probs
    def diff_confusion_matrix(threshold):
        """Compute difference between confusion matrices."""
        pred_probs = (df_logprobs['disc_score'].values > threshold).astype(int)
        cm_probs = confusion_matrix(y_true, pred_probs)
        return np.sum(np.abs(cm_odds - cm_probs))
    
    # Search over a range of thresholds
    result = minimize_scalar(diff_confusion_matrix, bounds=(-5, 0), method='bounded')
    equiv_threshold_probs = result.x
    
    # Get the resulting confusion matrix
    pred_probs = (df_logprobs['disc_score'].values > equiv_threshold_probs).astype(int)
    cm_probs = confusion_matrix(y_true, pred_probs)
    
    print(f"\nEquivalent log-probs threshold: {equiv_threshold_probs:.6f}")
    print(f"Confusion matrix for log-probs (threshold = {equiv_threshold_probs:.6f}):")
    print(cm_probs)
    print(f"Difference: {diff_confusion_matrix(equiv_threshold_probs)}")
    
    return equiv_threshold_probs


def plot_empirical_vs_theoretical(df_logodds, df_logprobs, 
                                   save_path='../outputs/logodds_vs_logprobs_scatter.png'):
    """
    Plot empirical relationship between log-odds and log-probs,
    along with theoretical relationship.
    """
    # Get discriminator scores
    logodds = df_logodds['disc_score'].values
    logprobs = df_logprobs['disc_score'].values
    ground_truth = df_logodds['ground_truth'].values
    
    # Compute theoretical relationship
    # If logodds = log(p/(1-p)), then p = exp(logodds)/(1+exp(logodds))
    # and logprobs = log(p)
    logodds_range = np.linspace(logodds.min(), logodds.max(), 1000)
    p_theoretical = np.exp(logodds_range) / (1 + np.exp(logodds_range))
    logprobs_theoretical = np.log(p_theoretical)
    
    plt.figure(figsize=(12, 8))
    
    # Scatter plot colored by ground truth
    colors = ['blue' if gt == 0 else 'orange' for gt in ground_truth]
    plt.scatter(logodds, logprobs, c=colors, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Theoretical relationship
    plt.plot(logodds_range, logprobs_theoretical, 'r-', linewidth=3, 
             label='Theoretical: log(p) = log(exp(logodds)/(1+exp(logodds)))', zorder=5)
    
    plt.xlabel('Discriminator Log-odds', fontsize=12)
    plt.ylabel('Discriminator Log-probs', fontsize=12)
    plt.title('Empirical vs Theoretical Relationship\n(Blue=Negative, Orange=Positive)', fontsize=14)
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Add reference lines for thresholds
    plt.axvline(x=0.0, color='blue', linestyle='--', alpha=0.5, linewidth=1, label='Log-odds threshold (0.0)')
    plt.axhline(y=np.log(0.5), color='red', linestyle='--', alpha=0.5, linewidth=1, label='Log-probs threshold (log(0.5))')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Scatter plot saved to: {save_path}")
    plt.close()


def print_summary_statistics(df_logodds, df_logprobs):
    """Print summary statistics for both metrics."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print("\nLog-odds:")
    print(df_logodds['disc_score'].describe())
    
    print("\nLog-probs:")
    print(df_logprobs['disc_score'].describe())
    
    print("\nGround truth distribution:")
    print(df_logodds['ground_truth'].value_counts())


def main():
    if len(sys.argv) > 1:
        task = sys.argv[1]
    else:
        task = 'collie'
    
    print(f"Analyzing task: {task}")
    print("="*60)
    
    # Load data
    df_logodds, df_logprobs = load_data(task)
    
    # Verify same ground truth
    assert (df_logodds['ground_truth'].values == df_logprobs['ground_truth'].values).all(), \
        "Ground truth labels don't match!"
    print("✓ Ground truth labels match\n")
    
    # Print summary statistics
    print_summary_statistics(df_logodds, df_logprobs)
    
    # Compute ROC curves
    print("\nComputing ROC curves...")
    roc_odds, roc_probs = compute_roc_curves(df_logodds, df_logprobs)
    
    # Plot ROC curves
    plot_roc_curves(roc_odds, roc_probs)
    
    # Find equivalent thresholds
    print("\n" + "="*60)
    print("FINDING EQUIVALENT THRESHOLDS")
    print("="*60)
    equiv_thresh = find_equivalent_thresholds(df_logodds, df_logprobs, target_threshold_odds=0.0)
    
    print(f"\nTheoretical log(0.5) = {np.log(0.5):.6f}")
    print(f"Empirical equivalent threshold = {equiv_thresh:.6f}")
    print(f"Difference = {abs(equiv_thresh - np.log(0.5)):.6f}")
    
    # Plot empirical vs theoretical relationship
    print("\n" + "="*60)
    print("PLOTTING EMPIRICAL VS THEORETICAL RELATIONSHIP")
    print("="*60)
    plot_empirical_vs_theoretical(df_logodds, df_logprobs)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

