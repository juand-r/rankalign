"""
Proper Train/Test Analysis for Typicality Models

Fits models on TRAIN data and evaluates on TEST data.

Two models tested:
1. Baseline: gen_score = disc_score + typicality_score (no parameters to fit)
2. Regression: gen_score = β₀ + β₁*disc_score + β₂*typicality_score (fit β on train)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from scipy import stats

sns.set_style("whitegrid")


def load_and_verify_data(train_path, test_path):
    """Load train and test data"""
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    print(f"\nTrain set: {len(df_train)} examples")
    print(f"  Positive: {(df_train['ground_truth']==1).sum()} ({(df_train['ground_truth']==1).mean()*100:.1f}%)")
    print(f"  Negative: {(df_train['ground_truth']==0).sum()} ({(df_train['ground_truth']==0).mean()*100:.1f}%)")
    
    print(f"\nTest set: {len(df_test)} examples")
    print(f"  Positive: {(df_test['ground_truth']==1).sum()} ({(df_test['ground_truth']==1).mean()*100:.1f}%)")
    print(f"  Negative: {(df_test['ground_truth']==0).sum()} ({(df_test['ground_truth']==0).mean()*100:.1f}%)")
    
    # Verify no overlap in noun pairs
    train_pairs = set(zip(df_train['noun1'], df_train['noun2']))
    test_pairs = set(zip(df_test['noun1'], df_test['noun2']))
    overlap = train_pairs & test_pairs
    
    if len(overlap) > 0:
        print(f"\n⚠️  WARNING: {len(overlap)} overlapping pairs between train and test!")
        print("First few overlaps:", list(overlap)[:5])
    else:
        print(f"\n✓ No overlap between train and test sets")
    
    return df_train, df_test


def fit_and_evaluate_models(df_train, df_test, output_dir, use_freq=False):
    """Fit regression on train, evaluate both models on test"""
    
    print("\n" + "="*70)
    print("TRAINING AND EVALUATION")
    print("="*70)
    
    if use_freq:
        # Use word frequency instead of model probabilities
        typicality_vars = [
            ('log_wordfreq_noun2', 'WordFreq(noun2)'),
            ('log_wordfreq_noun1', 'WordFreq(noun1)'),
            ('log_prob_noun2_given_context', 'P(noun2|context)')
        ]
    else:
        # Use model probabilities (original)
        typicality_vars = [
            ('log_prob_noun2', 'P(noun2)'),
            ('log_prob_noun1', 'P(noun1)'),
            ('log_prob_noun2_given_context', 'P(noun2|context)')
        ]
    
    results = []
    
    for typ_var, typ_label in typicality_vars:
        print(f"\n{'-'*70}")
        print(f"Typicality measure: {typ_label}")
        print(f"{'-'*70}")
        
        # ===========================
        # BASELINE MODEL (no fitting)
        # ===========================
        # Just apply: gen_score = disc_score + typicality_score on TEST
        y_test = df_test['gen_score'].values
        y_pred_baseline_test = df_test['disc_score'].values + df_test[typ_var].values
        
        # Compute test R²
        ss_res_baseline = np.sum((y_test - y_pred_baseline_test) ** 2)
        ss_tot_test = np.sum((y_test - y_test.mean()) ** 2)
        r2_baseline_test = 1 - (ss_res_baseline / ss_tot_test)
        
        rmse_baseline = np.sqrt(np.mean((y_test - y_pred_baseline_test) ** 2))
        corr_baseline = np.corrcoef(y_test, y_pred_baseline_test)[0, 1]
        
        print(f"\nBaseline Model (gen = disc + typ):")
        print(f"  Test R²:  {r2_baseline_test:.4f}")
        print(f"  Test RMSE: {rmse_baseline:.4f}")
        print(f"  Test corr: {corr_baseline:.4f}")
        
        # ===========================
        # REGRESSION MODEL
        # ===========================
        # Fit on TRAIN
        X_train = np.column_stack([
            np.ones(len(df_train)),
            df_train['disc_score'].values,
            df_train[typ_var].values
        ])
        y_train = df_train['gen_score'].values
        
        # OLS solution
        beta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
        
        # Evaluate on TRAIN (for comparison)
        y_pred_train = X_train @ beta
        ss_res_train = np.sum((y_train - y_pred_train) ** 2)
        ss_tot_train = np.sum((y_train - y_train.mean()) ** 2)
        r2_train = 1 - (ss_res_train / ss_tot_train)
        
        # Apply to TEST
        X_test = np.column_stack([
            np.ones(len(df_test)),
            df_test['disc_score'].values,
            df_test[typ_var].values
        ])
        y_pred_regression_test = X_test @ beta
        
        # Compute test R²
        ss_res_regression = np.sum((y_test - y_pred_regression_test) ** 2)
        r2_regression_test = 1 - (ss_res_regression / ss_tot_test)
        
        rmse_regression = np.sqrt(np.mean((y_test - y_pred_regression_test) ** 2))
        corr_regression = np.corrcoef(y_test, y_pred_regression_test)[0, 1]
        
        print(f"\nRegression Model (gen = β₀ + β₁*disc + β₂*typ):")
        print(f"  Fitted on TRAIN:")
        print(f"    β₀ (intercept): {beta[0]:.4f}")
        print(f"    β₁ (disc):      {beta[1]:.4f}")
        print(f"    β₂ (typ):       {beta[2]:.4f}")
        print(f"    Train R²:       {r2_train:.4f}")
        print(f"  Evaluated on TEST:")
        print(f"    Test R²:        {r2_regression_test:.4f}")
        print(f"    Test RMSE:      {rmse_regression:.4f}")
        print(f"    Test corr:      {corr_regression:.4f}")
        
        # Compare models
        print(f"\nModel Comparison:")
        print(f"  Baseline R²:   {r2_baseline_test:.4f}")
        print(f"  Regression R²: {r2_regression_test:.4f}")
        print(f"  Improvement:   {r2_regression_test - r2_baseline_test:.4f}")
        
        results.append({
            'var': typ_var,
            'label': typ_label,
            # Baseline
            'r2_baseline_test': r2_baseline_test,
            'rmse_baseline_test': rmse_baseline,
            'y_pred_baseline_test': y_pred_baseline_test,
            # Regression
            'beta': beta,
            'r2_train': r2_train,
            'r2_regression_test': r2_regression_test,
            'rmse_regression_test': rmse_regression,
            'y_pred_regression_test': y_pred_regression_test,
        })
    
    return results


def plot_results(df_train, df_test, results, output_dir, use_freq=False):
    """Create 2x3 plot grid"""
    
    print("\n" + "="*70)
    print("CREATING PLOTS")
    print("="*70)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    colors = df_test['ground_truth'].map({0: 'red', 1: 'blue'})
    y_test = df_test['gen_score'].values
    
    # Plot each typicality measure
    for i, res in enumerate(results):
        # Top row: REGRESSION MODEL
        ax = axes[0, i]
        y_pred = res['y_pred_regression_test']
        
        ax.scatter(y_pred, y_test, c=colors, alpha=0.5, s=20)
        
        # y=x line
        min_val = min(y_pred.min(), y_test.min())
        max_val = max(y_pred.max(), y_test.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2, label='y=x')
        
        ax.set_xlabel('Predicted Gen Score', fontsize=11)
        ax.set_ylabel('Actual Gen Score', fontsize=11)
        ax.set_title(f'Regression: {res["label"]}\nTest R²={res["r2_regression_test"]:.3f}',
                     fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add coefficient info
        beta = res['beta']
        ax.text(0.02, 0.98, f'β₀={beta[0]:.2f}\nβ₁={beta[1]:.2f}\nβ₂={beta[2]:.2f}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Bottom row: BASELINE MODEL
        ax = axes[1, i]
        y_pred = res['y_pred_baseline_test']
        
        ax.scatter(y_pred, y_test, c=colors, alpha=0.5, s=20)
        
        # y=x line
        min_val = min(y_pred.min(), y_test.min())
        max_val = max(y_pred.max(), y_test.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2, label='y=x')
        
        ax.set_xlabel('Predicted Gen Score', fontsize=11)
        ax.set_ylabel('Actual Gen Score', fontsize=11)
        ax.set_title(f'Baseline: {res["label"]}\nTest R²={res["r2_baseline_test"]:.3f}',
                     fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add model info
        ax.text(0.02, 0.98, 'gen = disc + typ\n(no parameters)',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Use different filename based on whether we're using frequency or probabilities
    if use_freq:
        filename = 'train_test_comparison-freq.png'
    else:
        filename = 'train_test_comparison.png'
    
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir / filename}")
    plt.close()


def create_summary_table(results, output_dir, use_freq=False):
    """Create and save summary table"""
    
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    
    # Create DataFrame
    summary_data = []
    for res in results:
        summary_data.append({
            'Typicality': res['label'],
            'Baseline Test R²': res['r2_baseline_test'],
            'Regression Train R²': res['r2_train'],
            'Regression Test R²': res['r2_regression_test'],
            'Improvement': res['r2_regression_test'] - res['r2_baseline_test'],
            'β₀': res['beta'][0],
            'β₁ (disc)': res['beta'][1],
            'β₂ (typ)': res['beta'][2],
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Print
    print("\n", df_summary.to_string(index=False))
    
    # Save to CSV with different filename based on whether we're using frequency
    if use_freq:
        filename = 'model_comparison_summary-freq.csv'
    else:
        filename = 'model_comparison_summary.csv'
    
    summary_path = output_dir / filename
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")
    
    return df_summary


def main():
    parser = argparse.ArgumentParser(description='Train/Test Analysis')
    parser.add_argument('--train', type=str, required=True,
                        help='Path to train CSV file')
    parser.add_argument('--test', type=str, required=True,
                        help='Path to test CSV file')
    parser.add_argument('--output_dir', type=str, default='train_test_outputs',
                        help='Directory to save outputs')
    parser.add_argument('--freq', action='store_true',
                        help='Use word frequency instead of model probabilities for noun1/noun2')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    df_train, df_test = load_and_verify_data(args.train, args.test)
    
    # Fit and evaluate
    results = fit_and_evaluate_models(df_train, df_test, output_dir, use_freq=args.freq)
    
    # Plot
    plot_results(df_train, df_test, results, output_dir, use_freq=args.freq)
    
    # Summary table
    create_summary_table(results, output_dir, use_freq=args.freq)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nAll outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()

