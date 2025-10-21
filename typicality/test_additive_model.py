"""
Test the direct additive relationship:
    log(gen_prob) ∝ log(validator_score) + log(typicality_score)
    
Or equivalently:
    gen_score ≈ disc_score + typicality_score
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from scipy import stats

sns.set_style("whitegrid")


def test_additive_relationship(df, output_dir):
    """Test gen_score vs (disc_score + typicality_score)"""
    
    print("="*70)
    print("TESTING DIRECT ADDITIVE RELATIONSHIP")
    print("="*70)
    print("\nTheory: gen_score ≈ disc_score + typicality_score")
    print("(In product form: P_gen ∝ P_validator × P_typicality)\n")
    
    typicality_vars = [
        ('log_prob_noun2', 'P(noun2)'),
        ('log_prob_noun1', 'P(noun1)'),
        ('log_prob_noun2_given_context', 'P(noun2|context)')
    ]
    
    results = []
    
    # Test each typicality measure
    for typ_var, typ_label in typicality_vars:
        # Compute the sum
        df[f'disc_plus_{typ_var}'] = df['disc_score'] + df[typ_var]
        
        # Compute correlation
        pearson_r, pearson_p = stats.pearsonr(df[f'disc_plus_{typ_var}'], df['gen_score'])
        spearman_r, spearman_p = stats.spearmanr(df[f'disc_plus_{typ_var}'], df['gen_score'])
        
        # Compute R² (how much variance explained)
        # If perfect additive: gen_score = disc_score + typicality, then R² = 1
        y_pred = df[f'disc_plus_{typ_var}'].values
        y_true = df['gen_score'].values
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        
        # For direct comparison (y = x), we don't fit intercept/slope
        # So let's also compute with optimal scaling
        # y_pred_scaled = a + b * (disc + typ)
        X = np.column_stack([np.ones(len(df)), df[f'disc_plus_{typ_var}'].values])
        beta = np.linalg.lstsq(X, y_true, rcond=None)[0]
        y_pred_scaled = X @ beta
        ss_res_scaled = np.sum((y_true - y_pred_scaled) ** 2)
        r2_scaled = 1 - (ss_res_scaled / ss_tot)
        
        # Mean absolute error
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        print(f"\n{typ_label}:")
        print(f"  Pearson r:  {pearson_r:.4f} (p={pearson_p:.2e})")
        print(f"  Spearman ρ: {spearman_r:.4f} (p={spearman_p:.2e})")
        print(f"  Direct comparison (gen = disc + typ):")
        print(f"    MAE:  {mae:.4f}")
        print(f"    RMSE: {rmse:.4f}")
        print(f"  With optimal scaling (gen = a + b*(disc + typ)):")
        print(f"    Intercept: {beta[0]:.4f}")
        print(f"    Slope:     {beta[1]:.4f}")
        print(f"    R²:        {r2_scaled:.4f}")
        
        results.append({
            'var': typ_var,
            'label': typ_label,
            'pearson_r': pearson_r,
            'spearman_r': spearman_r,
            'mae': mae,
            'rmse': rmse,
            'r2_scaled': r2_scaled,
            'beta0': beta[0],
            'beta1': beta[1],
            'y_pred_scaled': y_pred_scaled
        })
    
    return results


def plot_additive_relationship(df, results, output_dir):
    """Plot gen_score vs (disc_score + typicality_score)"""
    
    print("\n" + "="*70)
    print("CREATING PLOTS")
    print("="*70)
    
    # Create 3x2 subplot: top row = raw, bottom row = with optimal scaling
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    colors = df['ground_truth'].map({0: 'red', 1: 'blue'})
    
    for i, res in enumerate(results):
        typ_var = res['var']
        typ_label = res['label']
        
        # Top row: Direct relationship (y = x)
        ax = axes[0, i]
        combined = df['disc_score'] + df[typ_var]
        ax.scatter(combined, df['gen_score'], c=colors, alpha=0.5, s=15)
        
        # Plot y=x line
        min_val = min(combined.min(), df['gen_score'].min())
        max_val = max(combined.max(), df['gen_score'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2, label='y=x')
        
        ax.set_xlabel(f'Disc + {typ_label}', fontsize=11)
        ax.set_ylabel('Gen Score', fontsize=11)
        ax.set_title(f'Direct: Gen vs (Disc + {typ_label})\nr={res["pearson_r"]:.3f}, RMSE={res["rmse"]:.2f}',
                     fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Bottom row: With optimal scaling
        ax = axes[1, i]
        ax.scatter(res['y_pred_scaled'], df['gen_score'], c=colors, alpha=0.5, s=15)
        
        # Plot y=x line
        min_val = min(res['y_pred_scaled'].min(), df['gen_score'].min())
        max_val = max(res['y_pred_scaled'].max(), df['gen_score'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2, label='y=x')
        
        ax.set_xlabel(f'Predicted: {res["beta0"]:.2f} + {res["beta1"]:.2f}*(Disc + {typ_label})', fontsize=10)
        ax.set_ylabel('Gen Score (Actual)', fontsize=11)
        ax.set_title(f'Scaled: Gen vs (Disc + {typ_label})\nR²={res["r2_scaled"]:.3f}',
                     fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'additive_model_test.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'additive_model_test.png'}")
    plt.close()
    
    # Also create a comparison plot showing residuals
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, res in enumerate(results):
        typ_var = res['var']
        typ_label = res['label']
        
        ax = axes[i]
        combined = df['disc_score'] + df[typ_var]
        residuals = df['gen_score'] - combined
        
        ax.scatter(combined, residuals, c=colors, alpha=0.5, s=15)
        ax.axhline(0, color='k', linestyle='--', alpha=0.5, linewidth=2)
        ax.set_xlabel(f'Disc + {typ_label}', fontsize=11)
        ax.set_ylabel('Residual (Gen - Predicted)', fontsize=11)
        ax.set_title(f'Residuals: {typ_label}\nRMSE={res["rmse"]:.2f}',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'additive_model_residuals.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'additive_model_residuals.png'}")
    plt.close()


def compare_to_baseline(df, results, output_dir):
    """Compare additive model to baseline (disc only)"""
    
    print("\n" + "="*70)
    print("COMPARISON TO BASELINE (Disc Only)")
    print("="*70)
    
    # Baseline: gen_score vs disc_score only
    baseline_r = df['gen_score'].corr(df['disc_score'])
    
    X_baseline = np.column_stack([np.ones(len(df)), df['disc_score'].values])
    y = df['gen_score'].values
    beta_baseline = np.linalg.lstsq(X_baseline, y, rcond=None)[0]
    y_pred_baseline = X_baseline @ beta_baseline
    ss_res_baseline = np.sum((y - y_pred_baseline) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_baseline = 1 - (ss_res_baseline / ss_tot)
    
    print(f"\nBaseline (Gen ~ Disc only):")
    print(f"  Correlation: {baseline_r:.4f}")
    print(f"  R²:          {r2_baseline:.4f}")
    
    print(f"\nAdditive Models (Gen ~ Disc + Typicality):")
    for res in results:
        improvement = res['r2_scaled'] - r2_baseline
        print(f"\n  {res['label']}:")
        print(f"    R²:          {res['r2_scaled']:.4f}")
        print(f"    ΔR²:         {improvement:.4f} ({improvement/r2_baseline*100:.1f}% relative improvement)")


def main():
    parser = argparse.ArgumentParser(description='Test additive relationship')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to merged CSV file')
    parser.add_argument('--output_dir', type=str, default='eda_outputs',
                        help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    df = pd.read_csv(args.data)
    print(f"\nLoaded {len(df)} examples from {args.data}\n")
    
    # Run analysis
    results = test_additive_relationship(df, output_dir)
    plot_additive_relationship(df, results, output_dir)
    compare_to_baseline(df, results, output_dir)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()

