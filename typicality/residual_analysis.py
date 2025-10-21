"""
Residual Analysis: Does typicality explain G-V gap variance?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

sns.set_style("whitegrid")


def main():
    parser = argparse.ArgumentParser(description='Residual analysis')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='eda_outputs')
    args = parser.parse_args()
    
    df = pd.read_csv(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("RESIDUAL ANALYSIS")
    print("="*60)
    
    # Model 1: Gen ~ Disc only (baseline)
    X1 = np.column_stack([np.ones(len(df)), df['disc_score'].values])
    y = df['gen_score'].values
    beta1 = np.linalg.lstsq(X1, y, rcond=None)[0]
    y_pred1 = X1 @ beta1
    residuals1 = y - y_pred1
    r2_1 = 1 - np.sum(residuals1**2) / np.sum((y - y.mean())**2)
    
    print(f"\nModel 1: Gen ~ Disc")
    print(f"  R² = {r2_1:.4f}")
    print(f"  Residual std = {residuals1.std():.4f}")
    
    # Model 2: Gen ~ Disc + P(noun2|context)
    X2 = np.column_stack([np.ones(len(df)), df['disc_score'].values, 
                          df['log_prob_noun2_given_context'].values])
    beta2 = np.linalg.lstsq(X2, y, rcond=None)[0]
    y_pred2 = X2 @ beta2
    residuals2 = y - y_pred2
    r2_2 = 1 - np.sum(residuals2**2) / np.sum((y - y.mean())**2)
    
    print(f"\nModel 2: Gen ~ Disc + P(noun2|context)")
    print(f"  R² = {r2_2:.4f}")
    print(f"  Residual std = {residuals2.std():.4f}")
    print(f"  ΔR² = {r2_2 - r2_1:.4f}")
    
    # Variance explained by typicality
    var_explained = (residuals1.std()**2 - residuals2.std()**2) / residuals1.std()**2
    print(f"\nTypicality explains {var_explained*100:.1f}% of residual variance")
    
    # Plot: Residuals vs Typicality
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Top left: Residuals from Model 1 vs P(noun2|context)
    ax = axes[0, 0]
    colors = df['ground_truth'].map({0: 'red', 1: 'blue'})
    ax.scatter(df['log_prob_noun2_given_context'], residuals1, c=colors, alpha=0.5, s=20)
    corr = np.corrcoef(df['log_prob_noun2_given_context'], residuals1)[0,1]
    ax.set_xlabel('P(noun2|context)', fontsize=11)
    ax.set_ylabel('Residuals from Gen~Disc', fontsize=11)
    ax.set_title(f'Residuals vs Typicality (r={corr:.3f})', fontsize=12, fontweight='bold')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    # Top right: Residuals from Model 2 vs P(noun2|context) (should be ~0 correlation)
    ax = axes[0, 1]
    ax.scatter(df['log_prob_noun2_given_context'], residuals2, c=colors, alpha=0.5, s=20)
    corr = np.corrcoef(df['log_prob_noun2_given_context'], residuals2)[0,1]
    ax.set_xlabel('P(noun2|context)', fontsize=11)
    ax.set_ylabel('Residuals from Gen~Disc+Typ', fontsize=11)
    ax.set_title(f'After Adding Typicality (r={corr:.3f})', fontsize=12, fontweight='bold')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    # Bottom left: Histogram of residuals
    ax = axes[1, 0]
    ax.hist(residuals1, bins=50, alpha=0.5, color='red', label='Model 1: Gen~Disc')
    ax.hist(residuals2, bins=50, alpha=0.5, color='blue', label='Model 2: Gen~Disc+Typ')
    ax.set_xlabel('Residuals', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Residual Distributions', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Bottom right: Q-Q plot for Model 2
    ax = axes[1, 1]
    from scipy import stats as sp_stats
    sp_stats.probplot(residuals2, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Model 2)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'residual_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'residual_analysis.png'}")
    plt.close()
    
    # Additional analysis: By ground truth
    print("\n" + "="*60)
    print("BY GROUND TRUTH")
    print("="*60)
    
    for gt in [0, 1]:
        label = "Negative" if gt == 0 else "Positive"
        mask = df['ground_truth'] == gt
        
        # Model 1
        X1_subset = X1[mask]
        y_subset = y[mask]
        beta1_subset = np.linalg.lstsq(X1_subset, y_subset, rcond=None)[0]
        y_pred1_subset = X1_subset @ beta1_subset
        r2_1_subset = 1 - np.sum((y_subset - y_pred1_subset)**2) / np.sum((y_subset - y_subset.mean())**2)
        
        # Model 2
        X2_subset = X2[mask]
        beta2_subset = np.linalg.lstsq(X2_subset, y_subset, rcond=None)[0]
        y_pred2_subset = X2_subset @ beta2_subset
        r2_2_subset = 1 - np.sum((y_subset - y_pred2_subset)**2) / np.sum((y_subset - y_subset.mean())**2)
        
        print(f"\n{label} Examples:")
        print(f"  Model 1 R² = {r2_1_subset:.4f}")
        print(f"  Model 2 R² = {r2_2_subset:.4f}")
        print(f"  ΔR² = {r2_2_subset - r2_1_subset:.4f}")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()

