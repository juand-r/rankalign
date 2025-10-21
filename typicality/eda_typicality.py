"""
Exploratory Data Analysis for Typicality and Generator-Validator Gap
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


def load_data(csv_path):
    """Load merged data"""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} examples from {csv_path}")
    print(f"Columns: {list(df.columns)}")
    return df


def basic_stats(df):
    """Print basic statistics"""
    print("\n" + "="*60)
    print("BASIC STATISTICS")
    print("="*60)
    
    print(f"\nTotal examples: {len(df)}")
    print(f"Positive examples: {(df['ground_truth'] == 1).sum()} ({(df['ground_truth'] == 1).mean()*100:.1f}%)")
    print(f"Negative examples: {(df['ground_truth'] == 0).sum()} ({(df['ground_truth'] == 0).mean()*100:.1f}%)")
    
    print("\n" + "-"*60)
    print("Score Statistics:")
    print("-"*60)
    
    for col in ['gen_score', 'disc_score', 'log_prob_noun2', 'log_prob_noun1', 'log_prob_noun2_given_context']:
        print(f"\n{col}:")
        print(f"  Mean: {df[col].mean():.4f}")
        print(f"  Std:  {df[col].std():.4f}")
        print(f"  Min:  {df[col].min():.4f}")
        print(f"  Max:  {df[col].max():.4f}")


def correlation_analysis(df, output_dir):
    """Compute and visualize correlations"""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    # Select numerical columns
    score_cols = ['gen_score', 'disc_score', 'log_prob_noun2', 'log_prob_noun1', 'log_prob_noun2_given_context']
    
    # Pearson correlations
    print("\nPearson Correlations:")
    corr_matrix = df[score_cols].corr()
    print(corr_matrix.round(3))
    
    # Spearman correlations
    print("\nSpearman Correlations:")
    spearman_matrix = df[score_cols].corr(method='spearman')
    print(spearman_matrix.round(3))
    
    # Plot correlation matrix
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Pearson
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
                vmin=-1, vmax=1, square=True, ax=axes[0], cbar_kws={'shrink': 0.8})
    axes[0].set_title('Pearson Correlation Matrix', fontsize=14, fontweight='bold')
    
    # Spearman
    sns.heatmap(spearman_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                vmin=-1, vmax=1, square=True, ax=axes[1], cbar_kws={'shrink': 0.8})
    axes[1].set_title('Spearman Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrices.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'correlation_matrices.png'}")
    plt.close()


def gv_gap_analysis(df, output_dir):
    """Analyze the Generator-Validator gap"""
    print("\n" + "="*60)
    print("GENERATOR-VALIDATOR GAP ANALYSIS")
    print("="*60)
    
    # Compute G-V gap
    df['gv_gap'] = df['gen_score'] - df['disc_score']
    
    print(f"\nG-V Gap statistics:")
    print(f"  Mean: {df['gv_gap'].mean():.4f}")
    print(f"  Std:  {df['gv_gap'].std():.4f}")
    print(f"  Median: {df['gv_gap'].median():.4f}")
    
    # By ground truth
    print(f"\nG-V Gap by ground truth:")
    for gt in [0, 1]:
        gap_mean = df[df['ground_truth'] == gt]['gv_gap'].mean()
        gap_std = df[df['ground_truth'] == gt]['gv_gap'].std()
        label = "Negative" if gt == 0 else "Positive"
        print(f"  {label}: {gap_mean:.4f} ± {gap_std:.4f}")
    
    # Correlation with typicality
    print(f"\nG-V Gap correlation with typicality scores:")
    print(f"  P(noun2):           {df['gv_gap'].corr(df['log_prob_noun2']):.4f}")
    print(f"  P(noun1):           {df['gv_gap'].corr(df['log_prob_noun1']):.4f}")
    print(f"  P(noun2|context):   {df['gv_gap'].corr(df['log_prob_noun2_given_context']):.4f}")


def scatter_plots(df, output_dir, use_freq=False):
    """Create scatter plots for key relationships"""
    print("\n" + "="*60)
    print("CREATING SCATTER PLOTS")
    print("="*60)
    
    # 1. Gen vs Disc (colored by ground truth)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = df['ground_truth'].map({0: 'red', 1: 'blue'})
    ax.scatter(df['disc_score'], df['gen_score'], c=colors, alpha=0.5, s=30)
    ax.plot([df['disc_score'].min(), df['disc_score'].max()], 
            [df['disc_score'].min(), df['disc_score'].max()], 
            'k--', alpha=0.3, label='y=x')
    ax.set_xlabel('Discriminator Score (Validator)', fontsize=12)
    ax.set_ylabel('Generator Score', fontsize=12)
    ax.set_title('Generator vs Validator Scores', fontsize=14, fontweight='bold')
    ax.legend(['y=x', 'Negative (0)', 'Positive (1)'], loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'gen_vs_disc.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'gen_vs_disc.png'}")
    plt.close()
    
    # 2. Typicality vs Gen/Disc scores (6 subplots)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
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
    
    for i, (typ_var, typ_label) in enumerate(typicality_vars):
        # vs gen_score
        ax = axes[0, i]
        colors = df['ground_truth'].map({0: 'red', 1: 'blue'})
        ax.scatter(df[typ_var], df['gen_score'], c=colors, alpha=0.5, s=20)
        ax.set_xlabel(typ_label, fontsize=11)
        ax.set_ylabel('Generator Score', fontsize=11)
        corr = df[typ_var].corr(df['gen_score'])
        ax.set_title(f'{typ_label} vs Gen (r={corr:.3f})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # vs disc_score
        ax = axes[1, i]
        ax.scatter(df[typ_var], df['disc_score'], c=colors, alpha=0.5, s=20)
        ax.set_xlabel(typ_label, fontsize=11)
        ax.set_ylabel('Discriminator Score', fontsize=11)
        corr = df[typ_var].corr(df['disc_score'])
        ax.set_title(f'{typ_label} vs Disc (r={corr:.3f})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Use different filename based on whether we're using frequency or probabilities
    if use_freq:
        filename = 'frequency_vs_scores.png'
    else:
        filename = 'typicality_vs_scores.png'
    
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / filename}")
    plt.close()
    
    # 3. G-V gap vs typicality
    df['gv_gap'] = df['gen_score'] - df['disc_score']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (typ_var, typ_label) in enumerate(typicality_vars):
        ax = axes[i]
        colors = df['ground_truth'].map({0: 'red', 1: 'blue'})
        ax.scatter(df[typ_var], df['gv_gap'], c=colors, alpha=0.5, s=20)
        ax.set_xlabel(typ_label, fontsize=11)
        ax.set_ylabel('G-V Gap (Gen - Disc)', fontsize=11)
        corr = df[typ_var].corr(df['gv_gap'])
        ax.set_title(f'{typ_label} vs G-V Gap (r={corr:.3f})', fontsize=12, fontweight='bold')
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Use different filename based on whether we're using frequency or probabilities
    if use_freq:
        gv_filename = 'gv_gap_vs_frequency.png'
    else:
        gv_filename = 'gv_gap_vs_typicality.png'
    
    plt.savefig(output_dir / gv_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / gv_filename}")
    plt.close()


def split_analysis(df, output_dir):
    """Analysis split by positive/negative examples"""
    print("\n" + "="*60)
    print("SPLIT ANALYSIS (Positive vs Negative)")
    print("="*60)
    
    for gt in [0, 1]:
        label = "Negative" if gt == 0 else "Positive"
        df_subset = df[df['ground_truth'] == gt]
        
        print(f"\n{label} Examples (n={len(df_subset)}):")
        print(f"  Gen-Disc correlation: {df_subset['gen_score'].corr(df_subset['disc_score']):.4f}")
        
        print(f"  Typicality correlations with Gen:")
        for col, name in [('log_prob_noun2', 'P(noun2)'), 
                          ('log_prob_noun1', 'P(noun1)'),
                          ('log_prob_noun2_given_context', 'P(noun2|context)')]:
            corr = df_subset[col].corr(df_subset['gen_score'])
            print(f"    {name:20s}: {corr:.4f}")
        
        print(f"  Typicality correlations with Disc:")
        for col, name in [('log_prob_noun2', 'P(noun2)'), 
                          ('log_prob_noun1', 'P(noun1)'),
                          ('log_prob_noun2_given_context', 'P(noun2|context)')]:
            corr = df_subset[col].corr(df_subset['disc_score'])
            print(f"    {name:20s}: {corr:.4f}")


def theoretical_relationship(df, output_dir):
    """Test the theoretical relationship: gen_prob ∝ disc_score × typicality"""
    print("\n" + "="*60)
    print("TESTING THEORETICAL RELATIONSHIP")
    print("="*60)
    print("Theory: gen_score ≈ disc_score + typicality (in log space)")
    
    typicality_vars = [
        ('log_prob_noun2', 'P(noun2)'),
        ('log_prob_noun1', 'P(noun1)'),
        ('log_prob_noun2_given_context', 'P(noun2|context)')
    ]
    
    results = []
    
    for typ_var, typ_label in typicality_vars:
        # Simple additive model in log space: gen_score ~ disc_score + typicality
        X = df[['disc_score', typ_var]].values
        y = df['gen_score'].values
        
        # Add intercept
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        # OLS solution
        beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        y_pred = X_with_intercept @ beta
        
        # Compute R²
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        print(f"\n{typ_label}:")
        print(f"  gen_score = {beta[0]:.4f} + {beta[1]:.4f}*disc_score + {beta[2]:.4f}*{typ_var}")
        print(f"  R² = {r_squared:.4f}")
        print(f"  Correlation with predicted: {np.corrcoef(y, y_pred)[0,1]:.4f}")
        
        results.append({
            'typicality': typ_label,
            'var': typ_var,
            'intercept': beta[0],
            'disc_coef': beta[1],
            'typ_coef': beta[2],
            'r_squared': r_squared,
            'y_pred': y_pred
        })
    
    # Plot: Predicted vs Actual
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, res in enumerate(results):
        ax = axes[i]
        colors = df['ground_truth'].map({0: 'red', 1: 'blue'})
        ax.scatter(res['y_pred'], df['gen_score'], c=colors, alpha=0.5, s=20)
        ax.plot([df['gen_score'].min(), df['gen_score'].max()],
                [df['gen_score'].min(), df['gen_score'].max()],
                'k--', alpha=0.3)
        ax.set_xlabel('Predicted Gen Score', fontsize=11)
        ax.set_ylabel('Actual Gen Score', fontsize=11)
        ax.set_title(f'{res["typicality"]} (R²={res["r_squared"]:.3f})', 
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'theoretical_model_fit.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'theoretical_model_fit.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='EDA for typicality analysis')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to merged CSV file')
    parser.add_argument('--output_dir', type=str, default='eda_outputs',
                        help='Directory to save plots')
    parser.add_argument('--freq', action='store_true',
                        help='Use word frequency instead of model probabilities for noun1/noun2')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    df = load_data(args.data)
    
    # Run analyses
    basic_stats(df)
    correlation_analysis(df, output_dir)
    gv_gap_analysis(df, output_dir)
    scatter_plots(df, output_dir, use_freq=args.freq)
    split_analysis(df, output_dir)
    theoretical_relationship(df, output_dir)
    
    print("\n" + "="*60)
    print("EDA COMPLETE!")
    print("="*60)
    print(f"\nAll plots saved to: {output_dir}/")


if __name__ == "__main__":
    main()

