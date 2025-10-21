"""
Word Frequency vs Generator/Validator Scores Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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


def frequency_vs_scores_plot(df, output_dir):
    """Create scatter plots for word frequency vs generator/validator scores"""
    print("\n" + "="*60)
    print("CREATING WORD FREQUENCY VS SCORES PLOTS")
    print("="*60)
    
    # Check if wordfreq columns exist
    if 'log_wordfreq_noun2' not in df.columns or 'log_wordfreq_noun1' not in df.columns:
        print("ERROR: log_wordfreq_noun2 or log_wordfreq_noun1 columns not found!")
        print("Please run add_wordfreq.py first to add word frequency columns.")
        return
    
    # Create 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    frequency_vars = [
        ('log_wordfreq_noun2', 'WordFreq(noun2)'),
        ('log_wordfreq_noun1', 'WordFreq(noun1)')
    ]
    
    for i, (freq_var, freq_label) in enumerate(frequency_vars):
        # vs gen_score (top row)
        ax = axes[0, i]
        colors = df['ground_truth'].map({0: 'red', 1: 'blue'})
        ax.scatter(df[freq_var], df['gen_score'], c=colors, alpha=0.5, s=20)
        ax.set_xlabel(freq_label, fontsize=11)
        ax.set_ylabel('Generator Score', fontsize=11)
        corr = df[freq_var].corr(df['gen_score'])
        ax.set_title(f'{freq_label} vs Gen (r={corr:.3f})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # vs disc_score (bottom row)
        ax = axes[1, i]
        ax.scatter(df[freq_var], df['disc_score'], c=colors, alpha=0.5, s=20)
        ax.set_xlabel(freq_label, fontsize=11)
        ax.set_ylabel('Discriminator Score', fontsize=11)
        corr = df[freq_var].corr(df['disc_score'])
        ax.set_title(f'{freq_label} vs Disc (r={corr:.3f})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'frequency_vs_scores.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # Print correlation statistics
    print("\nCorrelation statistics:")
    print(f"  WordFreq(noun2) vs gen_score:  {df['log_wordfreq_noun2'].corr(df['gen_score']):.4f}")
    print(f"  WordFreq(noun2) vs disc_score: {df['log_wordfreq_noun2'].corr(df['disc_score']):.4f}")
    print(f"  WordFreq(noun1) vs gen_score:  {df['log_wordfreq_noun1'].corr(df['gen_score']):.4f}")
    print(f"  WordFreq(noun1) vs disc_score: {df['log_wordfreq_noun1'].corr(df['disc_score']):.4f}")
    
    # Split by ground truth
    print("\nSplit by ground truth:")
    for gt in [0, 1]:
        label = "Negative" if gt == 0 else "Positive"
        df_subset = df[df['ground_truth'] == gt]
        print(f"\n  {label} Examples (n={len(df_subset)}):")
        print(f"    WordFreq(noun2) vs gen_score:  {df_subset['log_wordfreq_noun2'].corr(df_subset['gen_score']):.4f}")
        print(f"    WordFreq(noun2) vs disc_score: {df_subset['log_wordfreq_noun2'].corr(df_subset['disc_score']):.4f}")
        print(f"    WordFreq(noun1) vs gen_score:  {df_subset['log_wordfreq_noun1'].corr(df_subset['gen_score']):.4f}")
        print(f"    WordFreq(noun1) vs disc_score: {df_subset['log_wordfreq_noun1'].corr(df_subset['disc_score']):.4f}")


def main():
    parser = argparse.ArgumentParser(description='Word frequency vs scores analysis')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to merged CSV file with wordfreq columns')
    parser.add_argument('--output_dir', type=str, default='eda_outputs',
                        help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    df = load_data(args.data)
    
    # Create plots
    frequency_vs_scores_plot(df, output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nPlot saved to: {output_dir}/frequency_vs_scores.png")


if __name__ == "__main__":
    main()

