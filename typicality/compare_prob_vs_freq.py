"""
Compare P(noun2) from GPT-2 vs WordFreq(noun2) as predictors of Generator Score
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

sns.set_style("whitegrid")


def plot_comparison(test_path, output_dir):
    """Create side-by-side comparison of P(noun2) vs WordFreq(noun2)"""
    
    # Load test data
    df_test = pd.read_csv(test_path)
    print(f"Loaded test set: {len(df_test)} examples")
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Colors by ground truth
    colors = df_test['ground_truth'].map({0: 'red', 1: 'blue'})
    y = df_test['gen_score'].values
    
    # Plot 1: P(noun2) vs Actual Gen Score
    ax = axes[0]
    x1 = df_test['log_prob_noun2'].values
    ax.scatter(x1, y, c=colors, alpha=0.5, s=20)
    
    # Compute correlation
    corr1 = np.corrcoef(x1, y)[0, 1]
    
    ax.set_xlabel('P(noun2) - GPT-2 Log Probability', fontsize=12)
    ax.set_ylabel('Actual Generator Score', fontsize=12)
    ax.set_title(f'GPT-2 P(noun2) vs Gen Score\nPearson r={corr1:.3f}', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.5, label='Positive (Hypernym)'),
        Patch(facecolor='red', alpha=0.5, label='Negative (Non-hypernym)')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=10)
    
    # Plot 2: WordFreq(noun2) vs Actual Gen Score
    ax = axes[1]
    x2 = df_test['log_wordfreq_noun2'].values
    ax.scatter(x2, y, c=colors, alpha=0.5, s=20)
    
    # Compute correlation
    corr2 = np.corrcoef(x2, y)[0, 1]
    
    ax.set_xlabel('WordFreq(noun2) - Corpus Frequency (Zipf)', fontsize=12)
    ax.set_ylabel('Actual Generator Score', fontsize=12)
    ax.set_title(f'WordFreq(noun2) vs Gen Score\nPearson r={corr2:.3f}', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(handles=legend_elements, loc='best', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / 'prob_vs_freq_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    plt.close()
    
    # Print statistics
    print(f"\nCorrelation with Generator Score:")
    print(f"  P(noun2) (GPT-2):       {corr1:.4f}")
    print(f"  WordFreq(noun2):        {corr2:.4f}")
    print(f"  Difference:             {abs(corr1 - corr2):.4f}")
    
    # Correlation between P(noun2) and WordFreq(noun2)
    corr_measures = np.corrcoef(x1, x2)[0, 1]
    print(f"\nCorrelation between P(noun2) and WordFreq(noun2): {corr_measures:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Compare P(noun2) vs WordFreq(noun2)')
    parser.add_argument('--test', type=str, required=True,
                        help='Path to test CSV file')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save plot')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create plot
    plot_comparison(args.test, output_dir)
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()

