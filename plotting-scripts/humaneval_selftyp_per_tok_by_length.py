"""
Per-token P(y) vs length, showing incorrect code is 'weirder' per token at every length.
3-panel: scatter, binned means, box plots within length bands.
Output: output-metrics/humaneval_selftyp_per_tok_by_length.png
"""
import glob, pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

files = sorted(glob.glob('outputs/scores_self-*humaneval*20260501*.csv'))
dfs = [pd.read_csv(f) for f in files]
all_df = pd.concat(dfs, ignore_index=True)
all_df['label'] = (all_df['correct'] == 'yes').astype(int)
all_df['self_typ'] = all_df['gen_score'] - all_df['gen_score_typcorr']
all_df['selftyp_per_tok'] = all_df['self_typ'] / all_df['num_tokens']

correct = all_df[all_df.label == 1]
incorrect = all_df[all_df.label == 0]

mask = all_df.num_tokens <= 300
sub = all_df[mask]
c = sub[sub.label == 1]
ic = sub[sub.label == 0]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].scatter(ic.num_tokens, ic.selftyp_per_tok, alpha=0.25, s=12, c='red', label='incorrect', zorder=2)
axes[0].scatter(c.num_tokens, c.selftyp_per_tok, alpha=0.25, s=12, c='green', label='correct', zorder=3)
axes[0].set_xlabel('num_tokens (completion length)', fontsize=12)
axes[0].set_ylabel('log P(y) / num_tokens\n(per-token unconditional log-prob)', fontsize=12)
axes[0].set_title('Per-token P(y) vs length', fontsize=13)
axes[0].legend(fontsize=11)

bins = np.arange(0, 260, 20)
bin_centers = (bins[:-1] + bins[1:]) / 2
min_count = 5

c_means, _, _ = binned_statistic(c.num_tokens, c.selftyp_per_tok, statistic='mean', bins=bins)
ic_means, _, _ = binned_statistic(ic.num_tokens, ic.selftyp_per_tok, statistic='mean', bins=bins)
c_counts, _, _ = binned_statistic(c.num_tokens, c.selftyp_per_tok, statistic='count', bins=bins)
ic_counts, _, _ = binned_statistic(ic.num_tokens, ic.selftyp_per_tok, statistic='count', bins=bins)

c_mask = c_counts >= min_count
ic_mask = ic_counts >= min_count
both_mask = c_mask & ic_mask

axes[1].plot(bin_centers[c_mask], c_means[c_mask], 'g-o', markersize=6, linewidth=2, label='correct')
axes[1].plot(bin_centers[ic_mask], ic_means[ic_mask], 'r-o', markersize=6, linewidth=2, label='incorrect')
axes[1].fill_between(bin_centers[both_mask], c_means[both_mask], ic_means[both_mask], alpha=0.15, color='blue')
for i, bc in enumerate(bin_centers):
    if c_mask[i] and ic_mask[i]:
        axes[1].annotate(f'{int(c_counts[i])}/{int(ic_counts[i])}',
                         (bc, min(c_means[i], ic_means[i]) - 0.15),
                         fontsize=7, ha='center', color='gray')
axes[1].set_xlabel('num_tokens (bin center)', fontsize=12)
axes[1].set_ylabel('mean log P(y) / num_tokens', fontsize=12)
axes[1].set_title('Binned mean per-token P(y) by length\n(gap = incorrect code is "weirder" per token)', fontsize=13)
axes[1].legend(fontsize=11)

length_bands = [(20, 60), (60, 100), (100, 160)]
data_c = [c[(c.num_tokens >= lo) & (c.num_tokens < hi)].selftyp_per_tok.values for lo, hi in length_bands]
data_ic = [ic[(ic.num_tokens >= lo) & (ic.num_tokens < hi)].selftyp_per_tok.values for lo, hi in length_bands]
labels_band = [f'{lo}-{hi} tok' for lo, hi in length_bands]

x_pos = np.arange(len(length_bands))
bp_c = axes[2].boxplot(data_c, positions=x_pos - 0.17, widths=0.3, patch_artist=True,
                        boxprops=dict(facecolor='lightgreen', alpha=0.7),
                        medianprops=dict(color='darkgreen', linewidth=2),
                        flierprops=dict(markersize=3))
bp_ic = axes[2].boxplot(data_ic, positions=x_pos + 0.17, widths=0.3, patch_artist=True,
                         boxprops=dict(facecolor='lightsalmon', alpha=0.7),
                         medianprops=dict(color='darkred', linewidth=2),
                         flierprops=dict(markersize=3))
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(labels_band, fontsize=11)
axes[2].set_ylabel('log P(y) / num_tokens', fontsize=12)
axes[2].set_title('Per-token P(y) within length bands\n(controlling for length)', fontsize=13)
axes[2].legend([bp_c["boxes"][0], bp_ic["boxes"][0]], ['correct', 'incorrect'], fontsize=11)

plt.tight_layout()
plt.savefig('output-metrics/humaneval_selftyp_per_tok_by_length.png', dpi=150)
print("Saved output-metrics/humaneval_selftyp_per_tok_by_length.png")
