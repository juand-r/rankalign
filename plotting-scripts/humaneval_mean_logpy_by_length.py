"""
Binned mean unconditional log P(y) by completion length.
Shows incorrect code is less probable at every length.
Output: output-metrics/humaneval_mean_logpy_by_length.png
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

correct = all_df[all_df.label == 1]
incorrect = all_df[all_df.label == 0]

bins = np.arange(0, 180, 20)
bin_centers = (bins[:-1] + bins[1:]) / 2
min_count = 5

c_means, _, _ = binned_statistic(correct.num_tokens, correct.self_typ, statistic='mean', bins=bins)
ic_means, _, _ = binned_statistic(incorrect.num_tokens, incorrect.self_typ, statistic='mean', bins=bins)
c_counts, _, _ = binned_statistic(correct.num_tokens, correct.self_typ, statistic='count', bins=bins)
ic_counts, _, _ = binned_statistic(incorrect.num_tokens, incorrect.self_typ, statistic='count', bins=bins)

c_mask = c_counts >= min_count
ic_mask = ic_counts >= min_count
both_mask = c_mask & ic_mask

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(bin_centers[c_mask], c_means[c_mask], 'g-o', markersize=7, linewidth=2.5, label='correct')
ax.plot(bin_centers[ic_mask], ic_means[ic_mask], 'r-o', markersize=7, linewidth=2.5, label='incorrect')
ax.fill_between(bin_centers[both_mask], c_means[both_mask], ic_means[both_mask], alpha=0.15, color='blue')

for i, bc in enumerate(bin_centers):
    if c_mask[i] or ic_mask[i]:
        y_pos = min(
            c_means[i] if c_mask[i] else 0,
            ic_means[i] if ic_mask[i] else 0
        ) - 6
        nc = int(c_counts[i])
        ni = int(ic_counts[i])
        ax.annotate(f'{nc}/{ni}', (bc, y_pos), fontsize=9, ha='center', color='gray')

ax.set_xlabel('num_tokens (bin center)', fontsize=13)
ax.set_ylabel('mean log P(y)  (unconditional log-prob)', fontsize=13)
ax.set_title('Mean unconditional log P(y) by completion length\n(incorrect code is less probable at every length)', fontsize=14)
ax.legend(fontsize=12)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('output-metrics/humaneval_mean_logpy_by_length.png', dpi=150)
print("Saved output-metrics/humaneval_mean_logpy_by_length.png")
