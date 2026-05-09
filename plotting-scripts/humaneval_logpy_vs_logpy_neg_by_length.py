"""
Side-by-side comparison of unconditional log P(y) vs negated-prompt log P(y|neg_x)
binned by completion length.
Output: output-metrics/humaneval_logpy_vs_logpy_neg_by_length.png
"""
import glob, pandas as pd, numpy as np, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

self_files = sorted(glob.glob('outputs/scores_self-*humaneval*20260501*.csv'))
neg_files = sorted(glob.glob('outputs/scores_neg-*humaneval*20260501*.csv'))

def get_problem(f):
    m = re.search(r'humaneval-humaneval_(\d+)', f)
    return m.group(0) if m else None

self_by_prob = {get_problem(f): f for f in self_files}
neg_by_prob = {get_problem(f): f for f in neg_files}
common = sorted(set(self_by_prob) & set(neg_by_prob))

rows = []
for prob in common:
    sf = pd.read_csv(self_by_prob[prob])
    nf = pd.read_csv(neg_by_prob[prob])
    sf['log_py'] = sf['gen_score'] - sf['gen_score_typcorr']
    sf['log_py_neg'] = nf['gen_score'] - nf['gen_score_typcorr']
    rows.append(sf)

df = pd.concat(rows, ignore_index=True)
df['label'] = (df['correct'] == 'yes').astype(int)
correct = df[df.label == 1]
incorrect = df[df.label == 0]

bins = np.arange(0, 180, 20)
bin_centers = (bins[:-1] + bins[1:]) / 2
min_count = 5

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, score_col, title, ylabel in [
    (axes[0], 'log_py', 'Unconditional: log P(y)', 'mean log P(y)'),
    (axes[1], 'log_py_neg', 'Negated-prompt: log P(y | neg_x)', 'mean log P(y | neg_x)')
]:
    c_means, _, _ = binned_statistic(correct.num_tokens, correct[score_col], statistic='mean', bins=bins)
    ic_means, _, _ = binned_statistic(incorrect.num_tokens, incorrect[score_col], statistic='mean', bins=bins)
    c_counts, _, _ = binned_statistic(correct.num_tokens, correct[score_col], statistic='count', bins=bins)
    ic_counts, _, _ = binned_statistic(incorrect.num_tokens, incorrect[score_col], statistic='count', bins=bins)

    c_mask = c_counts >= min_count
    ic_mask = ic_counts >= min_count
    both = c_mask & ic_mask

    ax.plot(bin_centers[c_mask], c_means[c_mask], 'g-o', markersize=7, linewidth=2.5, label='correct')
    ax.plot(bin_centers[ic_mask], ic_means[ic_mask], 'r-o', markersize=7, linewidth=2.5, label='incorrect')
    ax.fill_between(bin_centers[both], c_means[both], ic_means[both], alpha=0.15, color='blue')

    for i, bc in enumerate(bin_centers):
        if c_mask[i] and ic_mask[i]:
            y_pos = min(c_means[i], ic_means[i]) - 6
            ax.annotate(f'{int(c_counts[i])}/{int(ic_counts[i])}', (bc, y_pos), fontsize=8, ha='center', color='gray')

    ax.set_xlabel('num_tokens (bin center)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

plt.suptitle('HumanEval: log P(y) vs log P(y|neg_x) by completion length', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('output-metrics/humaneval_logpy_vs_logpy_neg_by_length.png', dpi=150, bbox_inches='tight')
print("Saved output-metrics/humaneval_logpy_vs_logpy_neg_by_length.png")
