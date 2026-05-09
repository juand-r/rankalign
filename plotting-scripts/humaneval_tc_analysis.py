"""
4-panel overview of typicality correction on HumanEval base model scores.
Panels: length distribution, self-typ distribution, length vs self-typ scatter,
raw vs TC gen scores scatter.
Output: output-metrics/humaneval_tc_analysis.png
"""
import glob, pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

files = sorted(glob.glob('outputs/scores_self-*humaneval*20260501*.csv'))
dfs = [pd.read_csv(f) for f in files]
all_df = pd.concat(dfs, ignore_index=True)
all_df['label'] = (all_df['correct'] == 'yes').astype(int)
all_df['self_typ'] = all_df['gen_score'] - all_df['gen_score_typcorr']

correct = all_df[all_df.label == 1]
incorrect = all_df[all_df.label == 0]

from scipy.stats import pearsonr
r_all, _ = pearsonr(all_df.num_tokens, all_df.self_typ)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

bins_len = np.arange(0, 600, 20)
axes[0,0].hist(correct.num_tokens, bins=bins_len, alpha=0.6, label=f'correct (n={len(correct)})', color='green')
axes[0,0].hist(incorrect.num_tokens, bins=bins_len, alpha=0.6, label=f'incorrect (n={len(incorrect)})', color='red')
axes[0,0].set_xlabel('num_tokens (completion length)')
axes[0,0].set_ylabel('count')
axes[0,0].set_title('Completion length distribution')
axes[0,0].legend()

bins_typ = np.linspace(-200, 800, 60)
axes[0,1].hist(correct.self_typ, bins=bins_typ, alpha=0.6, label='correct', color='green')
axes[0,1].hist(incorrect.self_typ, bins=bins_typ, alpha=0.6, label='incorrect', color='red')
axes[0,1].set_xlabel('-log P(y)  (self-typicality term)')
axes[0,1].set_ylabel('count')
axes[0,1].set_title('Self-typicality correction term')
axes[0,1].legend()

axes[1,0].scatter(correct.num_tokens, correct.self_typ, alpha=0.3, s=10, c='green', label='correct')
axes[1,0].scatter(incorrect.num_tokens, incorrect.self_typ, alpha=0.3, s=10, c='red', label='incorrect')
axes[1,0].set_xlabel('num_tokens')
axes[1,0].set_ylabel('-log P(y)')
axes[1,0].set_title(f'Length vs self-typicality (r={r_all:.3f})')
axes[1,0].legend()

axes[1,1].scatter(correct.gen_score, correct.gen_score_typcorr, alpha=0.3, s=10, c='green', label='correct')
axes[1,1].scatter(incorrect.gen_score, incorrect.gen_score_typcorr, alpha=0.3, s=10, c='red', label='incorrect')
axes[1,1].set_xlabel('raw gen_score (log P(y|x))')
axes[1,1].set_ylabel('TC gen_score (log P(y|x) - log P(y))')
axes[1,1].set_title('Raw vs TC gen scores')
axes[1,1].legend()

plt.tight_layout()
plt.savefig('output-metrics/humaneval_tc_analysis.png', dpi=150)
print("Saved output-metrics/humaneval_tc_analysis.png")
