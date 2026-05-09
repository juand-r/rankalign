"""
4-panel comparison of score variants: per-token self-typ, per-token raw,
TC+lenorm, and lenorm-only histograms.
Output: output-metrics/humaneval_tc_lenorm_analysis.png
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
all_df['selftyp_per_tok'] = all_df['self_typ'] / all_df['num_tokens']
all_df['raw_per_tok'] = all_df['gen_score'] / all_df['num_tokens']

correct = all_df[all_df.label == 1]
incorrect = all_df[all_df.label == 0]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

bins = np.linspace(-6, 0, 60)
axes[0,0].hist(correct.selftyp_per_tok, bins=bins, alpha=0.6, color='green', label='correct')
axes[0,0].hist(incorrect.selftyp_per_tok, bins=bins, alpha=0.6, color='red', label='incorrect')
axes[0,0].axvline(correct.selftyp_per_tok.mean(), color='darkgreen', ls='--', label=f'correct mean={correct.selftyp_per_tok.mean():.3f}')
axes[0,0].axvline(incorrect.selftyp_per_tok.mean(), color='darkred', ls='--', label=f'incorrect mean={incorrect.selftyp_per_tok.mean():.3f}')
axes[0,0].set_xlabel('-log P(y) / num_tokens  (self-typ per token)')
axes[0,0].set_ylabel('count')
axes[0,0].set_title('Per-token self-typicality: correct vs incorrect')
axes[0,0].legend(fontsize=8)

bins2 = np.linspace(-6, 0, 60)
axes[0,1].hist(correct.raw_per_tok, bins=bins2, alpha=0.6, color='green', label='correct')
axes[0,1].hist(incorrect.raw_per_tok, bins=bins2, alpha=0.6, color='red', label='incorrect')
axes[0,1].axvline(correct.raw_per_tok.mean(), color='darkgreen', ls='--', label=f'correct mean={correct.raw_per_tok.mean():.3f}')
axes[0,1].axvline(incorrect.raw_per_tok.mean(), color='darkred', ls='--', label=f'incorrect mean={incorrect.raw_per_tok.mean():.3f}')
axes[0,1].set_xlabel('log P(y|x) / num_tokens  (raw per token)')
axes[0,1].set_ylabel('count')
axes[0,1].set_title('Per-token raw gen score: correct vs incorrect')
axes[0,1].legend(fontsize=8)

bins3 = np.linspace(-1, 2.5, 60)
axes[1,0].hist(correct.gen_score_typcorr_lenorm, bins=bins3, alpha=0.6, color='green', label='correct')
axes[1,0].hist(incorrect.gen_score_typcorr_lenorm, bins=bins3, alpha=0.6, color='red', label='incorrect')
axes[1,0].axvline(correct.gen_score_typcorr_lenorm.mean(), color='darkgreen', ls='--')
axes[1,0].axvline(incorrect.gen_score_typcorr_lenorm.mean(), color='darkred', ls='--')
axes[1,0].set_xlabel('tc+lenorm score')
axes[1,0].set_ylabel('count')
axes[1,0].set_title(f'TC+Lenorm: correct vs incorrect (ROC={0.6896:.3f})')
axes[1,0].legend(fontsize=8)

bins4 = np.linspace(-6, 0, 60)
axes[1,1].hist(correct.gen_score_lenorm, bins=bins4, alpha=0.6, color='green', label='correct')
axes[1,1].hist(incorrect.gen_score_lenorm, bins=bins4, alpha=0.6, color='red', label='incorrect')
axes[1,1].axvline(correct.gen_score_lenorm.mean(), color='darkgreen', ls='--')
axes[1,1].axvline(incorrect.gen_score_lenorm.mean(), color='darkred', ls='--')
axes[1,1].set_xlabel('lenorm score (log P(y|x) / len)')
axes[1,1].set_ylabel('count')
axes[1,1].set_title(f'Lenorm only: correct vs incorrect (ROC={0.7248:.3f})')
axes[1,1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('output-metrics/humaneval_tc_lenorm_analysis.png', dpi=150)
print("Saved output-metrics/humaneval_tc_lenorm_analysis.png")
