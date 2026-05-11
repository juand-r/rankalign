"""
Binned mean typicality-term log-prob by completion length, for gsm8k-v1
(gsm8k-v1.1 version of humaneval_v1_mean_logpy_by_length.py).

The typicality term = gen_score - gen_score_typcorr is:
  --mode self -> log P(y)         (unconditional self-typicality)
  --mode neg  -> log P(y | x_neg) (typicality under negated prompt)

By default reads scores from any eval-model in outputs/. Pass --eval-model
<substring> (e.g., gemma-2-9b-it) to filter to a specific model.

Output:
  output-metrics/gsm8k_v1.1_1_{mode}_mean_logptyp_by_length[__<eval_model>].png
"""
import argparse, glob, pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['self', 'neg'], default='self')
parser.add_argument('--eval-model', default=None,
                    help='Substring matched against eval-model in score filenames.')
parser.add_argument('--scores-dir', default='outputs',
                    help='Directory holding score CSVs (default outputs).')
args = parser.parse_args()

EVAL_MODEL = args.eval_model
# Use literal `_gsm8k-v1` after the model substring so EVAL_MODEL="gemma-2-2b"
# doesn't also match gemma-2-2b-it.
MODEL_GLOB = f'*{EVAL_MODEL}' if EVAL_MODEL else '*'
MODEL_TAG = f'__{EVAL_MODEL}' if EVAL_MODEL else ''
MODEL_TITLE = EVAL_MODEL if EVAL_MODEL else 'all eval models'

GLOB = f'{args.scores_dir}/scores_{args.mode}-{MODEL_GLOB}_gsm8k-v1.1-*.csv'
YLABEL = {'self': 'mean log P(y)  (unconditional log-prob)',
          'neg':  'mean log P(y | x_neg)  (under negated prompt)'}[args.mode]
TITLE_TYP = {'self': 'mean unconditional log P(y)',
             'neg':  'mean log P(y | x_neg)'}[args.mode]

files = sorted(glob.glob(GLOB))
print(f"Loading {len(files)} files...")
dfs = [pd.read_csv(f) for f in files]
all_df = pd.concat(dfs, ignore_index=True)
all_df['label'] = (all_df['correct'] == 'yes').astype(int)
all_df['self_typ'] = all_df['gen_score'] - all_df['gen_score_typcorr']

correct = all_df[all_df.label == 1]
incorrect = all_df[all_df.label == 0]
print(f"correct: {len(correct)}, incorrect: {len(incorrect)}")
print(f"num_tokens: min={all_df.num_tokens.min()}, max={all_df.num_tokens.max()}, mean={all_df.num_tokens.mean():.1f}")

ntok_max = int(all_df.num_tokens.quantile(0.98))
bin_width = max(50, ntok_max // 12)
bins = np.arange(0, ntok_max + bin_width, bin_width)
bin_centers = (bins[:-1] + bins[1:]) / 2
min_count = 5

c_means, _, _ = binned_statistic(correct.num_tokens, correct.self_typ, statistic='mean', bins=bins)
ic_means, _, _ = binned_statistic(incorrect.num_tokens, incorrect.self_typ, statistic='mean', bins=bins)
c_counts, _, _ = binned_statistic(correct.num_tokens, correct.self_typ, statistic='count', bins=bins)
ic_counts, _, _ = binned_statistic(incorrect.num_tokens, incorrect.self_typ, statistic='count', bins=bins)

c_mask = c_counts >= min_count
ic_mask = ic_counts >= min_count
both_mask = c_mask & ic_mask

fig, ax = plt.subplots(figsize=(11, 6.5))

ax.plot(bin_centers[c_mask], c_means[c_mask], 'g-o', markersize=7, linewidth=2.5, label='correct')
ax.plot(bin_centers[ic_mask], ic_means[ic_mask], 'r-o', markersize=7, linewidth=2.5, label='incorrect')
ax.fill_between(bin_centers[both_mask], c_means[both_mask], ic_means[both_mask], alpha=0.15, color='blue')

y_offset = (np.nanmax(np.concatenate([c_means, ic_means])) -
            np.nanmin(np.concatenate([c_means, ic_means]))) * 0.08
for i, bc in enumerate(bin_centers):
    if c_mask[i] or ic_mask[i]:
        y_pos = min(
            c_means[i] if c_mask[i] else 0,
            ic_means[i] if ic_mask[i] else 0
        ) - y_offset
        nc = int(c_counts[i])
        ni = int(ic_counts[i])
        ax.annotate(f'{nc}/{ni}', (bc, y_pos), fontsize=9, ha='center', color='gray')

ax.set_xlabel('num_tokens (bin center)', fontsize=13)
ax.set_ylabel(YLABEL, fontsize=13)
ax.set_title(f'gsm8k-v1 ({args.mode}-TC): {TITLE_TYP} by completion length\n'
             f'eval model: {MODEL_TITLE}, {len(files)} test problems, {len(all_df)} candidates'
             f' (counts shown as correct/incorrect per bin)', fontsize=13)
ax.legend(fontsize=12)
ax.grid(alpha=0.3)

plt.tight_layout()
out = f'output-metrics/gsm8k_v1.1_1_{args.mode}_mean_logptyp_by_length{MODEL_TAG}.png'
plt.savefig(out, dpi=150)
print(f"Saved {out}")
