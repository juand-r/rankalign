"""
Side-by-side / overlay comparison of mean typicality term by length for the
two TC modes:
  self-TC: log P(y)
  neg-TC : log P(y | x_neg)

By default loads scores from any eval-model. Pass --eval-model <substring>
(e.g., gemma-4-31B-it) to filter to a specific model — required when several
models' score files coexist in outputs/.

Output:
  output-metrics/humaneval_v1_self_vs_neg_logptyp_by_length[__<eval_model>].png
"""
import argparse, glob, pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

parser = argparse.ArgumentParser()
parser.add_argument('--eval-model', default=None,
                    help='Substring matched against eval-model in score filenames '
                         '(e.g. "gemma-4-31B-it"). Default: any model.')
args = parser.parse_args()
EVAL_MODEL = args.eval_model
MODEL_GLOB = f'*{EVAL_MODEL}*' if EVAL_MODEL else '*'
MODEL_TAG = f'__{EVAL_MODEL}' if EVAL_MODEL else ''
MODEL_TITLE = EVAL_MODEL if EVAL_MODEL else 'all eval models'

bins = np.arange(0, 220, 20)
bin_centers = (bins[:-1] + bins[1:]) / 2
MIN_PER_BIN = 5

modes = ['self', 'neg']
data = {}
for mode in modes:
    files = sorted(glob.glob(f'outputs/scores_{mode}-{MODEL_GLOB}humaneval-v1*.csv'))
    if not files:
        print(f'  WARN: no files matched scores_{mode}-{MODEL_GLOB}humaneval-v1*.csv')
        continue
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['label'] = (df['correct'] == 'yes').astype(int)
    df['typ'] = df['gen_score'] - df['gen_score_typcorr']
    c, ic = df[df.label == 1], df[df.label == 0]

    cm, _, _ = binned_statistic(c.num_tokens, c.typ, statistic='mean', bins=bins)
    cc, _, _ = binned_statistic(c.num_tokens, c.typ, statistic='count', bins=bins)
    icm, _, _ = binned_statistic(ic.num_tokens, ic.typ, statistic='mean', bins=bins)
    icc, _, _ = binned_statistic(ic.num_tokens, ic.typ, statistic='count', bins=bins)
    data[mode] = (cm, icm, cc, icc, df['typ'].mean(), len(files))

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# Panel A: same axes overlay — shows the absolute "lift" of neg vs self
ax = axes[0]
for mode, ls, color_c, color_ic in [
    ('self', '-', 'tab:green', 'tab:red'),
    ('neg', '--', 'tab:olive', 'tab:orange'),
]:
    if mode not in data:
        continue
    cm, icm, cc, icc, _, _ = data[mode]
    cm_mask, icm_mask = cc >= MIN_PER_BIN, icc >= MIN_PER_BIN
    ax.plot(bin_centers[cm_mask], cm[cm_mask], color=color_c, linestyle=ls, marker='o',
            markersize=6, linewidth=2, label=f'{mode}-TC correct')
    ax.plot(bin_centers[icm_mask], icm[icm_mask], color=color_ic, linestyle=ls, marker='o',
            markersize=6, linewidth=2, label=f'{mode}-TC incorrect')
ax.set_xlabel('num_tokens (bin center)', fontsize=12)
ax.set_ylabel('mean typicality term  (gen_score - gen_score_typcorr)', fontsize=12)
ax.set_title('Both modes overlaid — same y-axis\n'
             '(neg-TC is the "log P(y|x_neg)" series, lifted higher than self-TC log P(y))',
             fontsize=11)
ax.legend(fontsize=9, loc='lower left')
ax.grid(alpha=0.3)

# Panel B: gap (incorrect minus correct) per bin, both modes
ax = axes[1]
for mode, color in [('self', 'tab:blue'), ('neg', 'tab:orange')]:
    if mode not in data:
        continue
    cm, icm, cc, icc, _, _ = data[mode]
    mask = (cc >= MIN_PER_BIN) & (icc >= MIN_PER_BIN)
    gap = icm - cm
    ax.plot(bin_centers[mask], gap[mask], '-o', color=color, markersize=7, linewidth=2.2,
            label=f'{mode}-TC')
ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('num_tokens (bin center)', fontsize=12)
ax.set_ylabel('mean(incorrect) − mean(correct)   (more negative = bigger separation)', fontsize=11)
ax.set_title('Gap per bin: how much lower is incorrect typicality than correct?', fontsize=11)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

n_files_self = data.get('self', (None,)*6)[5] or 0
n_files_neg = data.get('neg', (None,)*6)[5] or 0
fig.suptitle(f'humaneval-v1 typicality term by completion length: self-TC vs neg-TC  '
             f'(eval model: {MODEL_TITLE}; {n_files_self} self-tc / {n_files_neg} neg-tc files)',
             fontsize=12, y=1.02)
plt.tight_layout()
out = f'output-metrics/humaneval_v1_self_vs_neg_logptyp_by_length{MODEL_TAG}.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')
print()
for mode in modes:
    if mode not in data:
        continue
    cm, icm, cc, icc, mean_typ, _ = data[mode]
    print(f'{mode}-TC: overall mean typ_term = {mean_typ:.2f}')
