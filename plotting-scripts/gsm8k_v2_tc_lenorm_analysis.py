"""
4-panel comparison of score variants for gsm8k-v1: per-token typicality
term, per-token raw, TC+lenorm, and lenorm-only histograms (gsm8k-v2 version
of humaneval_v1_tc_lenorm_analysis.py).

Mode selects the eval CSVs and the typicality-term interpretation:
  --mode self -> log P(y)          per-token "self-typicality"
  --mode neg  -> log P(y | x_neg)  per-token typicality under negated prompt

Output: output-metrics/gsm8k_v2_{mode}_tc_lenorm_analysis.png
"""
import argparse, glob, pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

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

GLOB = f'{args.scores_dir}/scores_{args.mode}-{MODEL_GLOB}_gsm8k-v2-*.csv'
TYP_LABEL = {'self': 'log P(y) / num_tokens   (self-typ per token)',
             'neg':  'log P(y | x_neg) / num_tokens   (neg-typ per token)'}[args.mode]
TYP_TITLE = {'self': 'Per-token self-typicality',
             'neg':  'Per-token neg-typicality'}[args.mode]

files = sorted(glob.glob(GLOB))
print(f"Loading {len(files)} files...")
dfs = [pd.read_csv(f) for f in files]
all_df = pd.concat(dfs, ignore_index=True)
all_df['label'] = (all_df['correct'] == 'yes').astype(int)
all_df['typ_term'] = all_df['gen_score'] - all_df['gen_score_typcorr']
all_df['typ_per_tok'] = all_df['typ_term'] / all_df['num_tokens']
all_df['raw_per_tok'] = all_df['gen_score'] / all_df['num_tokens']

correct = all_df[all_df.label == 1]
incorrect = all_df[all_df.label == 0]
print(f"correct: {len(correct)}, incorrect: {len(incorrect)}")

tc_lenorm_roc = roc_auc_score(all_df.label, all_df.gen_score_typcorr_lenorm)
lenorm_roc = roc_auc_score(all_df.label, all_df.gen_score_lenorm)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

stp_lo = np.percentile(all_df.typ_per_tok.dropna(), 1)
stp_hi = np.percentile(all_df.typ_per_tok.dropna(), 99)
bins = np.linspace(stp_lo, stp_hi, 60)
axes[0,0].hist(correct.typ_per_tok, bins=bins, alpha=0.6, color='green', label='correct')
axes[0,0].hist(incorrect.typ_per_tok, bins=bins, alpha=0.6, color='red', label='incorrect')
axes[0,0].axvline(correct.typ_per_tok.mean(), color='darkgreen', ls='--', label=f'correct mean={correct.typ_per_tok.mean():.3f}')
axes[0,0].axvline(incorrect.typ_per_tok.mean(), color='darkred', ls='--', label=f'incorrect mean={incorrect.typ_per_tok.mean():.3f}')
axes[0,0].set_xlabel(TYP_LABEL)
axes[0,0].set_ylabel('count')
axes[0,0].set_title(f'{TYP_TITLE}: correct vs incorrect')
axes[0,0].legend(fontsize=8)

rp_lo = np.percentile(all_df.raw_per_tok.dropna(), 1)
rp_hi = np.percentile(all_df.raw_per_tok.dropna(), 99)
bins2 = np.linspace(rp_lo, rp_hi, 60)
axes[0,1].hist(correct.raw_per_tok, bins=bins2, alpha=0.6, color='green', label='correct')
axes[0,1].hist(incorrect.raw_per_tok, bins=bins2, alpha=0.6, color='red', label='incorrect')
axes[0,1].axvline(correct.raw_per_tok.mean(), color='darkgreen', ls='--', label=f'correct mean={correct.raw_per_tok.mean():.3f}')
axes[0,1].axvline(incorrect.raw_per_tok.mean(), color='darkred', ls='--', label=f'incorrect mean={incorrect.raw_per_tok.mean():.3f}')
axes[0,1].set_xlabel('log P(y|x) / num_tokens  (raw per token)')
axes[0,1].set_ylabel('count')
axes[0,1].set_title('Per-token raw gen score: correct vs incorrect')
axes[0,1].legend(fontsize=8)

tcl_lo = np.percentile(all_df.gen_score_typcorr_lenorm.dropna(), 1)
tcl_hi = np.percentile(all_df.gen_score_typcorr_lenorm.dropna(), 99)
bins3 = np.linspace(tcl_lo, tcl_hi, 60)
axes[1,0].hist(correct.gen_score_typcorr_lenorm, bins=bins3, alpha=0.6, color='green', label='correct')
axes[1,0].hist(incorrect.gen_score_typcorr_lenorm, bins=bins3, alpha=0.6, color='red', label='incorrect')
axes[1,0].axvline(correct.gen_score_typcorr_lenorm.mean(), color='darkgreen', ls='--')
axes[1,0].axvline(incorrect.gen_score_typcorr_lenorm.mean(), color='darkred', ls='--')
axes[1,0].set_xlabel('tc+lenorm score')
axes[1,0].set_ylabel('count')
axes[1,0].set_title(f'TC+Lenorm: correct vs incorrect (ROC={tc_lenorm_roc:.3f})')
axes[1,0].legend(fontsize=8)

ln_lo = np.percentile(all_df.gen_score_lenorm.dropna(), 1)
ln_hi = np.percentile(all_df.gen_score_lenorm.dropna(), 99)
bins4 = np.linspace(ln_lo, ln_hi, 60)
axes[1,1].hist(correct.gen_score_lenorm, bins=bins4, alpha=0.6, color='green', label='correct')
axes[1,1].hist(incorrect.gen_score_lenorm, bins=bins4, alpha=0.6, color='red', label='incorrect')
axes[1,1].axvline(correct.gen_score_lenorm.mean(), color='darkgreen', ls='--')
axes[1,1].axvline(incorrect.gen_score_lenorm.mean(), color='darkred', ls='--')
axes[1,1].set_xlabel('lenorm score (log P(y|x) / len)')
axes[1,1].set_ylabel('count')
axes[1,1].set_title(f'Lenorm only: correct vs incorrect (ROC={lenorm_roc:.3f})')
axes[1,1].legend(fontsize=8)

plt.suptitle(f'gsm8k-v1 ({args.mode}-TC, eval model: {MODEL_TITLE}, '
             f'n={len(all_df)} candidates over {len(files)} problems)', fontsize=12, y=1.00)
plt.tight_layout()
out = f'output-metrics/gsm8k_v2_{args.mode}_tc_lenorm_analysis{MODEL_TAG}.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved {out}")
