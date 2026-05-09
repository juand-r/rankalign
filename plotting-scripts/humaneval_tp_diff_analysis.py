"""
Analyze (positive, negative) pairs flipped by typicality correction (T-P analysis).
Panels: length difference, self-typ difference, raw margin vs TC boost, per-problem counts.
Output: output-metrics/humaneval_tp_diff_analysis.png
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

raw_wrong_pairs = []
tc_wrong_pairs = []
total_pairs = 0

for prob, grp in all_df.groupby('problem_name'):
    pos = grp[grp.label == 1].reset_index(drop=True)
    neg = grp[grp.label == 0].reset_index(drop=True)
    if len(pos) == 0 or len(neg) == 0:
        continue
    for _, p in pos.iterrows():
        for _, n in neg.iterrows():
            total_pairs += 1
            pair_info = {
                'problem': prob,
                'pos_tokens': p['num_tokens'], 'neg_tokens': n['num_tokens'],
                'pos_raw': p['gen_score'], 'neg_raw': n['gen_score'],
                'pos_tc': p['gen_score_typcorr'], 'neg_tc': n['gen_score_typcorr'],
                'pos_selftyp': p['self_typ'], 'neg_selftyp': n['self_typ'],
                'pos_preview': str(p['solution_preview'])[:80],
                'neg_preview': str(n['solution_preview'])[:80],
            }
            if p['gen_score'] < n['gen_score']:
                raw_wrong_pairs.append(pair_info)
            if p['gen_score_typcorr'] < n['gen_score_typcorr']:
                tc_wrong_pairs.append(pair_info)

raw_wrong_set = {(p['problem'], p['pos_preview'], p['neg_preview']) for p in raw_wrong_pairs}
tc_wrong_set = {(p['problem'], p['pos_preview'], p['neg_preview']) for p in tc_wrong_pairs}

tc_wrong_dict = {(p['problem'], p['pos_preview'], p['neg_preview']): p for p in tc_wrong_pairs}
tp_diff_keys = tc_wrong_set - raw_wrong_set
tp_diff = [tc_wrong_dict[k] for k in tp_diff_keys]
tp_df = pd.DataFrame(tp_diff)

tp_df['raw_margin'] = tp_df['pos_raw'] - tp_df['neg_raw']
tp_df['tc_margin'] = tp_df['pos_tc'] - tp_df['neg_tc']
tp_df['selftyp_diff'] = tp_df['neg_selftyp'] - tp_df['pos_selftyp']
tp_df['len_diff'] = tp_df['neg_tokens'] - tp_df['pos_tokens']

print(f"Total pairs: {total_pairs}")
print(f"Raw wrong (P): {len(raw_wrong_pairs)}, TC wrong (T): {len(tc_wrong_pairs)}")
print(f"T-P (newly broken by TC): {len(tp_diff)}")
print(f"P-T (fixed by TC): {len(raw_wrong_set - tc_wrong_set)}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].hist(tp_df.len_diff, bins=50, alpha=0.7, color='purple')
axes[0,0].axvline(0, color='black', ls='-', lw=0.5)
axes[0,0].axvline(tp_df.len_diff.mean(), color='red', ls='--', label=f'mean={tp_df.len_diff.mean():.1f}')
axes[0,0].set_xlabel('neg_tokens - pos_tokens')
axes[0,0].set_title(f'T-P pairs: length difference (n={len(tp_df)})')
axes[0,0].legend()

axes[0,1].hist(tp_df.selftyp_diff, bins=50, alpha=0.7, color='orange')
axes[0,1].axvline(0, color='black', ls='-', lw=0.5)
axes[0,1].axvline(tp_df.selftyp_diff.mean(), color='red', ls='--', label=f'mean={tp_df.selftyp_diff.mean():.2f}')
axes[0,1].set_xlabel('neg_selftyp - pos_selftyp (neg TC boost - pos TC boost)')
axes[0,1].set_title('T-P pairs: self-typ difference')
axes[0,1].legend()

axes[1,0].scatter(tp_df.raw_margin, tp_df.selftyp_diff, alpha=0.15, s=8, c='purple')
axes[1,0].axhline(0, color='black', ls='-', lw=0.5)
axes[1,0].axvline(0, color='black', ls='-', lw=0.5)
axes[1,0].set_xlabel('raw margin (pos_raw - neg_raw)')
axes[1,0].set_ylabel('selftyp_diff (how much more TC boost neg got)')
axes[1,0].set_title('Raw margin vs TC boost difference (T-P pairs)')

prob_counts = tp_df.problem.value_counts()
axes[1,1].barh(range(len(prob_counts)), prob_counts.values, color='purple', alpha=0.7)
axes[1,1].set_yticks(range(len(prob_counts)))
axes[1,1].set_yticklabels(prob_counts.index, fontsize=6)
axes[1,1].set_xlabel('# T-P pairs')
axes[1,1].set_title('T-P pairs per problem')

plt.tight_layout()
plt.savefig('output-metrics/humaneval_tp_diff_analysis.png', dpi=150)
print("Saved output-metrics/humaneval_tp_diff_analysis.png")
