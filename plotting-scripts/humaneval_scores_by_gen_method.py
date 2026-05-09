"""
Visualize log P variants broken down by solution generation method.
Shows whether neg TC correction works differently for solutions generated
by different methods (gpt-4o standard, gpt-4o-mini intentional_bug, etc.)
"""
import json, csv, glob, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ── Load JSONL provenance ──
lookup_by_sol = {}
with open('data/humaneval/solutions.jsonl') as f:
    for line in f:
        r = json.loads(line)
        sol = r['solution']
        if sol not in lookup_by_sol:
            lookup_by_sol[sol] = r

def categorize(r):
    model = r.get('model', 'unknown')
    strategy = r.get('strategy', 'standard')
    return f'{model} {strategy}'

# ── Load and merge score CSVs ──
self_files = sorted(glob.glob('outputs/scores_self-*humaneval*20260501*.csv'))
neg_files = sorted(glob.glob('outputs/scores_neg-*humaneval*20260501*.csv'))

self_by_prob = {}
for f in self_files:
    df = pd.read_csv(f)
    prob = df['problem_name'].iloc[0]
    self_by_prob[prob] = df

neg_by_prob = {}
for f in neg_files:
    df = pd.read_csv(f)
    prob = df['problem_name'].iloc[0]
    neg_by_prob[prob] = df

rows = []
for prob in sorted(self_by_prob.keys()):
    sf = self_by_prob[prob].copy()
    nf = neg_by_prob[prob]
    sf['log_py'] = sf['gen_score'] - sf['gen_score_typcorr']
    sf['log_py_neg'] = nf['gen_score'].values - nf['gen_score_typcorr'].values
    sf['neg_tc'] = nf['gen_score_typcorr'].values
    sf['self_tc'] = sf['gen_score_typcorr']
    rows.append(sf)

df = pd.concat(rows, ignore_index=True)

# ── Match to generation method ──
# Score CSVs have solution_preview = answer[:200] with \n -> \\n
# Match these against JSONL solutions via the same preview transform
preview_to_method = {}
for sol, rec in lookup_by_sol.items():
    preview = sol[:200].replace("\n", "\\n")
    if preview not in preview_to_method:
        preview_to_method[preview] = categorize(rec)

df['gen_method'] = df['solution_preview'].map(preview_to_method)
matched = df['gen_method'].notna().sum()
print(f'Matched {matched}/{len(df)} rows to generation method')

df_matched = df[df['gen_method'].notna()].copy()

CATEGORIES = ['gpt-4o standard', 'gpt-4o intentional_bug',
              'gpt-4o-mini standard', 'gpt-4o-mini intentional_bug']
CAT_COLORS = {
    'gpt-4o standard': '#2ecc71',
    'gpt-4o intentional_bug': '#27ae60',
    'gpt-4o-mini standard': '#3498db',
    'gpt-4o-mini intentional_bug': '#2980b9',
}
SHORT = {
    'gpt-4o standard': '4o std',
    'gpt-4o intentional_bug': '4o bug',
    'gpt-4o-mini standard': '4o-mini std',
    'gpt-4o-mini intentional_bug': '4o-mini bug',
}

# ── Plots ──
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Log probability scores by generation method (test set, gemma-2-9b-it)',
             fontsize=14, fontweight='bold')

metrics = [
    ('gen_score', 'log P(y|x)', 'Raw gen score'),
    ('self_tc', 'log P(y|x) − log P(y)', 'Self-typicality corrected'),
    ('neg_tc', 'log P(y|x) − log P(y|neg_x)', 'Neg-typicality corrected'),
    ('log_py', 'log P(y)', 'Unconditional'),
    ('log_py_neg', 'log P(y|neg_x)', 'Negated prompt'),
    ('gen_score_lenorm', 'log P(y|x) / num_tokens', 'Length-normalized'),
]

for idx, (col, ylabel, title) in enumerate(metrics):
    ax = axes[idx // 3, idx % 3]

    positions = []
    pos = 0
    tick_positions = []
    tick_labels = []

    for cat in CATEGORIES:
        sub = df_matched[df_matched['gen_method'] == cat]
        correct = sub[sub['correct'] == 'yes'][col].dropna()
        incorrect = sub[sub['correct'] == 'no'][col].dropna()

        if len(correct) == 0 and len(incorrect) == 0:
            continue

        bp = ax.boxplot([correct.values, incorrect.values],
                        positions=[pos, pos+1], widths=0.6,
                        patch_artist=True, showfliers=False)

        bp['boxes'][0].set_facecolor(CAT_COLORS[cat])
        bp['boxes'][0].set_alpha(0.8)
        bp['boxes'][1].set_facecolor(CAT_COLORS[cat])
        bp['boxes'][1].set_alpha(0.3)

        for element in ['whiskers', 'caps']:
            for line in bp[element]:
                line.set_color('gray')
        for line in bp['medians']:
            line.set_color('black')
            line.set_linewidth(1.5)

        # Add mean markers
        ax.scatter([pos], [correct.mean()], marker='D', color='black', s=20, zorder=5)
        ax.scatter([pos+1], [incorrect.mean()], marker='D', color='black', s=20, zorder=5)

        tick_positions.extend([pos, pos+1])
        tick_labels.extend([f'{SHORT[cat]}\n✓', f'{SHORT[cat]}\n✗'])
        pos += 3

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=6.5)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='gray', alpha=0.8, label='Correct (✓)'),
                   Patch(facecolor='gray', alpha=0.3, label='Incorrect (✗)'),
                   plt.Line2D([0], [0], marker='D', color='black', linestyle='None',
                              markersize=5, label='Mean')]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('results/humaneval_scores_by_gen_method.png', dpi=150, bbox_inches='tight')
print('Saved to results/humaneval_scores_by_gen_method.png')
plt.close()

# ── Print summary stats ──
print('\nSummary: mean scores by method and correctness')
print(f'{"Method":<25s} {"Label":<10s} {"n":>4s} {"logP(y|x)":>10s} {"self_tc":>10s} {"neg_tc":>10s} {"logP(y)":>10s} {"logP(y|neg)":>11s}')
print('-' * 95)
for cat in CATEGORIES:
    for label in ['yes', 'no']:
        sub = df_matched[(df_matched['gen_method'] == cat) & (df_matched['correct'] == label)]
        if len(sub) == 0:
            continue
        lbl = 'correct' if label == 'yes' else 'incorrect'
        print(f'{SHORT[cat]:<25s} {lbl:<10s} {len(sub):>4d} '
              f'{sub["gen_score"].mean():>10.1f} '
              f'{sub["self_tc"].mean():>10.1f} '
              f'{sub["neg_tc"].mean():>10.1f} '
              f'{sub["log_py"].mean():>10.1f} '
              f'{sub["log_py_neg"].mean():>10.1f}')
