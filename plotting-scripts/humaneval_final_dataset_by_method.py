"""Visualize final balanced dataset composition by generation method."""
import json, csv, glob, os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Load JSONL lookup
lookup = {}
with open('data/humaneval/solutions.jsonl') as f:
    for line in f:
        r = json.loads(line)
        key = (r['task_id'], r['solution'].strip())
        if key not in lookup:
            lookup[key] = r

def categorize(r):
    model = r.get('model', 'unknown')
    strategy = r.get('strategy', 'standard')
    return f'{model} {strategy}'

CATEGORIES = ['gpt-4o standard', 'gpt-4o intentional_bug',
              'gpt-4o-mini standard', 'gpt-4o-mini intentional_bug']
CAT_COLORS = {
    'gpt-4o standard': '#2ecc71',
    'gpt-4o intentional_bug': '#27ae60',
    'gpt-4o-mini standard': '#3498db',
    'gpt-4o-mini intentional_bug': '#2980b9',
}
SHORT_NAMES = {
    'gpt-4o standard': '4o\nstandard',
    'gpt-4o intentional_bug': '4o\nintentional_bug',
    'gpt-4o-mini standard': '4o-mini\nstandard',
    'gpt-4o-mini intentional_bug': '4o-mini\nintentional_bug',
}

def match_csv(csv_path):
    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    stats = defaultdict(lambda: {'pass': 0, 'fail': 0})
    for row in rows:
        answer = row['answer'].strip()
        for (tid, sol), rec in lookup.items():
            if sol == answer:
                cat = categorize(rec)
                label = 'pass' if row['correct'].strip().lower() == 'yes' else 'fail'
                stats[cat][label] += 1
                break
    return dict(stats), len(rows)

# Gather per-problem test stats
test_csvs = sorted(glob.glob('data/humaneval/with_solutions/humaneval_*.csv'))
per_problem = {}
for csv_path in test_csvs:
    slug = os.path.basename(csv_path)[:-4]
    stats, n = match_csv(csv_path)
    per_problem[slug] = stats

# Aggregate test and train
all_test = defaultdict(lambda: {'pass': 0, 'fail': 0})
for slug, stats in per_problem.items():
    for cat, counts in stats.items():
        all_test[cat]['pass'] += counts['pass']
        all_test[cat]['fail'] += counts['fail']

train_stats, n_train = match_csv('data/humaneval/with_solutions/train.csv')

# Pre-balance stats (from JSONL)
pre_balance = defaultdict(lambda: {'pass': 0, 'fail': 0})
for (tid, sol), rec in lookup.items():
    cat = categorize(rec)
    if rec['passed']:
        pre_balance[cat]['pass'] += 1
    else:
        pre_balance[cat]['fail'] += 1

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('HumanEval: Final Balanced Dataset by Generation Method',
             fontsize=14, fontweight='bold')

# Panel 1: Before vs after balancing - stacked bars
ax = axes[0, 0]
x = np.arange(len(CATEGORIES))
w = 0.35

pre_pass = [pre_balance[c]['pass'] for c in CATEGORIES]
pre_fail = [pre_balance[c]['fail'] for c in CATEGORIES]
post_pass = [all_test[c]['pass'] + train_stats.get(c, {}).get('pass', 0) for c in CATEGORIES]
post_fail = [all_test[c]['fail'] + train_stats.get(c, {}).get('fail', 0) for c in CATEGORIES]

b1 = ax.bar(x - w/2, [p+f for p, f in zip(pre_pass, pre_fail)], w,
            color=[CAT_COLORS[c] for c in CATEGORIES], alpha=0.4, label='Before balancing')
b2 = ax.bar(x + w/2, [p+f for p, f in zip(post_pass, post_fail)], w,
            color=[CAT_COLORS[c] for c in CATEGORIES], alpha=0.9, label='After balancing')
ax.set_xticks(x)
ax.set_xticklabels([SHORT_NAMES[c] for c in CATEGORIES], fontsize=8)
ax.set_ylabel('Number of solutions')
ax.set_title('Solutions before vs after balancing')
ax.legend()
for bar, count in zip(b1, [p+f for p, f in zip(pre_pass, pre_fail)]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            str(count), ha='center', fontsize=8, alpha=0.6)
for bar, count in zip(b2, [p+f for p, f in zip(post_pass, post_fail)]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            str(count), ha='center', fontsize=8, fontweight='bold')

# Panel 2: Final dataset composition (pie-ish) - train + test combined
ax = axes[0, 1]
final_totals = {c: post_pass[i] + post_fail[i] for i, c in enumerate(CATEGORIES)}
sizes = [final_totals[c] for c in CATEGORIES]
colors = [CAT_COLORS[c] for c in CATEGORIES]
labels = [f'{c}\n({final_totals[c]})' for c in CATEGORIES]
wedges, texts, autotexts = ax.pie(sizes, labels=None, colors=colors, autopct='%1.1f%%',
                                    startangle=90, pctdistance=0.75)
ax.legend(wedges, [c.replace('gpt-4o', '4o').replace('-mini', '-mini') for c in CATEGORIES],
          loc='lower left', fontsize=8)
ax.set_title(f'Final dataset composition\n(train + test = {sum(sizes)} solutions)')

# Panel 3: Test set - per-problem stacked bars by method
ax = axes[1, 0]
slugs_sorted = sorted(per_problem.keys(),
                       key=lambda s: sum(v['pass']+v['fail']
                                         for v in per_problem[s].values()))
y = np.arange(len(slugs_sorted))
left = np.zeros(len(slugs_sorted))

for cat in CATEGORIES:
    vals = [per_problem[s].get(cat, {}).get('pass', 0) +
            per_problem[s].get(cat, {}).get('fail', 0) for s in slugs_sorted]
    ax.barh(y, vals, left=left, color=CAT_COLORS[cat], alpha=0.85,
            height=0.8, label=cat)
    left += np.array(vals)

ax.set_yticks(y)
ax.set_yticklabels([s.replace('humaneval_', '') for s in slugs_sorted], fontsize=7)
ax.set_xlabel('Number of solutions')
ax.set_title(f'Test set: solutions per problem by method')
ax.legend(fontsize=7, loc='lower right')

# Panel 4: Where do the INCORRECT solutions come from?
ax = axes[1, 1]
test_fail = {c: all_test[c]['fail'] for c in CATEGORIES}
train_fail = {c: train_stats.get(c, {}).get('fail', 0) for c in CATEGORIES}
combined_fail = {c: test_fail[c] + train_fail[c] for c in CATEGORIES}

x = np.arange(len(CATEGORIES))
w = 0.6
vals = [combined_fail[c] for c in CATEGORIES]
bars = ax.bar(x, vals, w, color=[CAT_COLORS[c] for c in CATEGORIES], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([SHORT_NAMES[c] for c in CATEGORIES], fontsize=8)
ax.set_ylabel('Number of incorrect solutions')
ax.set_title('Source of incorrect (failing) solutions in final dataset')
for bar, count in zip(bars, vals):
    pct = count / sum(vals) * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{count}\n({pct:.0f}%)', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('results/humaneval_final_dataset_by_method.png', dpi=150, bbox_inches='tight')
print('Saved to results/humaneval_final_dataset_by_method.png')
plt.close()
