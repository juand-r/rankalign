"""Visualize HumanEval solutions per problem by generation method."""
import json, os, random
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

with open('data/humaneval/solutions.jsonl') as f:
    records = [json.loads(line) for line in f]

# Categorize each solution
def categorize(r):
    model = r.get('model', 'unknown')
    strategy = r.get('strategy', 'standard')
    if model == 'gpt-4o' and strategy == 'standard':
        return 'gpt-4o standard'
    elif model == 'gpt-4o' and strategy == 'intentional_bug':
        return 'gpt-4o intentional_bug'
    elif model == 'gpt-4o-mini' and strategy == 'standard':
        return 'gpt-4o-mini standard'
    elif model == 'gpt-4o-mini' and strategy == 'intentional_bug':
        return 'gpt-4o-mini intentional_bug'
    return f'{model} {strategy}'

categories = ['gpt-4o standard', 'gpt-4o intentional_bug',
              'gpt-4o-mini standard', 'gpt-4o-mini intentional_bug']
cat_colors = {
    'gpt-4o standard': '#2ecc71',
    'gpt-4o intentional_bug': '#27ae60',
    'gpt-4o-mini standard': '#3498db',
    'gpt-4o-mini intentional_bug': '#2980b9',
}

# Build per-problem stats
by_problem = defaultdict(lambda: defaultdict(lambda: {'pass': 0, 'fail': 0}))
for r in records:
    cat = categorize(r)
    tid = r['task_id']
    if r['passed']:
        by_problem[tid][cat]['pass'] += 1
    else:
        by_problem[tid][cat]['fail'] += 1

# Determine train/test split (reproduce from build script)
qualified = {}
for tid, cats in by_problem.items():
    total_pass = sum(v['pass'] for v in cats.values())
    total_fail = sum(v['fail'] for v in cats.values())
    if total_pass >= 10 and total_fail >= 10:
        qualified[tid] = cats

task_ids = sorted(qualified.keys())
rng = random.Random(42)
rng.shuffle(task_ids)
train_ids = set(task_ids[:100])
test_ids = set(task_ids[100:])

def num_from_tid(tid):
    return int(tid.split('/')[1])

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle(f'HumanEval Solutions by Generation Method\n'
             f'({len(records)} total solutions across {len(by_problem)} problems, '
             f'{len(qualified)} qualified)',
             fontsize=14, fontweight='bold')

# Panel 1: Test problems - stacked bar of pass/fail by method
ax = axes[0, 0]
test_problems = sorted([tid for tid in test_ids], key=num_from_tid)
y = np.arange(len(test_problems))
left_pass = np.zeros(len(test_problems))
left_fail = np.zeros(len(test_problems))

for cat in categories:
    pass_vals = [by_problem[tid][cat]['pass'] for tid in test_problems]
    fail_vals = [by_problem[tid][cat]['fail'] for tid in test_problems]
    labels = [tid.split('/')[1] for tid in test_problems]

    ax.barh(y, pass_vals, left=left_pass, color=cat_colors[cat],
            alpha=0.9, height=0.8, label=f'{cat} (pass)')
    ax.barh(y, [-v for v in fail_vals], left=-left_fail, color=cat_colors[cat],
            alpha=0.4, height=0.8, label=f'{cat} (fail)')
    left_pass += np.array(pass_vals)
    left_fail += np.array(fail_vals)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=7)
ax.set_xlabel('← Failing | Passing →')
ax.set_title(f'Test problems ({len(test_problems)}): solutions by method')
ax.axvline(0, color='black', linewidth=0.5)
handles = [plt.Rectangle((0,0),1,1, fc=cat_colors[c], alpha=a)
           for c in categories for a in [0.9, 0.4]]
labels_leg = [f'{c} {"pass" if i==0 else "fail"}'
              for c in categories for i in [0, 1]]
ax.legend(handles, labels_leg, fontsize=6, loc='lower right', ncol=2)

# Panel 2: Aggregate by method - how many pass vs fail
ax = axes[0, 1]
method_totals = defaultdict(lambda: {'pass': 0, 'fail': 0})
for r in records:
    cat = categorize(r)
    if r['passed']:
        method_totals[cat]['pass'] += 1
    else:
        method_totals[cat]['fail'] += 1

x = np.arange(len(categories))
pass_counts = [method_totals[c]['pass'] for c in categories]
fail_counts = [method_totals[c]['fail'] for c in categories]
w = 0.35
bars1 = ax.bar(x - w/2, pass_counts, w, color=[cat_colors[c] for c in categories],
               alpha=0.9, label='Pass')
bars2 = ax.bar(x + w/2, fail_counts, w, color=[cat_colors[c] for c in categories],
               alpha=0.4, label='Fail')
ax.set_xticks(x)
ax.set_xticklabels([c.replace(' ', '\n') for c in categories], fontsize=8)
ax.set_ylabel('Number of solutions')
ax.set_title('Overall pass/fail by generation method')
ax.legend()
for bar, count in zip(bars1, pass_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            str(count), ha='center', fontsize=8)
for bar, count in zip(bars2, fail_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            str(count), ha='center', fontsize=8)

# Panel 3: Pass rate by method
ax = axes[1, 0]
pass_rates = []
for cat in categories:
    t = method_totals[cat]
    rate = t['pass'] / (t['pass'] + t['fail']) * 100
    pass_rates.append(rate)
bars = ax.bar(categories, pass_rates, color=[cat_colors[c] for c in categories], alpha=0.85)
ax.set_ylabel('Pass rate (%)')
ax.set_title('Pass rate by generation method')
ax.set_xticklabels([c.replace(' ', '\n') for c in categories], fontsize=8)
for bar, rate in zip(bars, pass_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{rate:.1f}%', ha='center', fontsize=10, fontweight='bold')
ax.set_ylim(0, 100)
ax.axhline(50, color='gray', linestyle='--', alpha=0.5)

# Panel 4: Temperature distribution by method, colored by pass/fail
ax = axes[1, 1]
temp_data = defaultdict(lambda: defaultdict(lambda: {'pass': 0, 'fail': 0}))
for r in records:
    cat = categorize(r)
    t = r.get('temperature', -1)
    if r['passed']:
        temp_data[cat][t]['pass'] += 1
    else:
        temp_data[cat][t]['fail'] += 1

temps = sorted(set(r.get('temperature', -1) for r in records))
x = np.arange(len(temps))
width = 0.2
for i, cat in enumerate(categories):
    totals = [temp_data[cat][t]['pass'] + temp_data[cat][t]['fail'] for t in temps]
    ax.bar(x + i*width - 1.5*width, totals, width,
           color=cat_colors[cat], alpha=0.85, label=cat)
ax.set_xticks(x)
ax.set_xticklabels([str(t) for t in temps], fontsize=8)
ax.set_xlabel('Temperature')
ax.set_ylabel('Number of solutions')
ax.set_title('Solutions by temperature and method')
ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('results/humaneval_solutions_by_method.png', dpi=150, bbox_inches='tight')
print('Saved to results/humaneval_solutions_by_method.png')
plt.close()
