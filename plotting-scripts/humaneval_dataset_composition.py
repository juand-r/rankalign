"""Visualize HumanEval dataset composition."""
import csv, glob, os
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = 'data/humaneval/with_solutions'

# Load test problem stats
test_csvs = sorted(glob.glob(os.path.join(DATA_DIR, 'humaneval_*.csv')))
problems = []
for csv_path in test_csvs:
    slug = os.path.basename(csv_path)[:-4]
    num = int(slug.split('_')[1])
    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    pos = sum(1 for r in rows if r['correct'].strip().lower() == 'yes')
    neg = len(rows) - pos
    problems.append({'slug': slug, 'num': num, 'total': len(rows), 'pos': pos, 'neg': neg})

# Load train stats
with open(os.path.join(DATA_DIR, 'train.csv'), newline='', encoding='utf-8') as f:
    train_rows = list(csv.DictReader(f))

train_by_q = {}
for r in train_rows:
    q = r['question'][:80]
    if q not in train_by_q:
        train_by_q[q] = {'pos': 0, 'neg': 0}
    if r['correct'].strip().lower() == 'yes':
        train_by_q[q]['pos'] += 1
    else:
        train_by_q[q]['neg'] += 1

train_problems = sorted(train_by_q.values(), key=lambda x: x['pos'] + x['neg'])

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('HumanEval Dataset Composition\n(164 original problems → 120 qualified → 80 train + 40 test)',
             fontsize=14, fontweight='bold')

# Panel 1: Pipeline funnel
ax = axes[0, 0]
stages = ['Original\nHumanEval', 'Qualified\n(≥10 pos, ≥10 neg)', 'Train\nproblems', 'Test\nproblems']
counts = [164, 120, 80, 40]
colors = ['#bdc3c7', '#95a5a6', '#3498db', '#e74c3c']
bars = ax.barh(stages, counts, color=colors, edgecolor='white', linewidth=1.5)
for bar, count in zip(bars, counts):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
            str(count), va='center', fontweight='bold', fontsize=13)
ax.set_xlabel('Number of problems')
ax.set_title('Dataset pipeline')
ax.set_xlim(0, 185)

# Panel 2: Test problems - solutions per problem (sorted)
ax = axes[0, 1]
problems_sorted = sorted(problems, key=lambda x: x['total'])
slugs = [p['slug'].replace('humaneval_', '') for p in problems_sorted]
pos_vals = [p['pos'] for p in problems_sorted]
neg_vals = [p['neg'] for p in problems_sorted]
y = np.arange(len(slugs))
ax.barh(y, pos_vals, color='#2ecc71', label='Correct', height=0.8)
ax.barh(y, [-n for n in neg_vals], color='#e74c3c', label='Incorrect', height=0.8)
ax.set_yticks(y)
ax.set_yticklabels(slugs, fontsize=7)
ax.set_xlabel('← Incorrect | Correct →')
ax.set_title('Test set: solutions per problem (40 problems)')
ax.legend(loc='lower right', fontsize=9)
ax.axvline(0, color='black', linewidth=0.5)

# Panel 3: Distribution of solutions per problem (histogram)
ax = axes[1, 0]
test_totals = [p['total'] for p in problems]
train_totals = [v['pos'] + v['neg'] for v in train_by_q.values()]
bins = np.arange(15, 85, 5)
ax.hist(train_totals, bins=bins, alpha=0.7, color='#3498db', label=f'Train (n={len(train_by_q)})', edgecolor='white')
ax.hist(test_totals, bins=bins, alpha=0.7, color='#e74c3c', label=f'Test (n={len(problems)})', edgecolor='white')
ax.set_xlabel('Solutions per problem')
ax.set_ylabel('Number of problems')
ax.set_title('Distribution of solutions per problem')
ax.legend()

# Panel 4: Overall numbers
ax = axes[1, 1]
ax.axis('off')
train_pos = sum(1 for r in train_rows if r['correct'].strip().lower() == 'yes')
train_neg = len(train_rows) - train_pos
test_pos = sum(p['pos'] for p in problems)
test_neg = sum(p['neg'] for p in problems)

text = (
    f"Original HumanEval benchmark: 164 problems\n"
    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    f"Solution generation:\n"
    f"  • GPT-4o at temp 0.2–1.4 (40 samples/problem)\n"
    f"  • GPT-4o-mini for extra negatives (subtle bug prompt)\n"
    f"  • GPT-4o-mini at high temp for more failures\n"
    f"  • Each solution tested against HumanEval unit tests\n\n"
    f"Filtering: ≥10 passing AND ≥10 failing solutions\n"
    f"  → 120 qualified problems (44 dropped)\n\n"
    f"Split: OOD by problem (seed=42)\n"
    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    f"Train: 80 problems, {len(train_rows)} solutions\n"
    f"  ({train_pos} correct, {train_neg} incorrect — balanced)\n\n"
    f"Test:  40 problems, {test_pos + test_neg} solutions\n"
    f"  ({test_pos} correct, {test_neg} incorrect — balanced)\n\n"
    f"All problems balanced at 50% pos / 50% neg\n"
    f"Solutions range from 20–78 per problem"
)
ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#dee2e6'))

plt.tight_layout()
plt.savefig('results/humaneval_dataset_composition.png', dpi=150, bbox_inches='tight')
print('Saved to results/humaneval_dataset_composition.png')
plt.close()
