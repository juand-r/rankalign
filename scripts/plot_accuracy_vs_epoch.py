"""
Plot discriminator accuracy vs epoch for selected tasks.
"""

import pandas as pd
import matplotlib.pyplot as plt
import re

# Load data
df = pd.read_csv('/datastor1/jdr/gv-gap/rankalign/outputs/eval_results.csv')

# Tasks to plot
tasks = ['kites', 'ducklings', 'elephants', 'dolls', 'jackets']

# Extract epoch from model name
def get_epoch(model_name):
    match = re.search(r'epoch(\d+)', model_name)
    return int(match.group(1)) if match else None

def get_task(task_name):
    # Extract task from "hypernym-kites" -> "kites"
    return task_name.replace('hypernym-', '')

# Add epoch and clean task columns
df['epoch'] = df['model'].apply(get_epoch)
df['task_clean'] = df['task'].apply(get_task)

# Filter to our tasks and epochs 1-9
df_filtered = df[df['task_clean'].isin(tasks) & df['epoch'].between(1, 9)]

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each task
colors = {'kites': 'blue', 'ducklings': 'orange', 'elephants': 'green', 'dolls': 'red', 'jackets': 'purple'}
markers = {'kites': 'o', 'ducklings': 's', 'elephants': '^', 'dolls': 'd', 'jackets': 'v'}

for task in tasks:
    task_data = df_filtered[df_filtered['task_clean'] == task].sort_values('epoch')
    ax.plot(task_data['epoch'], task_data['disc_acc'], 
            marker=markers[task], linestyle='-', linewidth=2, markersize=8,
            color=colors[task], label=task)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Discriminator Accuracy', fontsize=12)
ax.set_title('Discriminator Accuracy vs Training Epoch', fontsize=14)
ax.set_xticks(range(1, 10))
ax.set_xlim(0.5, 9.5)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/datastor1/jdr/gv-gap/rankalign/outputs/accuracy_vs_epoch.png', dpi=150)
print("Saved to /datastor1/jdr/gv-gap/rankalign/outputs/accuracy_vs_epoch.png")
plt.show()
