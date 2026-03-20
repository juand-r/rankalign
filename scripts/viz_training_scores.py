"""
Visualize how gen and val scores change during training.
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def load_training_logs(log_dir, task_pattern):
    """Load all step CSVs for a given task pattern."""
    pattern = os.path.join(log_dir, f"*{task_pattern}*-step*.csv")
    files = glob.glob(pattern)
    
    # Filter out pair files
    files = [f for f in files if '-pair.csv' not in f]
    
    # Extract step numbers and sort
    step_files = []
    for f in files:
        # Extract step number from filename like ...-step60.csv
        basename = os.path.basename(f)
        step_part = basename.split('-step')[-1].replace('.csv', '')
        try:
            step = int(step_part)
            step_files.append((step, f))
        except ValueError:
            continue
    
    step_files.sort(key=lambda x: x[0])
    return step_files


def load_ground_truth(task):
    """Load ground truth labels from the fixed-hypernyms data."""
    # Try to load the training data to get ground truth
    data_path = f"/datastor1/jdr/gv-gap/rankalign/data/fixed-hypernyms/hypernym_{task}_google-gemma-2-2b_train-fixed.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        # Create a lookup dict: (noun1, noun2) -> label
        # In fixed-hypernyms CSVs: noun2 is stored as 'fixed_hypernym_generator'
        # gpt4_ground_truth is 'Yes'/'No' string
        label_lookup = {}
        for _, row in df.iterrows():
            noun2 = row.get('fixed_hypernym_generator', row.get('noun2', ''))
            key = (row['noun1'], noun2)
            gt = row.get('gpt4_ground_truth', row.get('label', None))
            # Convert 'Yes'/'No' to 1/0
            if gt == 'Yes':
                label_lookup[key] = 1
            elif gt == 'No':
                label_lookup[key] = 0
            else:
                label_lookup[key] = gt
        return label_lookup
    return None


def plot_score_trajectories(step_files, task, label_lookup=None, output_path=None):
    """Plot how scores change over training steps."""
    
    steps = []
    mean_gen_pos, mean_gen_neg = [], []
    mean_val_pos, mean_val_neg = [], []
    std_gen_pos, std_gen_neg = [], []
    std_val_pos, std_val_neg = [], []
    
    for step, filepath in step_files:
        df = pd.read_csv(filepath)
        
        # Try to get labels
        if label_lookup:
            df['label'] = df.apply(
                lambda row: label_lookup.get((row['noun1'], row['noun2']), None), 
                axis=1
            )
        elif 'label' in df.columns:
            pass  # Already has labels
        else:
            print(f"Warning: No labels available for {filepath}")
            continue
        
        # Filter to rows with valid labels
        df = df[df['label'].notna()]
        
        pos = df[df['label'] == 1]
        neg = df[df['label'] == 0]
        
        steps.append(step)
        mean_gen_pos.append(pos['gen_score'].mean())
        mean_gen_neg.append(neg['gen_score'].mean())
        mean_val_pos.append(pos['val_score'].mean())
        mean_val_neg.append(neg['val_score'].mean())
        std_gen_pos.append(pos['gen_score'].std())
        std_gen_neg.append(neg['gen_score'].std())
        std_val_pos.append(pos['val_score'].std())
        std_val_neg.append(neg['val_score'].std())
    
    steps = np.array(steps)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Score Trajectories During Training: {task}', fontsize=14)
    
    # Plot 1: Generator scores by label
    ax = axes[0, 0]
    ax.plot(steps, mean_gen_pos, 'g-', label='Positive (label=1)', linewidth=2)
    ax.fill_between(steps, 
                    np.array(mean_gen_pos) - np.array(std_gen_pos),
                    np.array(mean_gen_pos) + np.array(std_gen_pos),
                    alpha=0.2, color='green')
    ax.plot(steps, mean_gen_neg, 'r-', label='Negative (label=0)', linewidth=2)
    ax.fill_between(steps,
                    np.array(mean_gen_neg) - np.array(std_gen_neg),
                    np.array(mean_gen_neg) + np.array(std_gen_neg),
                    alpha=0.2, color='red')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Generator Score (log-prob)')
    ax.set_title('Generator Scores by Label')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validator scores by label
    ax = axes[0, 1]
    ax.plot(steps, mean_val_pos, 'g-', label='Positive (label=1)', linewidth=2)
    ax.fill_between(steps,
                    np.array(mean_val_pos) - np.array(std_val_pos),
                    np.array(mean_val_pos) + np.array(std_val_pos),
                    alpha=0.2, color='green')
    ax.plot(steps, mean_val_neg, 'r-', label='Negative (label=0)', linewidth=2)
    ax.fill_between(steps,
                    np.array(mean_val_neg) - np.array(std_val_neg),
                    np.array(mean_val_neg) + np.array(std_val_neg),
                    alpha=0.2, color='red')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Threshold (0)')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Validator Score (log-odds)')
    ax.set_title('Validator Scores by Label')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Score separation (gap between pos and neg)
    ax = axes[1, 0]
    gen_gap = np.array(mean_gen_pos) - np.array(mean_gen_neg)
    val_gap = np.array(mean_val_pos) - np.array(mean_val_neg)
    ax.plot(steps, gen_gap, 'b-', label='Generator gap', linewidth=2)
    ax.plot(steps, val_gap, 'orange', label='Validator gap', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Score Gap (pos - neg)')
    ax.set_title('Score Separation Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Accuracy proxy (fraction above/below threshold)
    ax = axes[1, 1]
    acc_gen = []
    acc_val = []
    for step, filepath in step_files:
        df = pd.read_csv(filepath)
        if label_lookup:
            df['label'] = df.apply(
                lambda row: label_lookup.get((row['noun1'], row['noun2']), None),
                axis=1
            )
        df = df[df['label'].notna()]
        
        # For generator, higher score = more likely positive (use median as threshold)
        gen_thresh = df['gen_score'].median()
        gen_correct = ((df['gen_score'] > gen_thresh) & (df['label'] == 1)) | \
                      ((df['gen_score'] <= gen_thresh) & (df['label'] == 0))
        acc_gen.append(gen_correct.mean())
        
        # For validator, threshold is 0 (log-odds)
        val_correct = ((df['val_score'] > 0) & (df['label'] == 1)) | \
                      ((df['val_score'] <= 0) & (df['label'] == 0))
        acc_val.append(val_correct.mean())
    
    ax.plot(steps, acc_gen, 'b-', label='Generator (median thresh)', linewidth=2)
    ax.plot(steps, acc_val, 'orange', label='Validator (thresh=0)', linewidth=2)
    ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Random')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Accuracy')
    ax.set_title('Classification Accuracy Over Training')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='bananas', 
                        help='Task name (e.g., bananas, dogs, crows)')
    parser.add_argument('--log-dir', type=str, 
                        default='/datastor1/jdr/gv-gap/rankalign/outputs/training-logs',
                        help='Directory containing training logs')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for figure')
    args = parser.parse_args()
    
    # Load training logs
    step_files = load_training_logs(args.log_dir, f'hypernym-{args.task}')
    print(f"Found {len(step_files)} step files for {args.task}")
    
    if not step_files:
        print("No files found!")
        return
    
    for step, f in step_files[:5]:
        print(f"  Step {step}: {os.path.basename(f)}")
    if len(step_files) > 5:
        print(f"  ... and {len(step_files) - 5} more")
    
    # Load ground truth
    label_lookup = load_ground_truth(args.task)
    if label_lookup:
        print(f"Loaded {len(label_lookup)} ground truth labels")
    
    # Plot
    output_path = args.output or f'/datastor1/jdr/gv-gap/rankalign/outputs/training-logs/viz_{args.task}.png'
    plot_score_trajectories(step_files, args.task, label_lookup, output_path)


if __name__ == '__main__':
    main()
