"""
Animate how gen/val scores evolve during training.
Creates a video showing the point cloud at each training step.
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import argparse
from matplotlib.colors import LinearSegmentedColormap


def load_training_logs(log_dir, task_pattern):
    """Load all step CSVs for a given task pattern."""
    pattern = os.path.join(log_dir, f"*{task_pattern}*-step*.csv")
    files = glob.glob(pattern)
    
    # Filter out pair files
    files = [f for f in files if '-pair.csv' not in f]
    
    # Extract step numbers and sort
    step_files = []
    for f in files:
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
    data_path = f"/datastor1/jdr/gv-gap/rankalign/data/fixed-hypernyms/hypernym_{task}_google-gemma-2-2b_train-fixed.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        label_lookup = {}
        for _, row in df.iterrows():
            noun2 = row.get('fixed_hypernym_generator', row.get('noun2', ''))
            gt = row.get('gpt4_ground_truth', '')
            label_lookup[(row['noun1'], noun2)] = 1 if gt == 'Yes' else 0
        return label_lookup
    return None


def create_animation(step_files, task, label_lookup, output_path, fps=2):
    """Create animation of score evolution."""
    
    # Load all data first to determine global axis limits
    all_data = []
    for step, filepath in step_files:
        df = pd.read_csv(filepath)
        if label_lookup:
            df['label'] = df.apply(
                lambda row: label_lookup.get((row['noun1'], row['noun2']), None),
                axis=1
            )
        df = df[df['label'].notna()]
        df['step'] = step
        all_data.append(df)
    
    if not all_data:
        print("No data loaded!")
        return
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Determine axis limits with some padding
    gen_min, gen_max = combined['gen_score'].min(), combined['gen_score'].max()
    val_min, val_max = combined['val_score'].min(), combined['val_score'].max()
    gen_pad = (gen_max - gen_min) * 0.1
    val_pad = (val_max - val_min) * 0.1
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Initialize empty scatter plots
    scatter_pos = ax.scatter([], [], c='green', alpha=0.6, s=30, label='Positive (Yes)')
    scatter_neg = ax.scatter([], [], c='red', alpha=0.6, s=30, label='Negative (No)')
    
    # Add threshold line
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1, label='Val threshold (0)')
    
    # Set axis limits
    ax.set_xlim(gen_min - gen_pad, gen_max + gen_pad)
    ax.set_ylim(val_min - val_pad, val_max + val_pad)
    
    ax.set_xlabel('Generator Score (log-prob)', fontsize=12)
    ax.set_ylabel('Validator Score (log-odds)', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Title that will be updated
    title = ax.set_title(f'{task}: Step 0', fontsize=14, fontweight='bold')
    
    # Text annotations for stats
    stats_text = ax.text(0.98, 0.02, '', transform=ax.transAxes, 
                         fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def init():
        scatter_pos.set_offsets(np.empty((0, 2)))
        scatter_neg.set_offsets(np.empty((0, 2)))
        return scatter_pos, scatter_neg, title, stats_text
    
    def animate(frame_idx):
        df = all_data[frame_idx]
        step = step_files[frame_idx][0]
        
        pos = df[df['label'] == 1]
        neg = df[df['label'] == 0]
        
        # Update scatter data
        if len(pos) > 0:
            scatter_pos.set_offsets(pos[['gen_score', 'val_score']].values)
        else:
            scatter_pos.set_offsets(np.empty((0, 2)))
            
        if len(neg) > 0:
            scatter_neg.set_offsets(neg[['gen_score', 'val_score']].values)
        else:
            scatter_neg.set_offsets(np.empty((0, 2)))
        
        # Update title
        title.set_text(f'{task}: Step {step}')
        
        # Compute and display stats
        val_acc = ((df['val_score'] > 0) & (df['label'] == 1) | 
                   (df['val_score'] <= 0) & (df['label'] == 0)).mean()
        pos_val_mean = pos['val_score'].mean() if len(pos) > 0 else 0
        neg_val_mean = neg['val_score'].mean() if len(neg) > 0 else 0
        pos_gen_mean = pos['gen_score'].mean() if len(pos) > 0 else 0
        neg_gen_mean = neg['gen_score'].mean() if len(neg) > 0 else 0
        
        stats_str = (f'Val Acc: {val_acc:.1%}\n'
                    f'Pos val: {pos_val_mean:.2f}, gen: {pos_gen_mean:.2f}\n'
                    f'Neg val: {neg_val_mean:.2f}, gen: {neg_gen_mean:.2f}')
        stats_text.set_text(stats_str)
        
        return scatter_pos, scatter_neg, title, stats_text
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(all_data), interval=1000//fps, blit=True
    )
    
    # Save animation
    print(f"Saving animation to {output_path}...")
    if output_path.endswith('.gif'):
        writer = animation.PillowWriter(fps=fps)
    else:
        writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    
    anim.save(output_path, writer=writer)
    print(f"Animation saved!")
    
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Animate training score evolution')
    parser.add_argument('--task', type=str, default='bananas',
                        help='Task name (e.g., bananas, dogs, crows)')
    parser.add_argument('--log-dir', type=str,
                        default='/datastor1/jdr/gv-gap/rankalign/outputs/training-logs',
                        help='Directory containing training logs')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for animation (default: outputs/training-logs/anim_<task>.gif)')
    parser.add_argument('--fps', type=int, default=2,
                        help='Frames per second')
    parser.add_argument('--format', type=str, choices=['gif', 'mp4'], default='gif',
                        help='Output format')
    args = parser.parse_args()
    
    # Load training logs
    step_files = load_training_logs(args.log_dir, f'hypernym-{args.task}')
    print(f"Found {len(step_files)} step files for {args.task}")
    
    if not step_files:
        print("No files found!")
        return
    
    for step, f in step_files:
        print(f"  Step {step}: {os.path.basename(f)}")
    
    # Load ground truth
    label_lookup = load_ground_truth(args.task)
    if label_lookup:
        print(f"Loaded {len(label_lookup)} ground truth labels")
    else:
        print("Warning: No ground truth labels found")
        return
    
    # Output path
    output_path = args.output or f'{args.log_dir}/anim_{args.task}.{args.format}'
    
    # Create animation
    create_animation(step_files, args.task, label_lookup, output_path, fps=args.fps)


if __name__ == '__main__':
    main()
