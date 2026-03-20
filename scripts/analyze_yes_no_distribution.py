#!/usr/bin/env python3
"""
Analyze the probability distribution over Yes/No tokens for V2G trained models.

For each hypernym-X task, load the V2G trained model and run it on the test set
discriminator prompts, tracking:
- P(Yes variants): "Yes", " Yes", "YES", "yes", " yes"
- P(No variants): "No", " No", "NO", "no", " no"  
- P(remaining): 1 - P(Yes) - P(No)
"""

from typing import Any


import sys
sys.path.insert(0, '/datastor1/jdr/gv-gap/rankalign/src')

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
from collections import defaultdict

from utils import get_final_logit_prob, get_model_input_device
from task_registry import get_task

# Import tasks to register them
import tasks

# Yes/No token variants
YES_WORDS = ["Yes", " Yes", "YES", "yes", " yes"]
NO_WORDS = ["No", " No", "NO", "no", " no"]

TASKS = [
    "hypernym-bananas",
    "hypernym-bazookas", 
    "hypernym-cabinets",
    "hypernym-cars",
    "hypernym-chairs",
    "hypernym-crows",
    "hypernym-diapers",
    "hypernym-dogs"
]

# Model paths (V2G = d2g, epoch2, typcorr, full-completion)
MODEL_TEMPLATE = "/datastor1/jdr/gv-gap/rankalign/models/v5-google--gemma-2-2b-delta0.15-epoch2--{task}-all--d2g--random--alpha1.0--typcorr--full-completion--nllv1.0--nllg1.0"
BASE_MODEL = "google/gemma-2-2b"


def get_token_ids(tokenizer, words):
    """Get token IDs for a list of words."""
    token_ids = []
    for w in words:
        tokens = tokenizer.encode(w, add_special_tokens=False)
        # Take the last token (handles space prefix)
        token_ids.append(tokens[-1])
    return token_ids


def analyze_model_on_task(model, tokenizer, task, device, disc_shots=5):
    """Run model on task's test set and collect probability distributions."""
    
    # Load test data using task registry
    task_config = get_task(task)
    if task_config is None:
        raise ValueError(f"Task {task} not found in registry")
    
    _, L_test = task_config['load_data'](seed=42, split_type='random', v2=True)
    make_prompt = task_config['make_prompt']
    
    # Get token IDs for Yes/No variants
    yes_token_ids = get_token_ids(tokenizer, YES_WORDS)
    no_token_ids = get_token_ids(tokenizer, NO_WORDS)
    
    print(f"  Yes token IDs: {dict(zip(YES_WORDS, yes_token_ids))}")
    print(f"  No token IDs: {dict(zip(NO_WORDS, no_token_ids))}")
    
    results = []
    
    for i, item in enumerate(tqdm(L_test, desc=f"  Processing {task}")):
        # Create discriminator prompt
        prompt_obj = make_prompt(item, style='discriminator', shots=disc_shots)
        prompt_disc = prompt_obj.prompt
        
        # get_final_logit_prob returns PROBABILITIES (not log probs) for the last position
        # Shape: [vocab_size]
        probs = get_final_logit_prob(prompt_disc, model, tokenizer, device, is_chat=False)
        
        # Debug first item
        if i == 0:
            print(f"  DEBUG: probs shape = {probs.shape}")
            print(f"  DEBUG: probs sum = {probs.sum().item():.4f}")
            print(f"  DEBUG: prompt_disc = {prompt_disc[:200]}...")
        
        # Extract probabilities for each Yes/No variant
        yes_probs = {w: probs[tid].item() for w, tid in zip(YES_WORDS, yes_token_ids)}
        no_probs = {w: probs[tid].item() for w, tid in zip(NO_WORDS, no_token_ids)}
        
        # Total Yes/No probability
        total_yes = sum(yes_probs.values())
        total_no = sum(no_probs.values())
        remaining = 1.0 - total_yes - total_no
        
        # Ground truth label
        gold_label = item.taxonomic.strip().lower()  # 'yes' or 'no'
        
        results.append({
            'gold': gold_label,
            'total_yes': total_yes,
            'total_no': total_no,
            'remaining': remaining,
            'log_odds': np.log(total_yes + 1e-12) - np.log(total_no + 1e-12),
            **{f'p_{w}': p for w, p in yes_probs.items()},
            **{f'p_{w}': p for w, p in no_probs.items()},
        })
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disc-shots", type=int, default=5, help="Number of few-shot examples for discriminator")
    parser.add_argument("--include-base", action="store_true", help="Also analyze base model")
    parser.add_argument("--tasks", nargs="+", default=TASKS, help="Tasks to analyze")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    all_results = {}
    
    # Optionally analyze base model first
    if args.include_base:
        print(f"\n{'='*60}")
        print(f"Loading BASE MODEL: {BASE_MODEL}")
        print(f"{'='*60}")
        
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model.eval()
        
        for task in args.tasks:
            print(f"\nAnalyzing {task} with BASE model...")
            df = analyze_model_on_task(model, tokenizer, task, device, args.disc_shots)
            all_results[f"base_{task}"] = df
            
            # Print summary
            pos_df = df[df['gold'] == 'yes']
            neg_df = df[df['gold'] == 'no']
            
            print(f"\n  Summary for {task} (BASE):")
            print(f"    Positive examples (gold=yes): n={len(pos_df)}")
            print(f"      Mean P(Yes): {pos_df['total_yes'].mean():.4f} ± {pos_df['total_yes'].std():.4f}")
            print(f"      Mean P(No):  {pos_df['total_no'].mean():.4f} ± {pos_df['total_no'].std():.4f}")
            print(f"      Mean log-odds: {pos_df['log_odds'].mean():.4f}")
            print(f"    Negative examples (gold=no): n={len(neg_df)}")
            print(f"      Mean P(Yes): {neg_df['total_yes'].mean():.4f} ± {neg_df['total_yes'].std():.4f}")
            print(f"      Mean P(No):  {neg_df['total_no'].mean():.4f} ± {neg_df['total_no'].std():.4f}")
            print(f"      Mean log-odds: {neg_df['log_odds'].mean():.4f}")
            
            # Accuracy
            df['pred'] = (df['log_odds'] > 0).map({True: 'yes', False: 'no'})
            acc = (df['pred'] == df['gold']).mean()
            print(f"    Accuracy: {acc:.4f}")
        
        # Free memory
        del model
        torch.cuda.empty_cache()
    
    # Analyze V2G trained models
    for task in args.tasks:
        model_path = MODEL_TEMPLATE.format(task=task)
        
        print(f"\n{'='*60}")
        print(f"Loading V2G model for {task}")
        print(f"Path: {model_path}")
        print(f"{'='*60}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model.eval()
        
        print(f"\nAnalyzing {task} with V2G model...")
        df = analyze_model_on_task(model, tokenizer, task, device, args.disc_shots)
        all_results[f"v2g_{task}"] = df
        
        # Print summary
        pos_df = df[df['gold'] == 'yes']
        neg_df = df[df['gold'] == 'no']
        
        print(f"\n  Summary for {task} (V2G):")
        print(f"    Positive examples (gold=yes): n={len(pos_df)}")
        print(f"      Mean P(Yes): {pos_df['total_yes'].mean():.4f} ± {pos_df['total_yes'].std():.4f}")
        print(f"      Mean P(No):  {pos_df['total_no'].mean():.4f} ± {pos_df['total_no'].std():.4f}")
        print(f"      Mean remaining: {pos_df['remaining'].mean():.4f}")
        print(f"      Mean log-odds: {pos_df['log_odds'].mean():.4f}")
        print(f"    Negative examples (gold=no): n={len(neg_df)}")
        print(f"      Mean P(Yes): {neg_df['total_yes'].mean():.4f} ± {neg_df['total_yes'].std():.4f}")
        print(f"      Mean P(No):  {neg_df['total_no'].mean():.4f} ± {neg_df['total_no'].std():.4f}")
        print(f"      Mean remaining: {neg_df['remaining'].mean():.4f}")
        print(f"      Mean log-odds: {neg_df['log_odds'].mean():.4f}")
        
        # Accuracy
        df['pred'] = (df['log_odds'] > 0).map({True: 'yes', False: 'no'})
        acc = (df['pred'] == df['gold']).mean()
        print(f"    Accuracy: {acc:.4f}")
        
        # Per-token breakdown
        print(f"\n    Per-token breakdown (positive examples):")
        for w in YES_WORDS:
            col = f'p_{w}'
            print(f"      P('{w}'): {pos_df[col].mean():.6f}")
        for w in NO_WORDS:
            col = f'p_{w}'
            print(f"      P('{w}'): {pos_df[col].mean():.6f}")
        
        # Free memory
        del model
        torch.cuda.empty_cache()
    
    # Save all results
    print(f"\n{'='*60}")
    print("Saving results...")
    print(f"{'='*60}")
    
    for key, df in all_results.items():
        output_path = f"/datastor1/jdr/gv-gap/rankalign/outputs/yes_no_dist_{key}.csv"
        df.to_csv(output_path, index=False)
        print(f"  Saved: {output_path}")
    
    # Create summary table
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    
    summary_rows = []
    for task in args.tasks:
        key = f"v2g_{task}"
        if key in all_results:
            df = all_results[key]
            pos_df = df[df['gold'] == 'yes']
            neg_df = df[df['gold'] == 'no']
            
            df['pred'] = (df['log_odds'] > 0).map({True: 'yes', False: 'no'})
            acc = (df['pred'] == df['gold']).mean()
            
            summary_rows.append({
                'task': task,
                'accuracy': acc,
                'pos_mean_p_yes': pos_df['total_yes'].mean(),
                'pos_mean_p_no': pos_df['total_no'].mean(),
                'pos_mean_remaining': pos_df['remaining'].mean(),
                'pos_mean_logodds': pos_df['log_odds'].mean(),
                'neg_mean_p_yes': neg_df['total_yes'].mean(),
                'neg_mean_p_no': neg_df['total_no'].mean(),
                'neg_mean_remaining': neg_df['remaining'].mean(),
                'neg_mean_logodds': neg_df['log_odds'].mean(),
            })
    
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    
    summary_path = "/datastor1/jdr/gv-gap/rankalign/outputs/yes_no_dist_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
