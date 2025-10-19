"""
Compute GPT-2 typicality scores for hypernymy task.

This script computes:
1. P_gpt2(noun2) - unconditional probability of the hypernym
2. P_gpt2(noun1) - unconditional probability of the hyponym  
3. P_gpt2(noun2 | "noun1 is a kind of") - conditional probability

Usage:
python compute_gpt2_typicality.py --seed 0 --split_type random --output typicality_scores.csv
"""

import os
import sys
import argparse
import torch
import csv
from tqdm import tqdm
import numpy as np

# Add parent src directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(parent_dir, "src")
sys.path.append(src_path)

from utils import load_noun_pair_data, split_train_test, split_train_test_no_overlap, split_train_test_no_overlap_both
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def compute_token_probability(text, model, tokenizer, device):
    """
    Compute the probability of a text under the model.
    For multi-token text, returns the joint probability (product of token probs).
    
    Returns log probability.
    """
    with torch.no_grad():
        # Tokenize without special tokens to get just the content tokens
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        
        if len(input_ids) == 0:
            return float('-inf')
        
        # For a single token, compute P(token)
        if len(input_ids) == 1:
            # Use empty context (just BOS if model has it)
            context_ids = tokenizer.encode("", add_special_tokens=True)
            full_ids = context_ids + input_ids
            
            input_tensor = torch.tensor([full_ids]).to(device)
            outputs = model(input_tensor)
            logits = outputs.logits
            
            # Get probability of the target token
            # logits shape: [batch_size, seq_len, vocab_size]
            # We want P(input_ids[0]) given context
            target_logits = logits[0, len(context_ids) - 1, :]
            probs = torch.softmax(target_logits, dim=-1)
            token_prob = probs[input_ids[0]].item()
            
            return np.log(token_prob + 1e-12)
        
        # For multi-token text, compute product of conditional probabilities
        # P(t1, t2, t3) = P(t1) * P(t2|t1) * P(t3|t1,t2)
        log_prob_sum = 0.0
        
        for i in range(len(input_ids)):
            # Context is all tokens before position i
            if i == 0:
                context_ids = tokenizer.encode("", add_special_tokens=True)
            else:
                context_ids = tokenizer.encode("", add_special_tokens=True)[:-1] + input_ids[:i]
            
            full_ids = context_ids + [input_ids[i]]
            input_tensor = torch.tensor([full_ids]).to(device)
            outputs = model(input_tensor)
            logits = outputs.logits
            
            # Get probability of token i given context
            target_logits = logits[0, len(context_ids) - 1, :]
            probs = torch.softmax(target_logits, dim=-1)
            token_prob = probs[input_ids[i]].item()
            
            log_prob_sum += np.log(token_prob + 1e-12)
        
        return log_prob_sum


def compute_conditional_probability(target, context, model, tokenizer, device):
    """
    Compute P(target | context) under the model.
    
    Args:
        target: The text to compute probability for (e.g., "dog")
        context: The conditioning context (e.g., "A corgi is a kind of")
        
    Returns log probability.
    """
    with torch.no_grad():
        # Tokenize context and target separately
        context_ids = tokenizer.encode(context, add_special_tokens=True)
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        
        if len(target_ids) == 0:
            return float('-inf')
        
        # Compute P(target | context) = product of P(target_token_i | context + target_tokens[:i])
        log_prob_sum = 0.0
        
        for i in range(len(target_ids)):
            # Full input is context + target tokens up to and including position i
            full_ids = context_ids + target_ids[:i+1]
            input_tensor = torch.tensor([full_ids]).to(device)
            
            outputs = model(input_tensor)
            logits = outputs.logits
            
            # Get probability of target_ids[i] given everything before it
            target_position = len(context_ids) + i - 1
            target_logits = logits[0, target_position, :]
            probs = torch.softmax(target_logits, dim=-1)
            token_prob = probs[target_ids[i]].item()
            
            log_prob_sum += np.log(token_prob + 1e-12)
        
        return log_prob_sum


def main(args):
    print("="*60)
    print("Computing GPT-2 Typicality Scores for Hypernymy Task")
    print("="*60)
    
    # Load data with same split as eval.py would use
    print(f"\nLoading data (seed={args.seed}, split_type={args.split_type})...")
    L = load_noun_pair_data()
    
    if args.split_type == 'hyper':
        L_train, L_test = split_train_test_no_overlap(L, seed=args.seed)
    elif args.split_type == 'random':
        L_train, L_test = split_train_test(L, seed=args.seed, subsample=False, num_train=3000)
    elif args.split_type == 'both':
        L_train, L_test = split_train_test_no_overlap_both(L, seed=2)
    else:
        raise ValueError(f"Unknown split_type: {args.split_type}")
    
    # Use test set (same as eval.py default)
    LL = L_test if not args.train else L_train
    print(f"Using {'train' if args.train else 'test'} set: {len(LL)} examples")
    
    # Load GPT-2
    device = get_device()
    print(f"\nLoading GPT-2 on device: {device}")
    model_name = "gpt2"  # Use base GPT-2
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    print(f"Loaded {model_name}")
    
    # Compute typicality scores
    print("\nComputing typicality scores...")
    results = []
    
    for idx, item in enumerate(tqdm(LL)):
        noun1 = item.noun1  # hyponym (e.g., "corgi")
        noun2 = item.noun2  # hypernym (e.g., "dog")
        taxonomic = item.taxonomic  # "yes" or "no"
        
        # Compute P(noun2) - unconditional probability of hypernym
        log_prob_noun2 = compute_token_probability(noun2, model, tokenizer, device)
        
        # Compute P(noun1) - unconditional probability of hyponym
        log_prob_noun1 = compute_token_probability(noun1, model, tokenizer, device)
        
        # Compute P(noun2 | "noun1 is a kind of") - conditional probability
        context = f"{noun1} is a kind of"
        log_prob_noun2_given_context = compute_conditional_probability(
            noun2, context, model, tokenizer, device
        )
        
        # Store results
        results.append({
            'index': idx,
            'noun1': noun1,
            'noun2': noun2,
            'taxonomic': taxonomic,
            'ground_truth': 1 if taxonomic.strip().lower() == 'yes' else 0,
            'log_prob_noun2': log_prob_noun2,
            'log_prob_noun1': log_prob_noun1,
            'log_prob_noun2_given_context': log_prob_noun2_given_context
        })
    
    # Save to CSV
    output_file = args.output
    print(f"\nSaving results to: {output_file}")
    
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['index', 'noun1', 'noun2', 'taxonomic', 'ground_truth',
                      'log_prob_noun2', 'log_prob_noun1', 'log_prob_noun2_given_context']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Saved {len(results)} examples")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics (log probabilities)")
    print("="*60)
    
    log_probs_noun2 = [r['log_prob_noun2'] for r in results]
    log_probs_noun1 = [r['log_prob_noun1'] for r in results]
    log_probs_conditional = [r['log_prob_noun2_given_context'] for r in results]
    
    print(f"\nP(noun2) - unconditional hypernym:")
    print(f"  Mean: {np.mean(log_probs_noun2):.4f}")
    print(f"  Std:  {np.std(log_probs_noun2):.4f}")
    print(f"  Min:  {np.min(log_probs_noun2):.4f}")
    print(f"  Max:  {np.max(log_probs_noun2):.4f}")
    
    print(f"\nP(noun1) - unconditional hyponym:")
    print(f"  Mean: {np.mean(log_probs_noun1):.4f}")
    print(f"  Std:  {np.std(log_probs_noun1):.4f}")
    print(f"  Min:  {np.min(log_probs_noun1):.4f}")
    print(f"  Max:  {np.max(log_probs_noun1):.4f}")
    
    print(f"\nP(noun2 | context) - conditional hypernym:")
    print(f"  Mean: {np.mean(log_probs_conditional):.4f}")
    print(f"  Std:  {np.std(log_probs_conditional):.4f}")
    print(f"  Min:  {np.min(log_probs_conditional):.4f}")
    print(f"  Max:  {np.max(log_probs_conditional):.4f}")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute GPT-2 typicality scores for hypernymy task")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for data split")
    parser.add_argument("--split_type", type=str, default='random', 
                        choices=['random', 'hyper', 'both'],
                        help="Type of train/test split")
    parser.add_argument("--train", action="store_true", default=False,
                        help="Use train set instead of test set")
    parser.add_argument("--output", type=str, default="gpt2_typicality_scores.csv",
                        help="Output CSV file path")
    
    args = parser.parse_args()
    main(args)

