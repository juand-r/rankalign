"""
Add GPT-2 typicality scores and word frequencies to predictions CSV.

This script adds:
1. log_prob_noun2 - P_gpt2(hypernym) unconditional probability
2. log_prob_noun1 - P_gpt2(hyponym) unconditional probability  
3. log_prob_noun2_given_context - P_gpt2(hypernym | "hyponym is a kind of")
4. log_wordfreq_noun2 - log normalized frequency of hypernym from wordfreq
5. log_wordfreq_noun1 - log normalized frequency of hyponym from wordfreq

Usage:
python add_gpt2_typicality_to_predictions.py \
    --input cars_combined_beam_and_handcrafted_with_gpt4_gt.csv \
    --output cars_combined_beam_and_handcrafted_with_typicality.csv
"""

import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from wordfreq import word_frequency

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
        context: The conditioning context (e.g., "cars are a kind of")
        
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


def get_word_frequency(word, lang='en'):
    """
    Get word frequency from wordfreq library.
    Returns log of corpus frequency.
    
    The frequency represents the proportion of times this word appears in a large corpus.
    For example, "the" has frequency ~0.054 (5.4% of all tokens).
    For rare/unknown words, assigns a very small frequency.
    """
    freq = word_frequency(word, lang, wordlist='large')
    if freq == 0:
        # Unknown word - assign very small frequency
        freq = 1e-8
    return np.log(freq)


def main(args):
    print("="*70)
    print("Adding GPT-2 Typicality Scores and Word Frequencies to Predictions")
    print("="*70)
    
    # Load the input CSV
    input_path = Path(args.input)
    print(f"\nLoading CSV: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  ✓ Loaded {len(df)} predictions")
    print(f"  ✓ Existing columns: {list(df.columns)}")
    
    # Check that required columns exist
    if 'noun1' not in df.columns or 'predicted_hypernym' not in df.columns:
        raise ValueError("CSV must have 'noun1' and 'predicted_hypernym' columns")
    
    # Load GPT-2
    device = get_device()
    print(f"\nLoading GPT-2 on device: {device}")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    print(f"  ✓ Loaded {model_name}")
    
    # Compute typicality scores for each prediction
    print(f"\nComputing GPT-2 typicality scores and word frequencies...")
    
    log_probs_noun2 = []
    log_probs_noun1 = []
    log_probs_conditional = []
    log_wordfreq_noun2 = []
    log_wordfreq_noun1 = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing predictions"):
        noun1 = row['noun1']
        noun2 = row['predicted_hypernym']
        
        # Compute P(noun2) - unconditional probability of hypernym
        log_prob_n2 = compute_token_probability(noun2, model, tokenizer, device)
        log_probs_noun2.append(log_prob_n2)
        
        # Compute P(noun1) - unconditional probability of hyponym
        log_prob_n1 = compute_token_probability(noun1, model, tokenizer, device)
        log_probs_noun1.append(log_prob_n1)
        
        # Compute P(noun2 | "noun1 are a kind of") - conditional probability
        # Use "are" since we're typically dealing with plurals like "cars"
        context = f"{noun1} are a kind of"
        log_prob_cond = compute_conditional_probability(noun2, context, model, tokenizer, device)
        log_probs_conditional.append(log_prob_cond)
        
        # Compute word frequencies
        log_wf_n2 = get_word_frequency(noun2)
        log_wordfreq_noun2.append(log_wf_n2)
        
        log_wf_n1 = get_word_frequency(noun1)
        log_wordfreq_noun1.append(log_wf_n1)
    
    # Add columns to dataframe
    df['log_prob_noun2'] = log_probs_noun2
    df['log_prob_noun1'] = log_probs_noun1
    df['log_prob_noun2_given_context'] = log_probs_conditional
    df['log_wordfreq_noun2'] = log_wordfreq_noun2
    df['log_wordfreq_noun1'] = log_wordfreq_noun1
    
    # Save to output CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"✓ Added typicality columns:")
    print(f"  - log_prob_noun2 (GPT-2 unconditional hypernym)")
    print(f"  - log_prob_noun1 (GPT-2 unconditional hyponym)")
    print(f"  - log_prob_noun2_given_context (GPT-2 conditional)")
    print(f"  - log_wordfreq_noun2 (corpus frequency hypernym)")
    print(f"  - log_wordfreq_noun1 (corpus frequency hyponym)")
    print(f"✓ Saved to: {output_path}")
    print(f"{'='*70}")
    
    # Print summary statistics
    print(f"\nSummary Statistics (log probabilities):")
    print(f"\nP(noun2) - GPT-2 unconditional hypernym:")
    print(f"  Mean: {np.mean(log_probs_noun2):.4f}")
    print(f"  Std:  {np.std(log_probs_noun2):.4f}")
    print(f"  Min:  {np.min(log_probs_noun2):.4f}")
    print(f"  Max:  {np.max(log_probs_noun2):.4f}")
    
    print(f"\nP(noun1) - GPT-2 unconditional hyponym:")
    print(f"  Mean: {np.mean(log_probs_noun1):.4f}")
    print(f"  Std:  {np.std(log_probs_noun1):.4f}")
    print(f"  Min:  {np.min(log_probs_noun1):.4f}")
    print(f"  Max:  {np.max(log_probs_noun1):.4f}")
    
    print(f"\nP(noun2 | context) - GPT-2 conditional hypernym:")
    print(f"  Mean: {np.mean(log_probs_conditional):.4f}")
    print(f"  Std:  {np.std(log_probs_conditional):.4f}")
    print(f"  Min:  {np.min(log_probs_conditional):.4f}")
    print(f"  Max:  {np.max(log_probs_conditional):.4f}")
    
    print(f"\nWordFreq(noun2) - corpus frequency hypernym:")
    print(f"  Mean: {np.mean(log_wordfreq_noun2):.4f}")
    print(f"  Std:  {np.std(log_wordfreq_noun2):.4f}")
    print(f"  Min:  {np.min(log_wordfreq_noun2):.4f}")
    print(f"  Max:  {np.max(log_wordfreq_noun2):.4f}")
    
    print(f"\nWordFreq(noun1) - corpus frequency hyponym:")
    print(f"  Mean: {np.mean(log_wordfreq_noun1):.4f}")
    print(f"  Std:  {np.std(log_wordfreq_noun1):.4f}")
    print(f"  Min:  {np.min(log_wordfreq_noun1):.4f}")
    print(f"  Max:  {np.max(log_wordfreq_noun1):.4f}")
    
    # Print top 10 predictions by different metrics
    print(f"\nTop 10 predictions by GPT-2 P(noun2 | context):")
    print(f"{'Rank':<6} {'Hypernym':<30} {'Conditional LogP':<16}")
    print("-" * 52)
    df_sorted = df.sort_values('log_prob_noun2_given_context', ascending=False)
    for rank, (_, row) in enumerate(df_sorted.head(10).iterrows(), 1):
        print(f"{rank:<6} {row['predicted_hypernym']:<30} {row['log_prob_noun2_given_context']:<16.4f}")
    
    print(f"\nTop 10 predictions by WordFreq(noun2):")
    print(f"{'Rank':<6} {'Hypernym':<30} {'WordFreq LogP':<16}")
    print("-" * 52)
    df_sorted = df.sort_values('log_wordfreq_noun2', ascending=False)
    for rank, (_, row) in enumerate(df_sorted.head(10).iterrows(), 1):
        print(f"{rank:<6} {row['predicted_hypernym']:<30} {row['log_wordfreq_noun2']:<16.4f}")
    
    print(f"\n✅ DONE!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add GPT-2 typicality scores and word frequencies to predictions CSV"
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV file path')
    
    args = parser.parse_args()
    main(args)

