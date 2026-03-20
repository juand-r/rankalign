#!/usr/bin/env python3
"""
Pipeline to extract and process points with log_prob in [-12, 0] range.

Steps:
1. Filter cars_combined_tok1_tok2.csv for log_prob in [-12, 0]
2. Add GPT-4 ground truth labels
3. Add GPT-2 typicality scores (log_prob_noun2, log_wordfreq_noun2, etc.)
4. Save as cars_midrange_logprob_with_gpt4_gt_and_typicality.csv

Usage:
    python pipeline_midrange_logprob.py [--skip-gpt4] [--skip-typicality]
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import os

# Import functions from existing scripts
from add_gpt4_ground_truth import get_gpt4_answer
from add_gpt2_typicality_to_predictions import (
    compute_gpt2_log_prob,
    compute_gpt2_log_prob_given_context,
    get_word_frequency
)


def step1_filter_data(input_file, output_file, min_logprob=-12, max_logprob=0):
    """Filter data to only include points with log_prob in specified range."""
    print(f"\n{'='*60}")
    print(f"STEP 1: Filtering data for log_prob in [{min_logprob}, {max_logprob}]")
    print(f"{'='*60}")
    
    df = pd.read_csv(input_file)
    print(f"  Input: {len(df)} rows")
    
    df_filtered = df[(df['log_prob'] >= min_logprob) & (df['log_prob'] <= max_logprob)]
    print(f"  Filtered: {len(df_filtered)} rows")
    
    # Save filtered data
    df_filtered.to_csv(output_file, index=False)
    print(f"  Saved to: {output_file}")
    
    return df_filtered


def step2_add_gpt4_labels(input_file, output_file):
    """Add GPT-4 ground truth labels."""
    print(f"\n{'='*60}")
    print(f"STEP 2: Adding GPT-4 ground truth labels")
    print(f"{'='*60}")
    
    from openai import OpenAI
    
    client = OpenAI()  # Uses OPENAI_API_KEY env var
    
    df = pd.read_csv(input_file)
    print(f"  Input: {len(df)} rows")
    
    # Check if already has gpt4_ground_truth column
    if 'gpt4_ground_truth' in df.columns:
        existing = df['gpt4_ground_truth'].notna().sum()
        print(f"  Already has {existing} GPT-4 labels")
        if existing == len(df):
            print("  Skipping - all labels already present")
            df.to_csv(output_file, index=False)
            return df
    else:
        df['gpt4_ground_truth'] = None
    
    # Process rows without labels
    for idx in tqdm(df.index, desc="Getting GPT-4 labels"):
        if pd.notna(df.loc[idx, 'gpt4_ground_truth']):
            continue
        
        noun1 = df.loc[idx, 'noun1']
        predicted_hypernym = df.loc[idx, 'predicted_hypernym']
        
        answer = get_gpt4_answer(noun1, predicted_hypernym, client)
        df.loc[idx, 'gpt4_ground_truth'] = answer
        
        # Save periodically
        if idx % 50 == 0:
            df.to_csv(output_file, index=False)
    
    # Final save
    df.to_csv(output_file, index=False)
    print(f"  Saved to: {output_file}")
    
    # Summary
    yes_count = (df['gpt4_ground_truth'] == 'Yes').sum()
    no_count = (df['gpt4_ground_truth'] == 'No').sum()
    print(f"  Labels: Yes={yes_count}, No={no_count}")
    
    return df


def step3_add_typicality(input_file, output_file):
    """Add GPT-2 typicality scores and word frequency."""
    print(f"\n{'='*60}")
    print(f"STEP 3: Adding typicality scores")
    print(f"{'='*60}")
    
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    df = pd.read_csv(input_file)
    print(f"  Input: {len(df)} rows")
    
    # Check which columns already exist
    cols_to_add = []
    for col in ['log_prob_noun2', 'log_prob_noun1', 'log_prob_noun2_given_context', 
                'log_wordfreq_noun2', 'log_wordfreq_noun1']:
        if col not in df.columns or df[col].isna().any():
            cols_to_add.append(col)
    
    if not cols_to_add:
        print("  All typicality columns already present")
        df.to_csv(output_file, index=False)
        return df
    
    print(f"  Computing: {cols_to_add}")
    
    # Load GPT-2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    
    # Initialize columns
    for col in cols_to_add:
        if col not in df.columns:
            df[col] = np.nan
    
    # Process each row
    for idx in tqdm(df.index, desc="Computing typicality"):
        noun1 = df.loc[idx, 'noun1']
        noun2 = df.loc[idx, 'predicted_hypernym']
        
        # GPT-2 unconditional log prob of noun2
        if 'log_prob_noun2' in cols_to_add and pd.isna(df.loc[idx, 'log_prob_noun2']):
            df.loc[idx, 'log_prob_noun2'] = compute_gpt2_log_prob(noun2, model, tokenizer, device)
        
        # GPT-2 unconditional log prob of noun1
        if 'log_prob_noun1' in cols_to_add and pd.isna(df.loc[idx, 'log_prob_noun1']):
            df.loc[idx, 'log_prob_noun1'] = compute_gpt2_log_prob(noun1, model, tokenizer, device)
        
        # GPT-2 conditional log prob P(noun2 | "noun1 are a kind of")
        if 'log_prob_noun2_given_context' in cols_to_add and pd.isna(df.loc[idx, 'log_prob_noun2_given_context']):
            context = f"{noun1} are a kind of"
            df.loc[idx, 'log_prob_noun2_given_context'] = compute_gpt2_log_prob_given_context(
                context, noun2, model, tokenizer, device
            )
        
        # Word frequency
        if 'log_wordfreq_noun2' in cols_to_add and pd.isna(df.loc[idx, 'log_wordfreq_noun2']):
            df.loc[idx, 'log_wordfreq_noun2'] = np.log(get_word_frequency(noun2) + 1e-12)
        
        if 'log_wordfreq_noun1' in cols_to_add and pd.isna(df.loc[idx, 'log_wordfreq_noun1']):
            df.loc[idx, 'log_wordfreq_noun1'] = np.log(get_word_frequency(noun1) + 1e-12)
        
        # Save periodically
        if idx % 100 == 0:
            df.to_csv(output_file, index=False)
    
    # Final save
    df.to_csv(output_file, index=False)
    print(f"  Saved to: {output_file}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Pipeline for mid-range log_prob points')
    parser.add_argument('--min-logprob', type=float, default=-12, help='Minimum log_prob (default: -12)')
    parser.add_argument('--max-logprob', type=float, default=0, help='Maximum log_prob (default: 0)')
    parser.add_argument('--skip-gpt4', action='store_true', help='Skip GPT-4 labeling')
    parser.add_argument('--skip-typicality', action='store_true', help='Skip typicality computation')
    args = parser.parse_args()
    
    # File paths
    input_file = Path('cars_combined_tok1_tok2.csv')
    filtered_file = Path('cars_midrange_logprob_filtered.csv')
    gpt4_file = Path('cars_midrange_logprob_with_gpt4_gt.csv')
    output_file = Path('cars_midrange_logprob_with_gpt4_gt_and_typicality.csv')
    
    print("\n" + "="*60)
    print("PIPELINE: Mid-range log_prob extraction and processing")
    print("="*60)
    print(f"  Range: [{args.min_logprob}, {args.max_logprob}]")
    print(f"  Input: {input_file}")
    print(f"  Output: {output_file}")
    
    # Step 1: Filter
    df = step1_filter_data(input_file, filtered_file, args.min_logprob, args.max_logprob)
    
    # Step 2: GPT-4 labels
    if not args.skip_gpt4:
        df = step2_add_gpt4_labels(filtered_file, gpt4_file)
    else:
        print("\n[SKIPPED] Step 2: GPT-4 labeling")
        gpt4_file = filtered_file
    
    # Step 3: Typicality
    if not args.skip_typicality:
        df = step3_add_typicality(gpt4_file, output_file)
    else:
        print("\n[SKIPPED] Step 3: Typicality computation")
        import shutil
        shutil.copy(gpt4_file, output_file)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"  Output: {output_file}")
    print(f"  Rows: {len(pd.read_csv(output_file))}")
    
    # Show summary
    df_final = pd.read_csv(output_file)
    if 'gpt4_ground_truth' in df_final.columns:
        print(f"  GPT-4 labels: Yes={( df_final['gpt4_ground_truth']=='Yes').sum()}, No={(df_final['gpt4_ground_truth']=='No').sum()}")
    
    print("\nNext step: Run combine_midrange_with_existing.py to merge with existing data")


if __name__ == '__main__':
    main()

