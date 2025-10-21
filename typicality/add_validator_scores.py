"""
Add validator log probability scores to the combined predictions CSV.

Uses the validator prompt from src/utils.py (few-shot, variation==0):
"Do you think bees are furniture? Answer: No

Do you think corgis are dogs? Answer: Yes

Do you think trucks are a fruit? Answer: No

Do you think robins are birds? Answer: Yes

Do you think {noun1} are a {predicted_hypernym}? Answer:"

Computes log P(Yes) for each prediction.
"""

import sys
sys.path.append('/datastor1/jdr/gv-gap/rankalign/src')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from string import Template

# Configuration
MODEL_NAME = "google/gemma-2-2b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_CSV = Path("hypernym_predictions/cars_combined_clean_predictions.csv")
OUTPUT_CSV = Path("hypernym_predictions/cars_combined_clean_predictions_with_validator.csv")

def get_validator_log_prob(prompt, model, tokenizer, device, yes_toks):
    """
    Compute validator log probability: log(sum of probabilities for all "yes" tokens)
    
    This matches the eval.py implementation:
        logprobs_disc = torch.log(torch.sum(P_disc[..., yestoks], dim=-1))
    
    Args:
        prompt: The validator prompt (without completion)
        model: The model
        tokenizer: The tokenizer
        device: Device to run on
        yes_toks: List of token IDs for different "yes" variations
    
    Returns:
        log(P("Yes") + P(" Yes") + P("YES") + P("yes") + P(" yes"))
    """
    with torch.no_grad():
        # Tokenize prompt
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        
        # Get model outputs
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]  # Last position logits
        
        # Get probabilities (not log probs!)
        probs = torch.softmax(logits, dim=-1)
        
        # Sum probabilities for all "yes" tokens
        yes_prob_sum = probs[yes_toks].sum().item()
        
        # Return log of the sum
        return np.log(yes_prob_sum)

if __name__ == '__main__':
    print("="*60)
    print("Adding Validator Scores to Combined Predictions CSV")
    print("="*60)
    
    # Load the CSV
    print(f"\nLoading CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"  ✓ Loaded {len(df)} predictions")
    
    # Load model and tokenizer
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    print(f"  ✓ Model loaded on {DEVICE}")
    
    # Define "yes" tokens (from eval.py line 31)
    yes_words = ["Yes", " Yes", "YES", "yes", " yes"]
    yes_toks = [tokenizer.encode(word)[-1] for word in yes_words]
    print(f"\nYes tokens:")
    for word, tok_id in zip(yes_words, yes_toks):
        print(f"  '{word}' -> token {tok_id}")
    
    # Validator prompt template from utils.py (few-shot, variation==0, line 450-452)
    validator_template = Template("Do you think bees are furniture? Answer: No\n\nDo you think corgis are dogs? Answer: Yes\n\nDo you think trucks are a fruit? Answer: No\n\nDo you think robins are birds? Answer: Yes\n\nDo you think $word are a $hypernym? Answer:")
    
    print(f"\nComputing validator log P(Yes) for {len(df)} predictions...")
    print(f"  Method: log(sum of probabilities for yes tokens)")
    
    validator_log_probs = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing predictions"):
        noun1 = row['noun1']
        predicted_hypernym = row['predicted_hypernym']
        
        # Create validator prompt (WITHOUT the answer)
        prompt = validator_template.substitute(
            word=noun1,
            hypernym=predicted_hypernym
        ).strip()
        
        # Compute log P(Yes) = log(sum of probabilities for all "yes" tokens)
        log_prob = get_validator_log_prob(prompt, model, tokenizer, DEVICE, yes_toks)
        validator_log_probs.append(log_prob)
    
    # Add to dataframe
    df['validator_log_prob'] = validator_log_probs
    
    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Added validator_log_prob column")
    print(f"✓ Saved to: {OUTPUT_CSV}")
    print(f"{'='*60}")
    
    # Print statistics
    print(f"\nValidator Log Prob Statistics:")
    print(f"  Min:  {df['validator_log_prob'].min():.4f}")
    print(f"  Max:  {df['validator_log_prob'].max():.4f}")
    print(f"  Mean: {df['validator_log_prob'].mean():.4f}")
    print(f"  Median: {df['validator_log_prob'].median():.4f}")
    
    # Print top 20 by validator score
    print(f"\nTop 20 predictions by validator log P(Yes):")
    print(f"{'Rank':<6} {'Hypernym':<30} {'Gen LogProb':<12} {'Val LogProb':<12}")
    print("-" * 70)
    df_sorted = df.sort_values('validator_log_prob', ascending=False)
    for rank, (_, row) in enumerate(df_sorted.head(20).iterrows(), 1):
        print(f"{rank:<6} {row['predicted_hypernym']:<30} {row['log_prob']:<12.4f} {row['validator_log_prob']:<12.4f}")
    
    # Print top 20 by generator score for comparison
    print(f"\nTop 20 predictions by generator log prob (for comparison):")
    print(f"{'Rank':<6} {'Hypernym':<30} {'Gen LogProb':<12} {'Val LogProb':<12}")
    print("-" * 70)
    df_sorted_gen = df.sort_values('log_prob', ascending=False)
    for rank, (_, row) in enumerate(df_sorted_gen.head(20).iterrows(), 1):
        print(f"{rank:<6} {row['predicted_hypernym']:<30} {row['log_prob']:<12.4f} {row['validator_log_prob']:<12.4f}")
    
    print(f"\n✅ DONE!")

