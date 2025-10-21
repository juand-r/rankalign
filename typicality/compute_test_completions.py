"""
Compute log probabilities for test completions
Uses the same prompt as the main experiments: "Complete the sentence: cars are a kind of"
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Configuration
MODEL_NAME = "google/gemma-2-2b"
NOUN1 = "cars"
TEST_COMPLETIONS_FILE = "test_completions.txt"
OUTPUT_CSV = "hypernym_predictions/cars_test_completions.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(f"\nLoading Gemma-2-2b model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)
model.eval()
print(f"Model loaded on: {model.device}")

# Load test completions
print(f"\nLoading test completions from {TEST_COMPLETIONS_FILE}...")
completions = []
current_category = None

with open(TEST_COMPLETIONS_FILE, 'r') as f:
    for line in f:
        line = line.strip()
        # Skip empty lines
        if not line:
            continue
        # Category headers start with #
        if line.startswith('#'):
            current_category = line[1:].strip()
            continue
        # Regular completion
        completions.append({
            'completion': line,
            'category': current_category if current_category else 'Uncategorized'
        })

print(f"  ✓ Loaded {len(completions)} test completions")

# Compute log probabilities
prompt = f"Complete the sentence: {NOUN1} are a kind of"
print(f"\nPrompt: '{prompt}'")
print(f"\nComputing log probabilities for each completion...")

results = []

for item in tqdm(completions, desc="Processing completions"):
    completion_text = item['completion']
    
    # Construct full text
    full_text = prompt + " " + completion_text
    
    # Tokenize
    prompt_tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
    full_tokens = tokenizer(full_text, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
    
    # The completion tokens are everything after the prompt
    prompt_length = prompt_tokens.shape[1]
    completion_tokens = full_tokens[:, prompt_length:]
    
    # Compute log probability of the completion given the prompt
    with torch.no_grad():
        outputs = model(full_tokens)
        logits = outputs.logits
        
        # Get log probabilities for each position
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Sum log probabilities for the completion tokens
        total_log_prob = 0.0
        for i in range(completion_tokens.shape[1]):
            token_id = completion_tokens[0, i].item()
            # The logits at position i predict token at position i+1
            # So logits[0, prompt_length + i - 1] predicts token at prompt_length + i
            position = prompt_length + i - 1
            token_log_prob = log_probs[0, position, token_id].item()
            total_log_prob += token_log_prob
    
    results.append({
        'noun1': NOUN1,
        'completion': completion_text,
        'category': item['category'],
        'log_prob': total_log_prob,
        'probability': np.exp(total_log_prob),
        'num_tokens': completion_tokens.shape[1]
    })

# Create DataFrame and sort by log_prob (descending)
df = pd.DataFrame(results)
df = df.sort_values('log_prob', ascending=False).reset_index(drop=True)

# Save to CSV
output_path = Path(OUTPUT_CSV)
df.to_csv(output_path, index=False)

print(f"\n{'='*60}")
print(f"✓ Saved {len(df)} test completions to:")
print(f"  {output_path}")
print(f"{'='*60}")

# Print results by category
print(f"\nResults by category:")
print(f"{'Rank':<6} {'Category':<35} {'Completion':<45} {'Log Prob':<12}")
print("-" * 100)
for i, row in df.iterrows():
    print(f"{i+1:<6} {row['category']:<35} {row['completion']:<45} {row['log_prob']:<12.4f}")

# Print summary statistics by category
print(f"\n{'='*60}")
print(f"Summary by category:")
print(f"{'='*60}")
for category in df['category'].unique():
    cat_df = df[df['category'] == category]
    print(f"\n{category}:")
    print(f"  Count: {len(cat_df)}")
    print(f"  Mean log_prob: {cat_df['log_prob'].mean():.4f}")
    print(f"  Best: {cat_df.iloc[0]['completion'][:50]} ({cat_df.iloc[0]['log_prob']:.4f})")
    print(f"  Worst: {cat_df.iloc[-1]['completion'][:50]} ({cat_df.iloc[-1]['log_prob']:.4f})")

print(f"\n✅ DONE!")

