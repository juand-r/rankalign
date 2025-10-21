"""
Get ALL noun predictions for noun1="cars"
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import spacy

# Configuration
MODEL_NAME = "google/gemma-2-2b"
NOUN1 = "cars"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("hypernym_predictions")

print(f"Device: {DEVICE}")
print(f"Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"\nLoading Gemma-2-2b model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)
model.eval()

print(f"Model loaded on: {model.device}")
print(f"Vocabulary size: {len(tokenizer)}")


def get_word_boundary_tokens(tokenizer):
    """Get all token IDs that start with ▁ (space/word boundary in SentencePiece)"""
    word_start_tokens = []
    for token_id in range(len(tokenizer)):
        try:
            token_str = tokenizer.convert_ids_to_tokens([token_id])[0]
            if token_str.startswith('▁'):
                word_start_tokens.append(token_id)
        except:
            continue
    return word_start_tokens


def is_valid_noun(word, nlp):
    """Check if word is a valid noun using spaCy"""
    word = word.strip()
    if not word or len(word) < 2:
        return False
    if not word.replace('-', '').replace("'", "").isalpha():
        return False
    
    doc = nlp(word)
    if len(doc) == 0:
        return False
    
    # Check if first token is a noun
    return doc[0].pos_ in ['NOUN', 'PROPN']


# Get word boundary tokens (precompute once)
print("\nIdentifying word-boundary tokens...")
word_start_tokens = get_word_boundary_tokens(tokenizer)
print(f"Found {len(word_start_tokens)} tokens starting with ▁ (out of {len(tokenizer)} total)")

# Construct prompt
prompt = f"Complete the sentence: {NOUN1} are a kind of"
print(f"\nPrompt: '{prompt}'")

# Tokenize and get logits
print("Getting model predictions...")
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]  # Last token logits

# Get log probabilities for word-starting tokens only
print("Computing log probabilities for word-starting tokens...")
word_start_logits = logits[word_start_tokens]
log_probs = torch.log_softmax(word_start_logits, dim=0)

# Sort by log probability (descending)
sorted_indices = torch.argsort(log_probs, descending=True)

print(f"Filtering for valid nouns...")
all_noun_predictions = []

for rank, idx in enumerate(tqdm(sorted_indices, desc="Processing tokens"), 1):
    token_id = word_start_tokens[idx.item()]
    word = tokenizer.decode([token_id]).strip()
    log_prob = log_probs[idx].item()
    prob = np.exp(log_prob)
    
    if is_valid_noun(word, nlp):
        all_noun_predictions.append({
            'noun1': NOUN1,
            'predicted_hypernym': word,
            'rank': len(all_noun_predictions) + 1,  # Rank among valid nouns
            'rank_overall': rank,  # Rank among all word-starting tokens
            'log_prob': log_prob,
            'probability': prob,
            'token_id': token_id
        })

# Create DataFrame and save
df = pd.DataFrame(all_noun_predictions)
output_csv = OUTPUT_DIR / f"{NOUN1}_all_noun_predictions.csv"
df.to_csv(output_csv, index=False)

print(f"\n{'='*60}")
print(f"✓ Saved {len(df)} noun predictions to {output_csv}")
print(f"{'='*60}")

# Print summary statistics
print(f"\nSummary:")
print(f"  Total valid nouns found: {len(df)}")
print(f"  Processed {len(sorted_indices)} word-starting tokens")
print(f"  Noun filtering rate: {100 * len(df) / len(sorted_indices):.1f}%")

# Print top 20
print(f"\nTop 20 predicted hypernyms for '{NOUN1}':")
print(f"{'Rank':<6} {'Word':<25} {'Log Prob':<12} {'Probability':<12}")
print("-" * 60)
for _, row in df.head(20).iterrows():
    print(f"{row['rank']:<6} {row['predicted_hypernym']:<25} {row['log_prob']:<12.4f} {row['probability']:<12.6f}")

# Print bottom 20
print(f"\nBottom 20 predicted hypernyms for '{NOUN1}':")
print(f"{'Rank':<6} {'Word':<25} {'Log Prob':<12} {'Probability':<12}")
print("-" * 60)
for _, row in df.tail(20).iterrows():
    print(f"{row['rank']:<6} {row['predicted_hypernym']:<25} {row['log_prob']:<12.4f} {row['probability']:<12.6f}")

print(f"\n✅ DONE!")

