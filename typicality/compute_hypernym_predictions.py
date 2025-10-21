"""
Phase 1: Compute predicted hypernyms for each noun1 using Gemma-2-2b

For each noun1, get the model's top-K predicted noun completions for:
"Complete the sentence: {noun1} are a kind of"

Filters:
1. Complete words only (tokens starting with ▁)
2. Valid nouns only (using spaCy POS tagging)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import spacy
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
MODEL_NAME = "google/gemma-2-2b"
TOP_K = 100  # Number of top predictions to save per noun1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_DATA_PATH = "merged_data_google-gemma-2-2b_random_test.csv"
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


def get_top_noun_predictions(noun1, model, tokenizer, word_start_tokens, nlp, top_k=100):
    """
    Get top-K noun predictions for: "{noun1} are a kind of"
    
    Returns:
        List of (word, log_prob, token_id) tuples
    """
    # Construct prompt (matching the format from utils.py)
    prompt = f"Complete the sentence: {noun1} are a kind of"
    
    # Tokenize and get logits
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Last token logits
    
    # Get log probabilities for word-starting tokens only
    word_start_logits = logits[word_start_tokens]
    log_probs = torch.log_softmax(word_start_logits, dim=0)
    
    # Get top-k indices among word-starting tokens
    top_k_expanded = min(top_k * 5, len(word_start_tokens))  # Get more initially for filtering
    top_k_idx = torch.topk(log_probs, k=top_k_expanded).indices
    
    # Decode and filter for valid nouns
    noun_candidates = []
    for idx in top_k_idx:
        token_id = word_start_tokens[idx.item()]
        word = tokenizer.decode([token_id]).strip()
        log_prob = log_probs[idx].item()
        
        if is_valid_noun(word, nlp):
            noun_candidates.append((word, log_prob, token_id))
        
        # Stop once we have enough valid nouns
        if len(noun_candidates) >= top_k:
            break
    
    return noun_candidates


# Load test data and extract unique noun1 values
print(f"\nLoading test data from {TEST_DATA_PATH}...")
df_test = pd.read_csv(TEST_DATA_PATH)
noun1_values = df_test['noun1'].unique()
print(f"Found {len(noun1_values)} unique noun1 values")

# Get word boundary tokens (precompute once)
print("\nIdentifying word-boundary tokens...")
word_start_tokens = get_word_boundary_tokens(tokenizer)
print(f"Found {len(word_start_tokens)} tokens starting with ▁ (out of {len(tokenizer)} total)")

# Process each noun1
print(f"\nComputing predictions for {len(noun1_values)} noun1 values...")
all_predictions = []

for noun1 in tqdm(noun1_values[:50], desc="Processing nouns"):  # Start with first 50 for testing
    try:
        predictions = get_top_noun_predictions(
            noun1, model, tokenizer, word_start_tokens, nlp, top_k=TOP_K
        )
        
        for rank, (word, log_prob, token_id) in enumerate(predictions, 1):
            all_predictions.append({
                'noun1': noun1,
                'predicted_hypernym': word,
                'rank': rank,
                'log_prob': log_prob,
                'probability': np.exp(log_prob),
                'token_id': token_id
            })
    except Exception as e:
        print(f"\nError processing '{noun1}': {e}")
        continue

# Save results
df_predictions = pd.DataFrame(all_predictions)
output_csv = OUTPUT_DIR / "gemma2_hypernym_predictions.csv"
df_predictions.to_csv(output_csv, index=False)
print(f"\n✓ Saved {len(df_predictions)} predictions to {output_csv}")

# Generate summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Total predictions: {len(df_predictions)}")
print(f"Unique noun1 values: {df_predictions['noun1'].nunique()}")
print(f"Unique predicted hypernyms: {df_predictions['predicted_hypernym'].nunique()}")
print(f"\nAverage predictions per noun1: {len(df_predictions) / df_predictions['noun1'].nunique():.1f}")

# Print some examples
print("\n" + "="*60)
print("EXAMPLE PREDICTIONS")
print("="*60)
for noun1 in noun1_values[:5]:
    subset = df_predictions[df_predictions['noun1'] == noun1].head(10)
    print(f"\n'{noun1}' → Top 10 predicted hypernyms:")
    for _, row in subset.iterrows():
        print(f"  {row['rank']:2d}. {row['predicted_hypernym']:20s} (log_prob={row['log_prob']:.3f}, prob={row['probability']:.4f})")

# Generate plots
print("\n" + "="*60)
print("GENERATING PLOTS")
print("="*60)

# Plot 1: Distribution of log probabilities
plt.figure(figsize=(10, 6))
plt.hist(df_predictions['log_prob'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Log Probability', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Distribution of Predicted Hypernym Log Probabilities', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plot1_path = OUTPUT_DIR / "log_prob_distribution.png"
plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved {plot1_path}")

# Plot 2: Top-20 most common predicted hypernyms
plt.figure(figsize=(12, 8))
top_hypernyms = df_predictions['predicted_hypernym'].value_counts().head(20)
plt.barh(range(len(top_hypernyms)), top_hypernyms.values, color='steelblue')
plt.yticks(range(len(top_hypernyms)), top_hypernyms.index, fontsize=10)
plt.xlabel('Frequency (across all noun1 values)', fontsize=12)
plt.title('Top 20 Most Frequently Predicted Hypernyms', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, axis='x', alpha=0.3)
plot2_path = OUTPUT_DIR / "top_predicted_hypernyms.png"
plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved {plot2_path}")

# Plot 3: Example visualizations for specific nouns
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Top Predicted Hypernyms for Selected Nouns', fontsize=16, fontweight='bold')

for idx, noun1 in enumerate(noun1_values[:4]):
    ax = axes[idx // 2, idx % 2]
    subset = df_predictions[df_predictions['noun1'] == noun1].head(15)
    
    ax.barh(range(len(subset)), subset['probability'].values, color='coral')
    ax.set_yticks(range(len(subset)))
    ax.set_yticklabels(subset['predicted_hypernym'].values, fontsize=9)
    ax.set_xlabel('Probability', fontsize=10)
    ax.set_title(f'"{noun1}" → predicted hypernyms', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plot3_path = OUTPUT_DIR / "example_predictions.png"
plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved {plot3_path}")

# Plot 4: Probability vs Rank (decay curve)
plt.figure(figsize=(10, 6))
for noun1 in noun1_values[:10]:
    subset = df_predictions[df_predictions['noun1'] == noun1]
    plt.plot(subset['rank'], subset['probability'], alpha=0.5, linewidth=1)

plt.xlabel('Rank', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.title('Probability Decay by Rank (first 10 nouns)', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plot4_path = OUTPUT_DIR / "probability_decay.png"
plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved {plot4_path}")

print("\n" + "="*60)
print("✅ DONE! All results saved to:", OUTPUT_DIR)
print("="*60)

