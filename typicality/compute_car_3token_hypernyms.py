"""
Get ALL 3-token noun phrase predictions for noun1="cars"
Uses top-K continuation strategy, checking all combinations
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
TOP_K_TOKEN1 = 500   # Top K tokens to continue from
TOP_M_TOKEN2 = 200   # For each token1, try top M token2 options
TOP_N_TOKEN3 = 100   # For each token1+token2, try top N token3 options
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("hypernym_predictions")

print(f"Device: {DEVICE}")
print(f"Strategy: Top-{TOP_K_TOKEN1} token1 × Top-{TOP_M_TOKEN2} token2 × Top-{TOP_N_TOKEN3} token3")
print(f"Total combinations: {TOP_K_TOKEN1} × {TOP_M_TOKEN2} × {TOP_N_TOKEN3} = {TOP_K_TOKEN1 * TOP_M_TOKEN2 * TOP_N_TOKEN3:,}")
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


def is_noun_phrase(text, nlp):
    """
    Check if text is a valid noun phrase
    - Last token is a noun (e.g., "motor vehicle")
    - Contains noun chunks
    """
    text = text.strip()
    if not text or len(text) < 2:
        return False
    
    # Basic filtering
    if not any(c.isalpha() for c in text):
        return False
    
    doc = nlp(text)
    if len(doc) == 0:
        return False
    
    # Check if last token is noun (most common pattern)
    if doc[-1].pos_ in ['NOUN', 'PROPN']:
        return True
    
    # Check for noun chunks
    if len(list(doc.noun_chunks)) > 0:
        return True
    
    return False


# Get word boundary tokens
print("\nIdentifying word-boundary tokens...")
word_start_tokens = get_word_boundary_tokens(tokenizer)
print(f"Found {len(word_start_tokens)} word-starting tokens")

# Step 1: Get top-K token1 options
prompt = f"Complete the sentence: {NOUN1} are a kind of"
print(f"\nPrompt: '{prompt}'")
print(f"\nStep 1: Getting top-{TOP_K_TOKEN1} first tokens...")

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)
    logits1 = outputs.logits[0, -1, :]

# Get top-K word-starting tokens
word_start_logits = logits1[word_start_tokens]
log_probs1 = torch.log_softmax(word_start_logits, dim=0)
top_k_idx = torch.topk(log_probs1, k=TOP_K_TOKEN1).indices

top_k_token1_ids = [word_start_tokens[idx.item()] for idx in top_k_idx]
top_k_log_probs1 = [log_probs1[idx].item() for idx in top_k_idx]

print(f"✓ Selected top-{len(top_k_token1_ids)} first tokens")

# Step 2-3: For each token1, get top-M token2, then top-N token3
print(f"\nStep 2-3: Checking all 3-token combinations...")

all_3token_phrases = []
total_checks = 0

for i, (token1_id, log_prob1) in enumerate(tqdm(zip(top_k_token1_ids, top_k_log_probs1), 
                                                   total=len(top_k_token1_ids),
                                                   desc="Processing token1")):
    
    # Construct new prompt with token1
    prompt_with_token1 = prompt + tokenizer.decode([token1_id])
    
    # Get logits for token2
    inputs = tokenizer(prompt_with_token1, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits2 = outputs.logits[0, -1, :]
    
    # Get top-M word-starting tokens for token2
    word_start_logits2 = logits2[word_start_tokens]
    log_probs2 = torch.log_softmax(word_start_logits2, dim=0)
    top_m_idx = torch.topk(log_probs2, k=min(TOP_M_TOKEN2, len(word_start_tokens))).indices
    
    # For each token2, get top-N token3
    for idx2 in top_m_idx:
        token2_id = word_start_tokens[idx2.item()]
        log_prob2 = log_probs2[idx2].item()
        
        # Construct prompt with token1+token2
        prompt_with_token12 = prompt_with_token1 + tokenizer.decode([token2_id])
        
        # Get logits for token3
        inputs = tokenizer(prompt_with_token12, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits3 = outputs.logits[0, -1, :]
        
        # Get top-N word-starting tokens for token3
        word_start_logits3 = logits3[word_start_tokens]
        log_probs3 = torch.log_softmax(word_start_logits3, dim=0)
        top_n_idx = torch.topk(log_probs3, k=min(TOP_N_TOKEN3, len(word_start_tokens))).indices
        
        # Check each 3-token combination
        for idx3 in top_n_idx:
            token3_id = word_start_tokens[idx3.item()]
            log_prob3 = log_probs3[idx3].item()
            
            # Decode the 3-token phrase
            phrase = tokenizer.decode([token1_id, token2_id, token3_id]).strip()
            total_checks += 1
            
            # Check if it's a noun phrase
            if is_noun_phrase(phrase, nlp):
                # Compute joint log probability (sum in log space)
                joint_log_prob = log_prob1 + log_prob2 + log_prob3
                
                all_3token_phrases.append({
                    'noun1': NOUN1,
                    'predicted_hypernym': phrase,
                    'num_tokens': 3,
                    'token1': tokenizer.decode([token1_id]).strip(),
                    'token2': tokenizer.decode([token2_id]).strip(),
                    'token3': tokenizer.decode([token3_id]).strip(),
                    'log_prob': joint_log_prob,
                    'probability': np.exp(joint_log_prob),
                    'token1_id': token1_id,
                    'token2_id': token2_id,
                    'token3_id': token3_id
                })

print(f"\n✓ Checked {total_checks:,} 3-token combinations")
print(f"✓ Found {len(all_3token_phrases):,} valid 3-token noun phrases")

# Sort by probability
all_3token_phrases.sort(key=lambda x: x['log_prob'], reverse=True)

# Add rank
for rank, item in enumerate(all_3token_phrases, 1):
    item['rank'] = rank

# Save to CSV
df = pd.DataFrame(all_3token_phrases)
output_csv = OUTPUT_DIR / f"{NOUN1}_3token_noun_predictions.csv"
df.to_csv(output_csv, index=False)

print(f"\n{'='*60}")
print(f"✓ Saved {len(df)} 3-token noun phrases to {output_csv}")
print(f"{'='*60}")

# Print statistics
print(f"\nSummary:")
print(f"  Total 3-token noun phrases found: {len(df)}")
print(f"  Total combinations checked: {total_checks:,}")
print(f"  Noun phrase rate: {100 * len(df) / total_checks:.1f}%")

# Print top 30
if len(df) > 0:
    print(f"\nTop 30 3-token noun phrases for '{NOUN1}':")
    print(f"{'Rank':<6} {'Phrase':<40} {'Log Prob':<12} {'Probability':<12}")
    print("-" * 75)
    for _, row in df.head(30).iterrows():
        print(f"{row['rank']:<6} {row['predicted_hypernym']:<40} {row['log_prob']:<12.4f} {row['probability']:<12.6f}")

print(f"\n✅ DONE!")
