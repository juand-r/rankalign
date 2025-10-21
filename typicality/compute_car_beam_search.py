"""
Get noun phrase predictions for noun1="cars" using beam search
Generates complete multi-token sequences
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
NUM_BEAMS = 2000  # Number of beams to track
NUM_RETURN_SEQUENCES = 2000  # Number of sequences to return
MAX_NEW_TOKENS = 5  # Maximum tokens to generate
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("hypernym_predictions")

print(f"Device: {DEVICE}")
print(f"Beam search: {NUM_BEAMS} beams, returning {NUM_RETURN_SEQUENCES} sequences")
print(f"Max tokens: {MAX_NEW_TOKENS}")
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


# Generate completions using beam search
prompt = f"Complete the sentence: {NOUN1} are a kind of"
print(f"\nPrompt: '{prompt}'")
print(f"\nGenerating {NUM_RETURN_SEQUENCES} completions using beam search...")

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print(f"Running beam search (this may take a few minutes)...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=NUM_BEAMS,
        num_return_sequences=NUM_RETURN_SEQUENCES,
        output_scores=True,
        return_dict_in_generate=True,
        early_stopping=False,  # Don't stop at EOS, let it generate full sequences
        do_sample=False,  # Deterministic beam search
        pad_token_id=tokenizer.pad_token_id,
    )

print(f"✓ Generated {len(outputs.sequences)} sequences")

# Process the generated sequences
results = []
prompt_length = inputs['input_ids'].shape[1]
rejected_sequences = []

for i, sequence in enumerate(tqdm(outputs.sequences, desc="Processing sequences")):
    # Extract only the generated part (after prompt)
    generated_ids = sequence[prompt_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # Remove everything after first punctuation or newline
    for punct in ['.', ',', '!', '?', ';', ':', '\n', '(', ')']:
        if punct in generated_text:
            generated_text = generated_text.split(punct)[0].strip()
    
    # Skip if empty
    if not generated_text:
        rejected_sequences.append(("EMPTY", ""))
        continue
    
    # Check if it's a noun phrase
    if is_noun_phrase(generated_text, nlp):
        # Count tokens in the ACTUAL phrase (after filtering)
        actual_tokens = tokenizer.encode(generated_text, add_special_tokens=False)
        num_tokens = len(actual_tokens)
        
        # Compute the sequence score (log probability)
        # The scores are returned as a tuple of tensors, one per generation step
        if hasattr(outputs, 'sequences_scores'):
            # Use the sequence score if available
            log_prob = outputs.sequences_scores[i].item()
        else:
            # Otherwise estimate from the transition scores
            # This is the sum of log probabilities for each token
            log_prob = 0.0
            if hasattr(outputs, 'scores') and outputs.scores:
                for step_idx, step_scores in enumerate(outputs.scores):
                    if step_idx < len(generated_ids):
                        token_id = generated_ids[step_idx]
                        # Get the log prob for this token
                        log_probs = torch.log_softmax(step_scores[i], dim=0)
                        log_prob += log_probs[token_id].item()
        
        results.append({
            'noun1': NOUN1,
            'predicted_hypernym': generated_text,
            'num_tokens': num_tokens,
            'log_prob': log_prob,
            'probability': np.exp(log_prob),
            'generated_ids': generated_ids.tolist()
        })
    else:
        # Rejected: not a noun phrase
        rejected_sequences.append(("NOT_NOUN", generated_text))

print(f"\n✓ Found {len(results)} valid noun phrases")
print(f"✓ Rejected {len(rejected_sequences)} sequences")

# Remove duplicates (keep highest probability)
seen = {}
duplicates = []
for r in results:
    phrase = r['predicted_hypernym']
    if phrase not in seen or r['log_prob'] > seen[phrase]['log_prob']:
        if phrase in seen:
            duplicates.append(phrase)
        seen[phrase] = r
    else:
        duplicates.append(phrase)

results = list(seen.values())
print(f"✓ After deduplication: {len(results)} unique noun phrases")
print(f"✓ Removed {len(duplicates)} duplicates")

# Sort by probability
results.sort(key=lambda x: x['log_prob'], reverse=True)

# Add rank
for rank, item in enumerate(results, 1):
    item['rank'] = rank

# Save to CSV
df = pd.DataFrame(results)
output_csv = OUTPUT_DIR / f"{NOUN1}_beam_search_predictions.csv"
df.to_csv(output_csv, index=False)

print(f"\n{'='*60}")
print(f"✓ Saved {len(df)} noun phrases to {output_csv}")
print(f"{'='*60}")

# Print statistics
print(f"\nSummary:")
print(f"  Total unique noun phrases found: {len(df)}")
print(f"  Average tokens per phrase: {df['num_tokens'].mean():.1f}")
print(f"  Token distribution:")
for n in sorted(df['num_tokens'].unique()):
    count = (df['num_tokens'] == n).sum()
    print(f"    {n} tokens: {count} phrases ({100*count/len(df):.1f}%)")

# Print top 30
if len(df) > 0:
    print(f"\nTop 30 noun phrases for '{NOUN1}' (beam search):")
    print(f"{'Rank':<6} {'#Tok':<5} {'Phrase':<45} {'Log Prob':<12} {'Probability':<12}")
    print("-" * 85)
    for _, row in df.head(30).iterrows():
        print(f"{row['rank']:<6} {row['num_tokens']:<5} {row['predicted_hypernym']:<45} {row['log_prob']:<12.4f} {row['probability']:<12.6f}")

# Print rejected sequences
if rejected_sequences:
    print(f"\n{'='*60}")
    print(f"REJECTED SEQUENCES (not noun phrases or empty):")
    print(f"{'='*60}")
    for reason, text in rejected_sequences[:50]:  # Show first 50
        if reason == "EMPTY":
            print(f"  {reason}: [empty after punctuation removal]")
        else:
            print(f"  {reason}: '{text}'")
    if len(rejected_sequences) > 50:
        print(f"  ... and {len(rejected_sequences) - 50} more")

# Print some duplicate examples
if duplicates:
    print(f"\n{'='*60}")
    print(f"DUPLICATE PHRASES (examples):")
    print(f"{'='*60}")
    unique_dups = list(set(duplicates))[:20]
    for dup in unique_dups:
        count = duplicates.count(dup) + 1  # +1 for the one we kept
        print(f"  '{dup}' appeared {count} times")
    if len(unique_dups) < len(set(duplicates)):
        print(f"  ... and {len(set(duplicates)) - len(unique_dups)} more unique duplicates")

print(f"\n✅ DONE!")

