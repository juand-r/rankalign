"""
Combine and clean all "cars" prediction CSV files
Removes garbage (HTML tags, underscores, etc.) and merges all sources
"""

import pandas as pd
from pathlib import Path
import re

# Configuration
INPUT_DIR = Path("hypernym_predictions")
OUTPUT_CSV = INPUT_DIR / "cars_combined_clean_predictions.csv"
GARBAGE_CSV = INPUT_DIR / "cars_combined_garbage.csv"

# Input files
INPUT_FILES = [
    "cars_all_noun_predictions.csv",       # 1-token
    "cars_2token_noun_predictions.csv",    # 2-token
    "cars_beam_search_predictions.csv",    # Beam search (variable length)
    "cars_test_completions.csv",           # Hand-crafted test completions
]

print("Loading and cleaning prediction files...")

# Load all dataframes
all_predictions = []

for filename in INPUT_FILES:
    filepath = INPUT_DIR / filename
    if not filepath.exists():
        print(f"  ⚠️  Skipping {filename} (file not found)")
        continue
    
    print(f"  Loading {filename}...")
    df = pd.read_csv(filepath)
    
    # Handle different column names (test_completions uses 'completion' instead of 'predicted_hypernym')
    if 'completion' in df.columns and 'predicted_hypernym' not in df.columns:
        df = df.rename(columns={'completion': 'predicted_hypernym'})
    
    # Extract only the columns we need
    if 'noun1' in df.columns and 'predicted_hypernym' in df.columns and 'log_prob' in df.columns:
        df_subset = df[['noun1', 'predicted_hypernym', 'log_prob']].copy()
        all_predictions.append(df_subset)
        print(f"    ✓ Loaded {len(df_subset)} predictions")
    else:
        print(f"    ⚠️  Skipping {filename} (missing required columns)")

if not all_predictions:
    print("\n❌ No valid prediction files found!")
    exit(1)

# Combine all predictions
print(f"\nCombining {len(all_predictions)} files...")
combined_df = pd.concat(all_predictions, ignore_index=True)
print(f"  ✓ Total predictions: {len(combined_df)}")

# Define garbage patterns to filter out
def is_garbage(text):
    """Check if a predicted hypernym is garbage"""
    text = str(text).strip()
    
    # Empty or too short
    if len(text) < 2:
        return True
    
    # Contains any underscore
    if '_' in text:
        return True
    
    # Contains problematic characters
    if any(char in text for char in ['=', ':', '(', '[']):
        return True
    
    # HTML tags
    if re.search(r'<[^>]+>', text):  # Contains HTML tags
        return True
    
    # Starts with punctuation or special chars
    if text[0] in '<>[]{}()_-=+*&^%$#@!~`':
        return True
    
    # Contains "Complete the sentence" or similar prompt repetitions
    if 'Complete the sentence' in text or 'sentence' in text.lower():
        return True
    
    # Math/XML namespaces
    if 'xmlns' in text or 'http://' in text:
        return True
    
    # All non-alphabetic (except spaces)
    if not any(c.isalpha() for c in text):
        return True
    
    return False

# Filter out garbage
print("\nFiltering out garbage predictions...")
before_count = len(combined_df)
garbage_mask = combined_df['predicted_hypernym'].apply(is_garbage)
garbage_df = combined_df[garbage_mask].copy()
combined_df = combined_df[~garbage_mask]
after_count = len(combined_df)
print(f"  ✓ Removed {before_count - after_count} garbage predictions")
print(f"  ✓ Remaining: {after_count} predictions")

# Save garbage to separate file
if len(garbage_df) > 0:
    garbage_df = garbage_df.sort_values('log_prob', ascending=False).reset_index(drop=True)
    garbage_df.to_csv(GARBAGE_CSV, index=False)
    print(f"  ✓ Saved {len(garbage_df)} garbage predictions to {GARBAGE_CSV.name}")

# Remove duplicates (keep the one with highest log_prob)
print("\nRemoving duplicates (keeping highest log_prob)...")
before_count = len(combined_df)
combined_df = combined_df.sort_values('log_prob', ascending=False)
combined_df = combined_df.drop_duplicates(subset=['noun1', 'predicted_hypernym'], keep='first')
after_count = len(combined_df)
print(f"  ✓ Removed {before_count - after_count} duplicates")
print(f"  ✓ Unique predictions: {after_count}")

# Sort by log_prob (descending - highest at top)
combined_df = combined_df.sort_values('log_prob', ascending=False).reset_index(drop=True)

# Save to CSV
combined_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n{'='*60}")
print(f"✓ Saved {len(combined_df)} clean predictions to:")
print(f"  {OUTPUT_CSV}")
if len(garbage_df) > 0:
    print(f"✓ Saved {len(garbage_df)} garbage predictions to:")
    print(f"  {GARBAGE_CSV}")
print(f"{'='*60}")

# Print statistics
print(f"\nSummary Statistics:")
print(f"  Total unique predictions: {len(combined_df)}")
print(f"  Highest log_prob: {combined_df['log_prob'].max():.4f}")
print(f"  Lowest log_prob: {combined_df['log_prob'].min():.4f}")
print(f"  Mean log_prob: {combined_df['log_prob'].mean():.4f}")

# Print top 30
print(f"\nTop 30 predictions for 'cars':")
print(f"{'Rank':<6} {'Predicted Hypernym':<40} {'Log Prob':<12}")
print("-" * 60)
for i, row in combined_df.head(30).iterrows():
    print(f"{i+1:<6} {row['predicted_hypernym']:<40} {row['log_prob']:<12.4f}")

print(f"\n✅ DONE!")

