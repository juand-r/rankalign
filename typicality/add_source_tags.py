"""
Add source tags to the combined predictions CSV based on which original file each prediction came from.
"""

import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Load the combined data
combined_file = Path('hypernym_predictions/cars_combined_clean_predictions_with_validator.csv')
df_combined = pd.read_csv(combined_file)
print(f"Loaded combined file: {len(df_combined)} rows")

# Load individual source files
base_dir = Path('hypernym_predictions')

sources = {
    'single_token': base_dir / 'cars_all_noun_predictions.csv',
    '2_token': base_dir / 'cars_2token_noun_predictions.csv',
    'beam_search': base_dir / 'cars_beam_search_predictions.csv',
    'test_completions': base_dir / 'cars_test_completions.csv',
}

# Initialize source column
df_combined['source'] = None

# For each source, mark matching rows
for source_name, source_file in tqdm(sources.items()):
    if source_file.exists():
        print(f"\nProcessing {source_name}...")
        df_source = pd.read_csv(source_file)
        
        # Standardize column name
        if 'noun2' in df_source.columns:
            hypernym_col = 'noun2'
        elif 'predicted_hypernym' in df_source.columns:
            hypernym_col = 'predicted_hypernym'
        else:
            print(f"  Warning: No hypernym column found in {source_file}")
            continue
        
        # Match based on noun1, predicted_hypernym, and log_prob
        for _, row in df_source.iterrows():
            mask = (
                (df_combined['noun1'] == row['noun1']) &
                (df_combined['predicted_hypernym'] == row[hypernym_col]) &
                (abs(df_combined['log_prob'] - row['log_prob']) < 0.0001)
            )
            
            num_matches = mask.sum()
            if num_matches > 0:
                # Tag these rows with source
                df_combined.loc[mask, 'source'] = source_name
        
        tagged = (df_combined['source'] == source_name).sum()
        print(f"  Tagged {tagged} rows as {source_name}")

# Check for untagged rows
untagged = df_combined['source'].isna().sum()
print(f"\nUntagged rows: {untagged}")

if untagged > 0:
    print("\nSample of untagged rows:")
    print(df_combined[df_combined['source'].isna()].head())

# Save the updated file
output_file = combined_file  # Overwrite the original
df_combined.to_csv(output_file, index=False)
print(f"\n✅ Saved updated file with source tags to {output_file}")

# Print summary statistics
print("\nSource distribution:")
print(df_combined['source'].value_counts())

