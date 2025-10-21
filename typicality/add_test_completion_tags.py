"""
Add source tags ONLY for test_completions to the combined predictions CSV.
All others will remain untagged (or be tagged as 'other').
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Load the combined data
combined_file = Path('hypernym_predictions/cars_combined_clean_predictions_with_validator.csv')
df_combined = pd.read_csv(combined_file)
print(f"Loaded combined file: {len(df_combined)} rows")

# Initialize source column with 'other'
df_combined['source'] = 'other'

# Load test completions
test_file = Path('hypernym_predictions/cars_test_completions.csv')
if test_file.exists():
    print(f"\nProcessing test_completions...")
    df_test = pd.read_csv(test_file)
    print(f"  Loaded {len(df_test)} test completions")
    print(f"  Columns: {df_test.columns.tolist()}")
    
    # Standardize column name
    if 'noun2' in df_test.columns:
        hypernym_col = 'noun2'
    elif 'predicted_hypernym' in df_test.columns:
        hypernym_col = 'predicted_hypernym'
    elif 'completion' in df_test.columns:
        hypernym_col = 'completion'
    else:
        print(f"  Warning: No hypernym column found")
        hypernym_col = None
    
    if hypernym_col:
        # Match based on noun1 and predicted_hypernym
        for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Matching test completions"):
            mask = (
                (df_combined['noun1'] == row['noun1']) &
                (df_combined['predicted_hypernym'] == row[hypernym_col])
            )
            
            num_matches = mask.sum()
            if num_matches > 0:
                df_combined.loc[mask, 'source'] = 'test_completions'
        
        tagged = (df_combined['source'] == 'test_completions').sum()
        print(f"  Tagged {tagged} rows as test_completions")

# Save the updated file
output_file = combined_file  # Overwrite the original
df_combined.to_csv(output_file, index=False)
print(f"\n✅ Saved updated file with source tags to {output_file}")

# Print summary statistics
print("\nSource distribution:")
print(df_combined['source'].value_counts())

