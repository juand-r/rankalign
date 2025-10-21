"""
Add wordfreq columns to merged data CSV
"""

import pandas as pd
import numpy as np
from pathlib import Path
from wordfreq import zipf_frequency

def add_wordfreq_to_csv(csv_path):
    """Add log wordfreq columns for noun1 and noun2"""
    print(f"\nProcessing: {csv_path}")
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows")
    
    # Compute zipf frequencies (already in log scale)
    # zipf_frequency returns log10(freq per billion words) + 3
    # Range is roughly 0-8, where 8 is very common, 0 is very rare
    print("  Computing wordfreq for noun1...")
    df['log_wordfreq_noun1'] = df['noun1'].apply(lambda x: zipf_frequency(x.lower(), 'en'))
    
    print("  Computing wordfreq for noun2...")
    df['log_wordfreq_noun2'] = df['noun2'].apply(lambda x: zipf_frequency(x.lower(), 'en'))
    
    # Save back to same file
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved with wordfreq columns")
    
    # Print summary
    print(f"\n  Summary statistics:")
    print(f"    log_wordfreq_noun1: min={df['log_wordfreq_noun1'].min():.2f}, max={df['log_wordfreq_noun1'].max():.2f}, mean={df['log_wordfreq_noun1'].mean():.2f}")
    print(f"    log_wordfreq_noun2: min={df['log_wordfreq_noun2'].min():.2f}, max={df['log_wordfreq_noun2'].max():.2f}, mean={df['log_wordfreq_noun2'].mean():.2f}")

if __name__ == '__main__':
    # Find all merged data files
    data_files = list(Path('.').glob('merged_data_*.csv'))
    
    if not data_files:
        print("No merged_data_*.csv files found!")
        exit(1)
    
    print(f"Found {len(data_files)} file(s) to process:")
    for f in data_files:
        print(f"  - {f}")
    
    # Process each file
    for csv_file in data_files:
        add_wordfreq_to_csv(csv_file)
    
    print("\n✅ All files processed!")

